"""Guarded normal-MMA TMEM packing/restore diagnostic, not MLA timing.

Two M64xN256 results occupy complementary Layout-F datapath halves in a
256-column allocation. Save the upper-column M64xN128 half in registers, overwrite it
with a score product, read scores and both64-key maxima, restore the half, then accumulate into
both original outputs. Exact integer-valued FP8 inputs make every reference
product and the final factor-of-two result exactly representable in FP32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__' and '--help' not in sys.argv and '-h' not in sys.argv:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--output-dir', type=Path)
    initial, _ = bootstrap.parse_known_args()
    if initial.output_dir is not None:
        directory = initial.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        os.chdir(directory)
        sys.argv += ['--output-dir', str(directory)]

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from kernel_v197 import tmem_before_sync, tmem_after_sync

GPU_UUID = 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'


class HalfPlaneProbe:
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, output: cute.Tensor,
                 scores: cute.Tensor, stream: cuda.CUstream):
        fp8 = cutlass.Float8E4M3FN
        wide = bw.make_trivial_tiled_mma(fp8, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, 256))
        narrow = bw.make_trivial_tiled_mma(fp8, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, 128))
        atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW32, fp8)
        alayout = cute.tile_to_shape(atom, (64, 32), order=(0, 1))
        blayout = cute.tile_to_shape(atom, (512, 32), order=(0, 1))
        self.kernel(a, b, output, scores, wide, narrow, alayout, blayout).launch(
            grid=(output.shape[0], 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, output: cute.Tensor,
               scores: cute.Tensor, wide: cute.TiledMma, narrow: cute.TiledMma,
               alayout: cute.ComposedLayout, blayout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        alloc = utils.SmemAllocator()
        sa = alloc.allocate_tensor(cutlass.Float8E4M3FN, alayout.outer,
                                   byte_alignment=128, swizzle=alayout.inner)
        sb = alloc.allocate_tensor(cutlass.Float8E4M3FN, blayout.outer,
                                   byte_alignment=128, swizzle=blayout.inner)
        barrier = alloc.allocate_array(cutlass.Int64, 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        if tid == 0:
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        for i in cutlass.range(tid, 64 * 32, 128):
            sa[i // 32, i % 32] = a[i // 32, i % 32]
        for i in cutlass.range(tid, 512 * 32, 128):
            sb[i // 32, i % 32] = b[i // 32, i % 32]
        if warp == 0:
            cute.arch.alloc_tmem(256, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                        ptr_to_buffer_holding_addr=holding)
        ac = wide.make_fragment_A(wide.get_slice(0).partition_A(sa))
        co = wide.make_fragment_C(wide.partition_shape_C((64, 256)))
        cs = narrow.make_fragment_C(narrow.partition_shape_C((64, 128)))
        assert tcgen05.find_tmem_tensor_col_offset(co) == 256
        assert tcgen05.find_tmem_tensor_col_offset(cs) == 128
        low = cute.make_tensor(tp, co.layout)
        high = cute.make_tensor(tp + (16 << 16), co.layout)
        score = cute.make_tensor(tp + (16 << 16) + 128, cs.layout)
        high2 = high[((None, None), 0, 0)]
        chunk_layout = cute.make_layout((high2.shape[0], 128),
                                       stride=(high2.stride[0], high2.stride[1]))
        saved_half = cute.make_tensor(high.iterator + 128, chunk_layout)
        load = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(64)),
            cutlass.Float32), saved_half)
        store = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(64)),
            cutlass.Float32), saved_half)
        reduce_load = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.LdRed16x32bx2Op(tcgen05.copy.Repetition(64),
                redOp=tcgen05.TmemLoadRedOp.MAX, nan=True, half_split_off=64),
            cutlass.Float32), saved_half)
        lane = load.get_slice(tid)
        coords = lane.partition_D(cute.make_identity_tensor((64, 128)))
        saved = cute.make_fragment_like(coords, cutlass.Float32)
        fragment = cute.make_fragment_like(coords, cutlass.Float32)
        loaded_max = cute.make_rmem_tensor(
            cute.make_layout((1, coords.shape[1], coords.shape[2])), cutlass.Float32)

        if warp == 0:
            wide.set(tcgen05.Field.ACCUMULATE, False)
            for plane in cutlass.range(2, unroll_full=True):
                tile = cute.local_tile(sb, (256, 32), (plane, 0))
                bc = wide.make_fragment_B(wide.get_slice(0).partition_B(tile))
                dest = cute.make_tensor(tp + plane * (16 << 16), co.layout)
                for k in cutlass.range(cute.size(ac, mode=[2]), unroll_full=True):
                    cute.gemm(wide, dest, ac[None, None, k], bc[None, None, k], dest)
            with cute.arch.elect_one():
                tcgen05.commit(barrier)
        cute.arch.mbarrier_wait(barrier, 0)
        tmem_after_sync()
        cute.copy(load, lane.partition_S(saved_half), saved)
        cute.arch.fence_view_async_tmem_load()
        tmem_before_sync()
        cute.arch.barrier()
        tmem_after_sync()

        if warp == 0:
            an = narrow.make_fragment_A(narrow.get_slice(0).partition_A(sa))
            tile = cute.local_tile(sb, (128, 32), (0, 0))
            bn = narrow.make_fragment_B(narrow.get_slice(0).partition_B(tile))
            narrow.set(tcgen05.Field.ACCUMULATE, False)
            for k in cutlass.range(cute.size(an, mode=[2]), unroll_full=True):
                cute.gemm(narrow, score, an[None, None, k], bn[None, None, k], score)
            with cute.arch.elect_one():
                tcgen05.commit(barrier)
        cute.arch.mbarrier_wait(barrier, 1)
        tmem_after_sync()
        cute.copy(reduce_load, reduce_load.get_slice(tid).partition_S(saved_half),
                  (fragment, loaded_max))
        cute.arch.fence_view_async_tmem_load()
        for j in cutlass.range(cute.size(fragment), unroll_full=True):
            row, col = coords[j]
            scores[block, row, col] = fragment[j]
        head = coords[0][0]
        scores[block, head, 128 + tid % 32 // 16] = loaded_max[0]
        cute.copy(store, saved, store.get_slice(tid).partition_D(saved_half))
        cute.arch.fence_view_async_tmem_store()
        tmem_before_sync()
        cute.arch.barrier()
        tmem_after_sync()

        if warp == 0:
            wide.set(tcgen05.Field.ACCUMULATE, True)
            for plane in cutlass.range(2, unroll_full=True):
                tile = cute.local_tile(sb, (256, 32), (plane, 0))
                bc = wide.make_fragment_B(wide.get_slice(0).partition_B(tile))
                dest = cute.make_tensor(tp + plane * (16 << 16), co.layout)
                for k in cutlass.range(cute.size(ac, mode=[2]), unroll_full=True):
                    cute.gemm(wide, dest, ac[None, None, k], bc[None, None, k], dest)
            with cute.arch.elect_one():
                tcgen05.commit(barrier)
        cute.arch.mbarrier_wait(barrier, 0)
        tmem_after_sync()
        for plane in cutlass.range(2, unroll_full=True):
            for chunk in cutlass.range(2, unroll_full=True):
                tile = cute.make_tensor(tp + plane * (16 << 16) + chunk * 128, chunk_layout)
                cute.copy(load, lane.partition_S(tile), fragment)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(fragment), unroll_full=True):
                    row, col = coords[j]
                    output[block, row, plane * 256 + chunk * 128 + col] = fragment[j]
        tmem_before_sync()
        cute.arch.barrier()
        tmem_after_sync()
        if warp == 0:
            cute.arch.dealloc_tmem(tp, 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, choices=(1, 296), default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        parser.error('select the authorized GPU1 UUID')
    if not 1 <= args.repeats <= 10:
        parser.error('repeats must be1..10')
    outdir = args.output_dir.resolve()
    if (outdir / 'result.json').exists():
        parser.error('use a fresh output directory')
    generator = torch.Generator(device='cpu').manual_seed(9187)
    ah = torch.randint(-2, 3, (64, 32), generator=generator).float()
    bh = torch.randint(-2, 3, (512, 32), generator=generator).float()
    a, b = [x.to(device='cuda', dtype=torch.float8_e4m3fn) for x in (ah, bh)]
    output = torch.empty((args.blocks, 64, 512), device='cuda', dtype=torch.float32)
    scores = torch.empty((args.blocks, 64, 130), device='cuda', dtype=torch.float32)
    tensors = [from_dlpack(x, assumed_align=16) for x in (a, b, output, scores)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    options = '--gpu-arch sm_103a --keep-cubin --keep-ptx --ptxas-options=-g-tmem-access-check'
    compiled = cute.compile(HalfPlaneProbe(), *tensors, stream, options=options)
    cubins = list(outdir.glob('*.cubin'))
    assert len(cubins) == 1, cubins
    for flag, name in [('--dump-resource-usage', 'resources.txt'), ('--dump-sass', 'sass.txt')]:
        (outdir / name).write_text(subprocess.check_output(['cuobjdump', flag, str(cubins[0])], text=True))
    report = {'purpose': __doc__, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'gpu_uuid': GPU_UUID, 'blocks': args.blocks, 'guardrails': True,
              'compile_only': args.compile_only, 'tmem_columns': 256, 'records': []}
    if not args.compile_only:
        reference = (ah @ bh.T) * 2
        raw_scores = ah @ bh[:128].T
        score_reference = torch.cat((raw_scores,
            raw_scores.reshape(64, 2, 64).max(-1).values), dim=1)
        for repeat in range(args.repeats):
            output.fill_(float('nan'))
            scores.fill_(float('nan'))
            compiled(*tensors, stream)
            torch.cuda.synchronize()
            actual, actual_scores = output.cpu(), scores.cpu()
            mismatches = int((actual != reference.unsqueeze(0)).sum())
            score_mismatches = int((actual_scores != score_reference.unsqueeze(0)).sum())
            report['records'].append({'repeat': repeat, 'output_mismatches': mismatches,
                                      'score_mismatches': score_mismatches,
                                      'output_elements': output.numel(), 'score_elements': scores.numel()})
            assert mismatches == score_mismatches == 0, report['records'][-1]
    (outdir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
