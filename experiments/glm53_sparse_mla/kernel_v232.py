"""Iteration232: normal-M64,128-key stages,one KV buffer and256 TMEM columns.

Keep one full-output CTA and reuse its completed KV tail as probabilities.
Save the second output plane's upper128 columns in64 FP32 values/thread
before QK overwrites them. This trades intra-CTA KV overlap for the resource
budget of two CTAs per SM; runtime benefit is not assumed.
"""
import math
import os
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu.tcgen05.helpers import smem_descriptor_to_int


@dsl_user_op
def tmem_before_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tmem_after_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def gather4(smem, tensor_map, col, i0, i1, i2, i3, barrier, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int32(smem.toint()).ir_value(loc=loc, ip=ip),
         cutlass.Int64(tensor_map.toint()).ir_value(loc=loc, ip=ip),
         *[cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (col, i0, i1, i2, i3)],
         cutlass.Int32(barrier.toint()).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1 "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7];",
        "r,l,r,r,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def store_output256(pointer, values, *, loc=None, ip=None):
    # One naturally aligned 32-byte sector; the runner verifies output alignment.
    llvm.inline_asm(None,
        [cutlass.Int64(pointer.toint()).ir_value(loc=loc, ip=ip),
         *[cutlass.Uint32(values[j]).ir_value(loc=loc, ip=ip) for j in range(8)]],
        "st.global.L1::no_allocate.L2::evict_first.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


TENSOR_MAP_BYTES = 512


@dsl_user_op
def load_q_tile(smem, tensor_map, col, row, barrier, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int32(smem.toint()).ir_value(loc=loc, ip=ip),
         cutlass.Int64(tensor_map.toint()).ir_value(loc=loc, ip=ip),
         cutlass.Int32(col).ir_value(loc=loc, ip=ip),
         cutlass.Int32(row).ir_value(loc=loc, ip=ip),
         cutlass.Int32(barrier.toint()).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 "
        "[$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def make_kv_map(kv, q):
    storage = torch.empty(TENSOR_MAP_BYTES, dtype=torch.uint8, device=kv.device)
    specs = []
    for tensor, tile_rows in ((kv, 1), (q.view(-1, 576), 64)):
        for offset, width, box, swizzle in (
            (0, 576, 128, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B),
            (512, 64, 64, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B)):
            specs.append((tensor, tile_rows, offset, width, box, swizzle))
    for part, (tensor, tile_rows, offset, width, box, swizzle) in enumerate(specs):
        status, desc = cuda.cuTensorMapEncodeTiled(
            cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, tensor.data_ptr() + offset,
            [cuda.cuuint64_t(width), cuda.cuuint64_t(tensor.shape[0])], [cuda.cuuint64_t(576)],
            [cuda.cuuint32_t(box), cuda.cuuint32_t(tile_rows)], [cuda.cuuint32_t(1), cuda.cuuint32_t(1)],
            cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
            cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuTensorMapEncodeTiled part {part}: {status}")
        status, = cuda.cuMemcpyHtoD(storage.data_ptr() + part * 128, desc.getPtr(), 128)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD tensor map: {status}")
    return storage


@dsl_user_op
def mma_normal(d, a, b, n, b_transpose, accumulate, *, loc=None, ip=None):
    adesc = smem_descriptor_to_int(tcgen05.make_umma_smem_desc(
        a.iterator, cute.make_layout((a.layout.shape, 1), stride=(a.layout.stride, 0)), OperandMajorMode.K._to_ir(), loc=loc, ip=ip), loc=loc, ip=ip)
    bmajor = OperandMajorMode.MN if b_transpose else OperandMajorMode.K
    bdesc = smem_descriptor_to_int(tcgen05.make_umma_smem_desc(
        b.iterator, cute.make_layout((b.layout.shape, 1), stride=(b.layout.stride, 0)), bmajor._to_ir(), loc=loc, ip=ip), loc=loc, ip=ip)
    # PTX descriptor: FP32 accumulator, E4M3 A/B, M64, requested N.
    desc = (1 << 4) | ((n // 8) << 17) | (4 << 24) | (int(b_transpose) << 16)
    llvm.inline_asm(None,
        [cutlass.Int32(d.toint()).ir_value(loc=loc, ip=ip),
         adesc.ir_value(loc=loc, ip=ip), bdesc.ir_value(loc=loc, ip=ip),
         cutlass.Int32(desc).ir_value(loc=loc, ip=ip),
         cutlass.Int32(accumulate).ir_value(loc=loc, ip=ip),
         cutlass.Int32(0).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, {$5, $5, $5, $5}, p; }",
        "r,l,l,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)



@dsl_user_op
def mask_reduce32(scores, bits, rowmax, fullmax, all_full, *, loc=None, ip=None):
    # Called by complete compute warps. all_full tests both32-key halves
    # and is warp-uniform; bits may differ between the two half-warps.
    # Tied inputs preserve scores/fullmax; early-clobber keeps late-read bits
    # and rowmax separate from score outputs modified earlier in the assembly.
    lines = ["{ .reg .pred skip, hole; .reg .b32 bit, choice; .reg .f32 m<16>;",
             "MASK_TARGETS: .branchtargets MASK_DONE, MASK_FALLBACK;",
             "setp.ne.u32 skip, $68, 0;",
             "selp.u32 choice, 0, 1, skip;",
             "brx.idx.uni choice, MASK_TARGETS; MASK_FALLBACK:"]
    for j in range(32):
        lines.extend([f"and.b32 bit, $66, {1 << j};",
                      "setp.eq.u32 hole, bit, 0;",
                      f"@hole mov.b32 ${j}, 0xf149f2ca;"])
    pending = [f"${j}" for j in range(32)] + ["$67"]
    for j in range(16):
        values = pending[:3]
        pending = pending[3:] + [f"m{j}"]
        lines.append(f"max.NaN.f32 m{j}, {', '.join(values)};")
    assert len(pending) == 1
    lines.extend([f"mov.f32 $32, {pending[0]};", "MASK_DONE: }"])
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 33),
        [*[cutlass.Float32(scores[j]).ir_value(loc=loc, ip=ip) for j in range(32)],
         cutlass.Float32(fullmax).ir_value(loc=loc, ip=ip),
         cutlass.Uint32(bits).ir_value(loc=loc, ip=ip),
         cutlass.Float32(rowmax).ir_value(loc=loc, ip=ip),
         cutlass.Int32(all_full).ir_value(loc=loc, ip=ip)],
        "\n".join(lines),
        ",".join(["=&f"] * 33 + [str(j) for j in range(33)] + ["r", "f", "r"]),
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    for j in range(32):
        scores[j] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [j], loc=loc, ip=ip))
    return cutlass.Float32(llvm.extractvalue(T.f32(), result, [32], loc=loc, ip=ip))


class SparseMLA:
    def __init__(self, block_k=128):
        assert block_k == 128, "v232 requires --block-k 128"
        self.block_k = block_k

    @cute.jit
    def __call__(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
                 lens: cute.Tensor, out: cute.Tensor, tensor_map: cute.Tensor, stream: cuda.CUstream):
        fp8 = cutlass.Float8E4M3FN
        qk = bw.make_trivial_tiled_mma(fp8, OperandMajorMode.K,
                                      OperandMajorMode.K, cutlass.Float32,
                                      tcgen05.CtaGroup.ONE, (64, self.block_k))
        pv = bw.make_trivial_tiled_mma(fp8, OperandMajorMode.K,
                                      OperandMajorMode.MN, cutlass.Float32,
                                      tcgen05.CtaGroup.ONE, (64, 256))
        atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, fp8)
        tail_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, fp8)
        qlayout = cute.tile_to_shape(atom, (64, 512), order=(0, 1))
        klayout = cute.tile_to_shape(atom, (self.block_k, 512), order=(0, 1))
        qtail_layout = cute.tile_to_shape(tail_atom, (64, 64), order=(0, 1))
        ktail_layout = cute.tile_to_shape(tail_atom, (self.block_k, 64), order=(0, 1))
        playout = cute.tile_to_shape(tail_atom, (64, self.block_k), order=(0, 1))
        vlayout = cute.tile_to_shape(atom, (256, self.block_k), order=(0, 1))
        self.kernel(q, kv, idx, lens, out, tensor_map, qk, pv,
                    qlayout, klayout, playout, vlayout, qtail_layout, ktail_layout).launch(
                        grid=(q.shape[0], 1, 1), block=(256, 1, 1), min_blocks_per_mp=2, stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
               lens: cute.Tensor, out: cute.Tensor, tensor_map: cute.Tensor, qk: cute.TiledMma,
               pv: cute.TiledMma, qlayout: cute.ComposedLayout,
               klayout: cute.ComposedLayout, playout: cute.ComposedLayout,
               vlayout: cute.ComposedLayout, qtail_layout: cute.ComposedLayout,
               ktail_layout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        qi, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        alloc = utils.SmemAllocator()
        sq = alloc.allocate_tensor(cutlass.Float8E4M3FN, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sq_tail = alloc.allocate_tensor(cutlass.Float8E4M3FN, qtail_layout.outer,
                                        byte_alignment=128, swizzle=qtail_layout.inner)
        kbytes = cute.cosize(klayout.outer)
        ktbytes = cute.cosize(ktail_layout.outer)
        sk_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(kbytes),
                                        byte_alignment=128, swizzle=klayout.inner)
        sk_tail_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(ktbytes),
                                             byte_alignment=128, swizzle=ktail_layout.inner)
        assert cute.cosize(playout.outer) == ktbytes
        assert playout.inner == ktail_layout.inner
        sp = cute.make_tensor(sk_tail_base.iterator, playout.outer)
        valid_base = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(self.block_k))
        valid_masks = alloc.allocate_tensor(cutlass.Uint32, cute.make_layout(4))
        qk_done = alloc.allocate_array(cutlass.Int64, 1)
        pv_done = alloc.allocate_array(cutlass.Int64, 1)
        p_ready = alloc.allocate_array(cutlass.Int64, 1)
        output_saved = alloc.allocate_array(cutlass.Int64, 1)
        qbar = alloc.allocate_array(cutlass.Int64, 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        full = alloc.allocate_array(cutlass.Int64, 1)
        empty = alloc.allocate_array(cutlass.Int64, 1)
        if tid == 0:
            cute.arch.mbarrier_init(qk_done, 1)
            cute.arch.mbarrier_init(pv_done, 1)
            cute.arch.mbarrier_init(p_ready, 128)
            cute.arch.mbarrier_init(output_saved, 128)
            cute.arch.mbarrier_init(qbar, 1)
            for stage in cutlass.range(1, unroll_full=True):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(256, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        # Two complete warpgroups: initial128*256=32768 registers/CTA;
        # final128*32+128*224=32768. Inspect emitted allocation before launch.
        if warp >= 4:
            cute.arch.setmaxregister_decrease(32)
        store128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                       cutlass.Float8E4M3FN, num_bits_per_copy=128)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(qbar, 64 * 576)
                for col_block in cutlass.range(4, unroll_full=True):
                    raw = cute.recast_ptr(sq.iterator) + cute.assume(col_block * 64 * 128, divby=128)
                    load_q_tile(raw, tensor_map.iterator + 256, col_block * 128, qi * 64, qbar)
                load_q_tile(cute.recast_ptr(sq_tail.iterator), tensor_map.iterator + 384, 0, qi * 64, qbar)
        if (warp >= 4) & (warp < 6):
            load_tid = tid - 128
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = 0
                row0 = (warp - 4) * 64 + load_tid % 32
                pos0 = block * self.block_k + row0
                pos1 = pos0 + 32
                slot0 = cutlass.Int32(-1)
                slot1 = cutlass.Int32(-1)
                if pos0 < nvalid:
                    slot0 = idx[qi, pos0]
                if pos1 < nvalid:
                    slot1 = idx[qi, pos1]
                if block > 0:
                    cute.arch.mbarrier_wait(empty, (block - 1) % 2)
                sk = cute.make_tensor(sk_base.iterator, klayout.outer)
                valid = valid_base
                valid[row0] = slot0
                valid[row0 + 32] = slot1
                bits0 = cute.arch.vote_ballot_sync(slot0 >= 0)
                bits1 = cute.arch.vote_ballot_sync(slot1 >= 0)
                if load_tid % 32 == 0:
                    valid_masks[(warp - 4) * 2] = bits0
                    valid_masks[(warp - 4) * 2 + 1] = bits1
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                if load_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, kbytes + ktbytes)
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
                with cute.arch.elect_one():
                    for group in cutlass.range(16, unroll_full=True):
                        row = (warp - 4) * 64 + group * 4
                        i0, i1 = valid[row], valid[row + 1]
                        i2, i3 = valid[row + 2], valid[row + 3]
                        raw = cute.recast_ptr(sk.iterator)
                        for col_block in cutlass.range(4, unroll_full=True):
                            dest = raw + cute.assume(col_block * self.block_k * 128 + row * 128, divby=128)
                            gather4(dest, tensor_map.iterator, col_block * 128,
                                    i0, i1, i2, i3, full + stage)
                        tail_raw = cute.recast_ptr(sk_tail_base.iterator) + cute.assume(stage * ktbytes + row * 64, divby=128)
                        gather4(tail_raw, tensor_map.iterator + 128, 0,
                                i0, i1, i2, i3, full + stage)
                cute.arch.barrier(barrier_id=2, number_of_threads=64)
        elif warp == 6:
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            cute.arch.mbarrier_wait(qbar, 0)
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = 0
                # Scores overlap the upper128 columns of output plane1.
                # All compute saves must finish before issuing QK.
                cute.arch.mbarrier_wait(output_saved, block % 2)
                tmem_after_sync()
                cute.arch.mbarrier_wait(full + stage, block % 2)
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                sk_tail = cute.make_tensor(sk_tail_base.iterator + cute.assume(stage * ktbytes, divby=128), ktail_layout.outer)
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                with cute.arch.elect_one():
                    for k in cutlass.range(16, unroll_full=True):
                        mma_normal(tp + (16 << 16) + 128,
                            cute.local_tile(sq, (64, 32), (0, k)),
                            cute.local_tile(sk, (128, 32), (0, k)),
                            128, False, k > 0)
                    for k in cutlass.range(2, unroll_full=True):
                        mma_normal(tp + (16 << 16) + 128,
                            cute.local_tile(sq_tail, (64, 32), (0, k)),
                            cute.local_tile(sk_tail, (128, 32), (0, k)),
                            128, False, True)
                    tcgen05.commit(qk_done)
                cute.arch.mbarrier_wait(p_ready, block % 2)
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                with cute.arch.elect_one():
                    for plane in cutlass.range(2, unroll_full=True):
                        for k in cutlass.range(4, unroll_full=True):
                            mma_normal(tp + plane * (16 << 16),
                                cute.local_tile(sp, (64, 32), (0, k)),
                                cute.local_tile(sk_transposed, (256, 32), (plane, k)),
                                256, True, (block > 0) | (k > 0))
                    tcgen05.commit(pv_done)
                # The next QK waits for PV and its output-half save.
        elif warp < 4:
            cute.arch.setmaxregister_increase(224)
            cute.arch.mbarrier_wait(qbar, 0)
            ctid = tid
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            cs = qk.make_fragment_C(qk.partition_shape_C((64, 128)))
            first = cs[((None, None), 0, 0)]
            chunk_layout = cute.make_layout((first.shape[0], 64),
                                           stride=(first.stride[0], first.stride[1]))
            half_layout = cute.make_layout((first.shape[0], 128),
                                          stride=(first.stride[0], first.stride[1]))
            saved_half = cute.make_tensor(tp + (16 << 16) + 128, half_layout)
            chunk = cute.make_tensor(tp, chunk_layout)
            scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.LdRed16x32bx2Op(tcgen05.copy.Repetition(64),
                    redOp=tcgen05.TmemLoadRedOp.MAX, nan=True,
                    half_split_off=64), cutlass.Float32), saved_half)
            ccopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), chunk)
            cstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), chunk)
            hcopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(64)), cutlass.Float32), saved_half)
            hstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(64)), cutlass.Float32), saved_half)
            st = scopy.get_slice(ctid)
            ct = ccopy.get_slice(ctid)
            coords = st.partition_D(cute.make_identity_tensor((64, 128)))
            out_coords = ct.partition_D(cute.make_identity_tensor((64, 64)))
            rs = cute.make_fragment_like(coords, cutlass.Float32)
            rc = cute.make_fragment_like(out_coords, cutlass.Float32)
            saved_o = cute.make_fragment_like(coords, cutlass.Float32)
            loaded_max = cute.make_rmem_tensor(
                cute.make_layout((1, coords.shape[1], coords.shape[2])), cutlass.Float32)
            head = coords[0][0]
            rowmax = cutlass.Float32(-1.0e30)
            rowsum = cutlass.Float32(0.0)
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = 0
                cute.arch.mbarrier_wait(full + stage, block % 2)
                tmem_after_sync()
                bits0, bits1 = valid_masks[0], valid_masks[1]
                bits2, bits3 = valid_masks[2], valid_masks[3]
                lo_bits, hi_bits = bits0, bits1
                if ctid % 32 >= 16:
                    lo_bits, hi_bits = bits2, bits3
                all_full = (bits0 & bits1 & bits2 & bits3) == cutlass.Uint32(0xffffffff)
                if block > 0:
                    cute.copy(hcopy, hcopy.get_slice(ctid).partition_S(saved_half), saved_o)
                    cute.arch.fence_view_async_tmem_load()
                tmem_before_sync()
                cute.arch.mbarrier_arrive(output_saved)
                cute.arch.mbarrier_wait(qk_done, block % 2)
                tmem_after_sync()
                cute.copy(scopy, st.partition_S(saved_half), (rs, loaded_max))
                cute.arch.fence_view_async_tmem_load()
                newmax = cute.arch.fmax(loaded_max[0], rowmax)
                lo_scores = cute.make_tensor(rs.iterator, cute.make_layout(32))
                hi_scores = cute.make_tensor(rs.iterator + 32, cute.make_layout(32))
                lo_max = mask_reduce32(lo_scores, lo_bits, rowmax, newmax, all_full)
                hi_max = mask_reduce32(hi_scores, hi_bits, rowmax, newmax, all_full)
                newmax = cute.arch.fmax(lo_max, hi_max)
                # Paired half-warps own one head's two64-key segments.
                newmax = cute.arch.fmax(newmax, cute.arch.shuffle_sync_bfly(newmax, offset=16))
                correction = cute.math.exp2((rowmax - newmax) * (0.0625 * math.log2(math.e)), fastmath=True)
                scaled_shift = math.log2(448.0) - newmax * (0.0625 * math.log2(math.e))
                probs = cute.math.exp2(rs.load() * (0.0625 * math.log2(math.e)) + scaled_shift, fastmath=True)
                sum_tree = cute.make_rmem_tensor(64, cutlass.Float32)
                sum_tree.store(probs)
                for level in cutlass.range_constexpr(6):
                    for j in cutlass.range_constexpr(32 >> level):
                        sum_tree[j] = sum_tree[2 * j] + sum_tree[2 * j + 1]
                rowsum = rowsum * correction + sum_tree[0]
                rowmax = newmax
                packed_p = cute.make_rmem_tensor(64, cutlass.Float8E4M3FN)
                packed_p.store(probs.to(cutlass.Float8E4M3FN))
                for j in cutlass.range(4, unroll_full=True):
                    h, col = coords[j * 16]
                    rvec = cute.make_tensor(packed_p.iterator + j * 16, cute.make_layout(16))
                    svec = cute.make_tensor(sp.iterator + cute.assume(sp.layout((h, col)), divby=16), cute.make_layout(16))
                    cute.copy(store128, rvec, svec)
                skip_correction = cute.arch.vote_all_sync(correction == 1.0)
                if block > 0:
                    if not skip_correction:
                        for plane in cutlass.range_constexpr(2):
                            for tile in cutlass.range_constexpr(4):
                                if cutlass.const_expr(plane == 0 or tile < 2):
                                    otile = cute.make_tensor(tp + (plane * 16 << 16) + tile * 64, chunk_layout)
                                    cute.copy(ccopy, ct.partition_S(otile), rc)
                                    cute.arch.fence_view_async_tmem_load()
                                    for j in cutlass.range(0, 32, 2, unroll_full=True):
                                        rc[j], rc[j + 1] = cute.arch.mul_packed_f32x2(
                                            (rc[j], rc[j + 1]), (correction, correction))
                                    cute.copy(cstore, rc, cstore.get_slice(ctid).partition_D(otile))
                    # Score MMA destroyed this half, even for correction1.
                    for j in cutlass.range(0, 64, 2, unroll_full=True):
                        saved_o[j], saved_o[j + 1] = cute.arch.mul_packed_f32x2(
                            (saved_o[j], saved_o[j + 1]), (correction, correction))
                    cute.copy(hstore, saved_o, hstore.get_slice(ctid).partition_D(saved_half))
                    cute.arch.fence_view_async_tmem_store()
                tmem_before_sync()
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(p_ready)
                cute.arch.mbarrier_wait(pv_done, block % 2)
                tmem_after_sync()
                tmem_before_sync()
                if tid == 0:
                    cute.arch.mbarrier_arrive(empty + stage)
            total_sum = rowsum + cute.arch.shuffle_sync_bfly(rowsum, offset=16)
            norm = 1.0 / total_sum
            packed_out = cute.make_rmem_tensor(32, cutlass.BFloat16)
            for plane in cutlass.range(2, unroll_full=True):
                for tile in cutlass.range(4, unroll_full=True):
                    otile = cute.make_tensor(tp + (plane * 16 << 16) + tile * 64, chunk_layout)
                    cute.copy(ccopy, ct.partition_S(otile), rc)
                    cute.arch.fence_view_async_tmem_load()
                    packed_out.store((rc.load() * norm).to(cutlass.BFloat16))
                    for v in cutlass.range(2, unroll_full=True):
                        channel = plane * 256 + tile * 64 + out_coords[v * 16][1]
                        offset = qi * 64 * 512 + head * 512 + channel
                        words = cute.make_tensor(cute.recast_ptr(packed_out.iterator + v * 16,
                            dtype=cutlass.Uint32), cute.make_layout(8))
                        store_output256(out.iterator + cute.assume(offset, divby=16), words)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            tmem_after_sync()
            if warp == 0:
                cute.arch.dealloc_tmem(tp, 256)



def make_runner(inputs, block_k=128):
    q = inputs['query'].squeeze(1)
    kv = inputs['kv_cache'].view(-1, 576)
    idx = inputs['block_tables'].view(q.shape[0], -1)
    lens = inputs['seq_lens']
    out = torch.empty((q.shape[0], 64, 512), device=q.device, dtype=torch.bfloat16)
    assert out.data_ptr() % 32 == 0, "STG256 requires32-byte output alignment"
    tensor_map = make_kv_map(kv, q)
    args = [from_dlpack(t, assumed_align=16) for t in (q, kv, idx, lens, out, tensor_map)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    guardrails = os.environ.get("MLA_TMEM_GUARDRAILS") == "1"
    compile_options = "--ptxas-options=-g-tmem-access-check" if guardrails else ""
    compiled = cute.compile(SparseMLA(block_k), *args, stream, options=compile_options)
    def run():
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
    return run
