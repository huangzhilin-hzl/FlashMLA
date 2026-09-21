"""Iteration 056: four compute groups for one-head weight-stationary softmax.

One CTA owns one query and all 512 output channels. Both PV N tiles reuse
one QK/softmax computation and one gathered KV tile. No input expansion,
BF16 dequantization, dense-attention fallback, or reference computation is used.
"""
import math
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
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
def mma_ws(d, a, b, n, b_transpose, accumulate, *, loc=None, ip=None):
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
         cutlass.Int32(accumulate).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; "
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }",
        "r,l,l,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)



class SparseMLA:
    def __init__(self, block_k=128):
        assert block_k == 128, "v056 requires --block-k 128"
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
                        grid=(q.shape[0], 1, 1), block=(640, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
               lens: cute.Tensor, out: cute.Tensor, tensor_map: cute.Tensor, qk: cute.TiledMma,
               pv: cute.TiledMma, qlayout: cute.ComposedLayout,
               klayout: cute.ComposedLayout, playout: cute.ComposedLayout,
               vlayout: cute.ComposedLayout, qtail_layout: cute.ComposedLayout,
               ktail_layout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        qi, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        alloc = utils.SmemAllocator()
        sq = alloc.allocate_tensor(cutlass.Float8E4M3FN, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sq_tail = alloc.allocate_tensor(cutlass.Float8E4M3FN, qtail_layout.outer,
                                        byte_alignment=128, swizzle=qtail_layout.inner)
        kbytes = cute.cosize(klayout.outer)
        ktbytes = cute.cosize(ktail_layout.outer)
        sk_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * kbytes),
                                        byte_alignment=128, swizzle=klayout.inner)
        sk_tail_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * ktbytes),
                                             byte_alignment=128, swizzle=ktail_layout.inner)
        sp = alloc.allocate_tensor(cutlass.Float8E4M3FN, playout.outer, byte_alignment=128, swizzle=playout.inner)
        denom = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        partial_max = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(512))
        partial_sum = partial_max  # Reused only after the final score tile.
        valid_base = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(2 * self.block_k))
        bar = alloc.allocate_array(cutlass.Int64, 1)
        qbar = alloc.allocate_array(cutlass.Int64, 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        full = alloc.allocate_array(cutlass.Int64, 2)
        empty = alloc.allocate_array(cutlass.Int64, 2)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(qbar, 1)
            for stage in cutlass.range(2, unroll_full=True):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(512, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        store128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                       cutlass.Float8E4M3FN, num_bits_per_copy=128)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(qbar, 64 * 576)
                for col_block in cutlass.range(4, unroll_full=True):
                    raw = cute.recast_ptr(sq.iterator) + cute.assume(col_block * 64 * 128, divby=128)
                    load_q_tile(raw, tensor_map.iterator + 256, col_block * 128, qi * 64, qbar)
                load_q_tile(cute.recast_ptr(sq_tail.iterator), tensor_map.iterator + 384, 0, qi * 64, qbar)
        if tid >= 512:
            load_tid = tid - 512
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = block % 2
                if block >= 2:
                    cute.arch.mbarrier_wait(empty + stage, ((block // 2) - 1) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                if load_tid < 128:
                    pos = block * self.block_k + load_tid
                    slot = cutlass.Int32(-1)
                    if pos < nvalid:
                        slot = idx[qi, pos]
                    valid[load_tid] = slot
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if load_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, kbytes + ktbytes)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                with cute.arch.elect_one():
                    for group in cutlass.range(8, unroll_full=True):
                        row = (warp - 16) * 32 + group * 4
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
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
        else:
            cute.arch.mbarrier_wait(qbar, 0)
            cgroup = tid // 128
            ctid = tid % 128
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            # Layout E packs N halves into DP[0,64) and DP[64,128).
            # Each group reads 16 physical columns, one head per thread.
            score_layout = cute.make_layout((128, 16), stride=(1 << 16, 1))
            score_half = cute.make_tensor(tp + 256 + cgroup * 16, score_layout)
            scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), cutlass.Float32), score_half)
            st = scopy.get_slice(ctid)
            src_s = st.partition_S(score_half)
            coords_s = st.partition_D(cute.make_identity_tensor((128, 16)))
            rs = cute.make_fragment_like(coords_s, cutlass.Float32)
            # Correction follows the same one-head datapath mapping.
            cchunk_layout = cute.make_layout((128, 32), stride=(1 << 16, 1))
            cchunk = cute.make_tensor(tp, cchunk_layout)
            ccopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32), cchunk)
            cstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32), cchunk)
            ct = ccopy.get_slice(ctid)
            rc = cute.make_fragment_like(ct.partition_D(cute.make_identity_tensor((128, 32))), cutlass.Float32)
            rowmax = cutlass.Float32(-1.0e30)
            rowsum = cutlass.Float32(0.0)
            phase = cutlass.Int32(0)
            nvalid = lens[qi]
            num_blocks = cute.ceil_div(nvalid, self.block_k)
            for block in cutlass.range(num_blocks):
                stage = block % 2
                cute.arch.mbarrier_wait(full + stage, (block // 2) % 2)
                tmem_after_sync()
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                sk_tail = cute.make_tensor(sk_tail_base.iterator + cute.assume(stage * ktbytes, divby=128), ktail_layout.outer)
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                cute.arch.fence_view_async_shared()
                if warp == 0:
                    with cute.arch.elect_one():
                        for k in cutlass.range(16, unroll_full=True):
                            mma_ws(tp + 256,
                                cute.local_tile(sq, (64, 32), (0, k)),
                                cute.local_tile(sk, (128, 32), (0, k)),
                                128, False, k > 0)
                        for k in cutlass.range(2, unroll_full=True):
                            mma_ws(tp + 256,
                                cute.local_tile(sq_tail, (64, 32), (0, k)),
                                cute.local_tile(sk_tail, (128, 32), (0, k)),
                                128, False, True)
                        tcgen05.commit(bar)
                cute.arch.mbarrier_wait(bar, phase)
                tmem_after_sync()
                phase ^= 1
                cute.copy(scopy, src_s, rs)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(rs), unroll_full=True):
                    h, col = coords_s[j]
                    val = cutlass.Float32(-1.0e30)
                    if valid[col + (h // 64) * 64 + cgroup * 16] >= 0:
                        val = rs[j] * (0.0625 * math.log2(math.e))
                    rs[j] = val
                newmax = rs.load().reduce(cute.ReductionOp.MAX, rowmax, 0)
                head = coords_s[0][0] % 64
                partial_max[cgroup * 128 + coords_s[0][0]] = newmax
                tmem_before_sync()
                cute.arch.barrier(barrier_id=1, number_of_threads=512)
                tmem_after_sync()
                newmax = partial_max[head]
                for part in cutlass.range(1, 8, unroll_full=True):
                    newmax = cute.arch.fmax(newmax, partial_max[part * 64 + head])
                correction = cute.math.exp2(rowmax - newmax, fastmath=True)
                probs = cute.math.exp2(rs.load() - newmax, fastmath=True)
                blocksum = probs.reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)
                # Both groups share each running maximum, so their separately
                # corrected denominator contributions can be summed once at exit.
                rowsum = rowsum * correction + blocksum
                rowmax = newmax
                packed_p = cute.make_fragment_like(rs, cutlass.Float8E4M3FN)
                packed_p.store((probs * 448.0).to(cutlass.Float8E4M3FN))
                for j in cutlass.range(cute.size(rs) // 16, unroll_full=True):
                    dp, col = coords_s[j * 16]
                    h = dp % 64
                    col = col + (dp // 64) * 64 + cgroup * 16
                    rvec = cute.make_tensor(packed_p.iterator + j * 16, cute.make_layout(16))
                    svec = cute.make_tensor(sp.iterator + cute.assume(sp.layout((h, col)), divby=16),
                                            cute.make_layout(16))
                    cute.copy(store128, rvec, svec)
                skip_correction = cute.arch.vote_all_sync(correction == 1.0)
                if (block > 0) & (not skip_correction):
                    for tile in cutlass.range(2, unroll_full=True):
                        otile = cute.make_tensor(tp + cute.assume(cgroup * 64 + tile * 32, divby=32), cchunk_layout)
                        src_o = ct.partition_S(otile)
                        dst_o = cstore.get_slice(ctid).partition_D(otile)
                        cute.copy(ccopy, src_o, rc)
                        cute.arch.fence_view_async_tmem_load()
                        for j in cutlass.range(0, cute.size(rc), 2, unroll_full=True):
                            rc[j], rc[j + 1] = cute.arch.mul_packed_f32x2(
                                (rc[j], rc[j + 1]), (correction, correction))
                        cute.copy(cstore, rc, dst_o)
                    cute.arch.fence_view_async_tmem_store()
                tmem_before_sync()
                cute.arch.barrier(barrier_id=1, number_of_threads=512)
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                if warp == 0:
                    with cute.arch.elect_one():
                        for ntile in cutlass.range(2, unroll_full=True):
                            for k in cutlass.range(4, unroll_full=True):
                                mma_ws(tp + ntile * 128,
                                    cute.local_tile(sp, (64, 32), (0, k)),
                                    cute.local_tile(sk_transposed, (256, 32), (ntile, k)),
                                    256, True, (block > 0) | (k > 0))
                        tcgen05.commit(bar)
                cute.arch.mbarrier_wait(bar, phase)
                tmem_after_sync()
                phase ^= 1
                tmem_before_sync()
                cute.arch.barrier(barrier_id=1, number_of_threads=512)
                tmem_after_sync()
                if tid == 0:
                    cute.arch.mbarrier_arrive(empty + stage)
            head = coords_s[0][0] % 64
            partial_sum[cgroup * 128 + coords_s[0][0]] = rowsum
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=512)
            tmem_after_sync()
            if tid < 64:
                total_sum = partial_sum[head]
                for part in cutlass.range(1, 8, unroll_full=True):
                    total_sum += partial_sum[part * 64 + head]
                denom[head] = 1.0 / (total_sum * 448.0)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=512)
            tmem_after_sync()
            # The final PV and compute barrier drained all KV reads. Reuse
            # the first 64 KiB KV-main stage as a BF16 output transpose buffer.
            out_atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16)
            out_layout = cute.tile_to_shape(out_atom, (64, 512), order=(0, 1))
            assert cute.cosize(out_layout.outer) * 2 <= kbytes
            shared_out = cute.make_tensor(cute.recast_ptr(sk_base.iterator,
                swizzle_=out_layout.inner, dtype=cutlass.BFloat16), out_layout.outer)
            copy_bf128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                             cutlass.BFloat16, num_bits_per_copy=128)
            packed_out = cute.make_rmem_tensor(32, cutlass.BFloat16)
            head = ctid % 64
            norm = denom[head]
            for tile in cutlass.range(2, unroll_full=True):
                fragment = cute.make_tensor(tp + cute.assume(cgroup * 64 + tile * 32, divby=32), cchunk_layout)
                cute.copy(ccopy, ct.partition_S(fragment), rc)
                cute.arch.fence_view_async_tmem_load()
                packed_out.store((rc.load() * norm).to(cutlass.BFloat16))
                for v in cutlass.range(4, unroll_full=True):
                    channel = (cgroup // 2) * 256 + (cgroup % 2) * 64 + (ctid // 64) * 128 + tile * 32 + v * 8
                    rvec = cute.make_tensor(packed_out.iterator + v * 8, cute.make_layout(8))
                    svec = cute.make_tensor(shared_out.iterator + cute.assume(shared_out.layout((head, channel)), divby=8), cute.make_layout(8))
                    cute.copy(copy_bf128, rvec, svec)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=512)
            tmem_after_sync()
            for v in cutlass.range(8, unroll_full=True):
                offset = (tid + v * 512) * 8
                h, d = offset // 512, offset % 512
                svec = cute.make_tensor(shared_out.iterator + cute.assume(shared_out.layout((h, d)), divby=8), cute.make_layout(8))
                gvec = cute.make_tensor(out.iterator + cute.assume(qi * 64 * 512 + offset, divby=8), cute.make_layout(8))
                cute.copy(copy_bf128, svec, gvec)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=512)
            tmem_after_sync()
            if warp == 0:
                cute.arch.dealloc_tmem(tp, 512)



def make_runner(inputs, block_k=128):
    q = inputs['query'].squeeze(1)
    kv = inputs['kv_cache'].view(-1, 576)
    idx = inputs['block_tables'].view(q.shape[0], -1)
    lens = inputs['seq_lens']
    out = torch.empty((q.shape[0], 64, 512), device=q.device, dtype=torch.bfloat16)
    tensor_map = make_kv_map(kv, q)
    args = [from_dlpack(t, assumed_align=16) for t in (q, kv, idx, lens, out, tensor_map)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(SparseMLA(block_k), *args, stream)
    def run():
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
    return run
