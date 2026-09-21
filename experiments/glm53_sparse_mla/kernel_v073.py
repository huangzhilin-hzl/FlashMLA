"""Iteration 073: normal MMA/TMEM-P with deferred sums and vector epilogue; based on036/054.

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


class SparseMLA:
    def __init__(self, block_k=128):
        assert block_k == 128, "v073 requires --block-k 128"
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
                                      tcgen05.CtaGroup.ONE, (64, 256), a_source=tcgen05.OperandSource.TMEM)
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
                        grid=(q.shape[0], 1, 1), block=(512, 1, 1), stream=stream)

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
        denom = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        partial_max = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(128))
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
        if tid >= 256:
            cute.arch.setmaxregister_decrease(32)
            load_tid = tid - 256
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
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                if load_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, kbytes + ktbytes)
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
                with cute.arch.elect_one():
                    for group in cutlass.range(4, unroll_full=True):
                        row = (warp - 8) * 16 + group * 4
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
                cute.arch.barrier(barrier_id=2, number_of_threads=256)
        else:
            cute.arch.setmaxregister_increase(208)
            cute.arch.mbarrier_wait(qbar, 0)
            cgroup = tid // 128
            ctid = tid % 128
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            sq_part = qk.get_slice(0).partition_A(sq)
            aq = qk.make_fragment_A(sq_part)
            aq_tail = qk.make_fragment_A(qk.get_slice(0).partition_A(sq_tail))
            cs = qk.make_fragment_C(qk.partition_shape_C((64, self.block_k)))
            # Explicitly place the second N256 output tile in the other
            # 16-lane half: O uses columns [0,256) across all datapaths.
            co_default = pv.make_fragment_C(pv.partition_shape_C((64, 512)))
            co = cute.make_tensor(co_default.iterator, cute.make_layout(co_default.shape,
                stride=(co_default.stride[0], co_default.stride[1], 16 << 16)))
            assert tcgen05.find_tmem_tensor_col_offset(co) == 256
            # QK scores use the lower lane half at [256,384). P is packed into
            # [384,416), duplicated across both lane halves so each PV A/D pair
            # has matching lane alignment as required by PTX Layout F.
            ts = cute.make_tensor(tp + 256, cs.layout)
            ap_layout = pv.make_fragment_A(pv.partition_shape_A((64, self.block_k))).layout
            ap = cute.make_tensor(cute.recast_ptr(tp + 384, dtype=cutlass.Float8E4M3FN), ap_layout)
            to = cute.make_tensor(tp, co.layout)
            ts2 = ts[((None, None), 0, 0)]
            to2 = to[((None, None), 0, 0)]
            # Each compute group handles 64 key columns across all 64 heads.
            score_layout = cute.make_layout((ts2.shape[0], 64),
                                             stride=(ts2.stride[0], ts2.stride[1]))
            score_half = cute.make_tensor(tp + 256 + cgroup * 64, score_layout)
            scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)),
                cutlass.Float32), score_half)
            ochunk_layout = cute.make_layout((to2.shape[0], 64),
                                             stride=(to2.stride[0], to2.stride[1]))
            ochunk = cute.make_tensor(tp, ochunk_layout)
            ocopy = tcgen05.make_tmem_copy(
                bw.get_tmem_load_op((64, 64, self.block_k), utils.LayoutEnum.ROW_MAJOR,
                    cutlass.Float32, cutlass.Float32, (64, 64), False), ochunk)
            st = scopy.get_slice(ctid)
            ot = ocopy.get_slice(ctid)
            src_s = st.partition_S(score_half)
            coords_s = st.partition_D(cute.make_identity_tensor((64, 64)))
            rs = cute.make_fragment_like(coords_s, cutlass.Float32)
            pfirst = ap[((None, None), 0, 0)]
            p2 = cute.make_tensor(ap.iterator, cute.make_layout(
                (pfirst.shape[0], 64), stride=(pfirst.stride[0], pfirst.stride[1])))
            pstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(8)),
                cutlass.Float8E4M3FN), p2)
            pt = pstore.get_slice(ctid)
            pcoords = pt.partition_S(cute.make_identity_tensor((64, 64)))
            packed_p = cute.make_fragment_like(pcoords, cutlass.Float8E4M3FN)
            coords_o = ot.partition_D(cute.make_identity_tensor((64, 64)))
            ro = cute.make_fragment_like(coords_o, cutlass.Float32)
            # This layout gives each lane the same head as its softmax data.
            # Preserve the original coalesced O layout for the final epilogue.
            ccopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            cstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            ct = ccopy.get_slice(ctid)
            ccoords = ct.partition_D(cute.make_identity_tensor((64, 64)))
            rc = cute.make_fragment_like(ccoords, cutlass.Float32)
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
                sk_part = qk.get_slice(0).partition_B(sk)
                bk = qk.make_fragment_B(sk_part)
                sk_tail = cute.make_tensor(sk_tail_base.iterator + cute.assume(stage * ktbytes, divby=128), ktail_layout.outer)
                bk_tail = qk.make_fragment_B(qk.get_slice(0).partition_B(sk_tail))
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                sv = cute.local_tile(sk_transposed, (512, self.block_k), (0, 0))
                sv_part = pv.get_slice(0).partition_B(sv)
                bv = pv.make_fragment_B(sv_part)
                cute.arch.fence_view_async_shared()
                if warp == 0:
                    qk.set(tcgen05.Field.ACCUMULATE, False)
                    for k in cutlass.range(cute.size(aq, mode=[2]), unroll_full=True):
                        cute.gemm(qk, ts, aq[None, None, k], bk[None, None, k], ts)
                        qk.set(tcgen05.Field.ACCUMULATE, True)
                    for k in cutlass.range(cute.size(aq_tail, mode=[2]), unroll_full=True):
                        cute.gemm(qk, ts, aq_tail[None, None, k], bk_tail[None, None, k], ts)
                    with cute.arch.elect_one():
                        tcgen05.commit(bar)
                cute.arch.mbarrier_wait(bar, phase)
                tmem_after_sync()
                phase ^= 1
                cute.copy(scopy, src_s, rs)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(rs), unroll_full=True):
                    h, col = coords_s[j]
                    val = cutlass.Float32(-1.0e30)
                    if valid[col + cgroup * 64] >= 0:
                        val = rs[j] * (0.0625 * math.log2(math.e))
                    rs[j] = val
                newmax = rs.load().reduce(cute.ReductionOp.MAX, rowmax, 0)
                newmax = cute.arch.fmax(newmax, cute.arch.shuffle_sync_bfly(newmax, offset=16))
                head = coords_s[0][0]
                if ctid % 32 < 16:
                    partial_max[cgroup * 64 + head] = newmax
                tmem_before_sync()
                cute.arch.barrier(barrier_id=1, number_of_threads=256)
                tmem_after_sync()
                newmax = cute.arch.fmax(partial_max[head], partial_max[64 + head])
                correction = cute.math.exp2(rowmax - newmax, fastmath=True)
                probs = cute.math.exp2(rs.load() - newmax, fastmath=True)
                blocksum = probs.reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)
                blocksum += cute.arch.shuffle_sync_bfly(blocksum, offset=16)
                rowsum = rowsum * correction + blocksum
                rowmax = newmax
                packed_p.store((probs * 448.0).to(cutlass.Float8E4M3FN))
                for lane_half in cutlass.range(2, unroll_full=True):
                    p_dst = cute.make_tensor(cute.recast_ptr(tp + 384 + cgroup * 16 + (lane_half * 16 << 16),
                                             dtype=cutlass.Float8E4M3FN), p2.layout)
                    cute.copy(pstore, packed_p, pt.partition_D(p_dst))
                cute.arch.fence_view_async_tmem_store()
                skip_correction = cute.arch.vote_all_sync(correction == 1.0)
                if (block > 0) & (not skip_correction):
                    for tile in cutlass.range(4, unroll_full=True):
                        otile = cute.make_tensor(tp + cute.assume(tile * 64 + (cgroup * 16 << 16), divby=64), ochunk_layout)
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
                cute.arch.barrier(barrier_id=1, number_of_threads=256)
                tmem_after_sync()
                cute.arch.fence_view_async_shared()
                if warp == 0:
                    for ntile in cutlass.range(2, unroll_full=True):
                        a_pv = cute.make_tensor(cute.recast_ptr(tp + 384 + (ntile * 16 << 16),
                                               dtype=cutlass.Float8E4M3FN), ap_layout)
                        pv.set(tcgen05.Field.ACCUMULATE, block > 0)
                        for k in cutlass.range(cute.size(a_pv, mode=[2]), unroll_full=True):
                            cute.gemm(pv, to[None, 0, ntile], a_pv[None, 0, k],
                                      bv[None, ntile, k], to[None, 0, ntile])
                            pv.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(bar)
                cute.arch.mbarrier_wait(bar, phase)
                tmem_after_sync()
                phase ^= 1
                tmem_before_sync()
                cute.arch.barrier(barrier_id=1, number_of_threads=256)
                tmem_after_sync()
                if tid == 0:
                    cute.arch.mbarrier_arrive(empty + stage)
            head = coords_s[0][0]
            if ctid % 32 < 16:
                partial_sum[cgroup * 64 + head] = rowsum
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            tmem_after_sync()
            if (cgroup == 0) & (ctid % 32 < 16):
                denom[head] = 1.0 / ((partial_sum[head] + partial_sum[64 + head]) * 448.0)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
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
            head = ccoords[0][0]
            norm = denom[head]
            for tile in cutlass.range(4, unroll_full=True):
                fragment = cute.make_tensor(tp + cute.assume(tile * 64 + (cgroup * 16 << 16), divby=64), ochunk_layout)
                cute.copy(ccopy, ct.partition_S(fragment), rc)
                cute.arch.fence_view_async_tmem_load()
                packed_out.store((rc.load() * norm).to(cutlass.BFloat16))
                for v in cutlass.range(4, unroll_full=True):
                    channel = cgroup * 256 + tile * 64 + ccoords[v * 8][1]
                    rvec = cute.make_tensor(packed_out.iterator + v * 8, cute.make_layout(8))
                    svec = cute.make_tensor(shared_out.iterator + cute.assume(shared_out.layout((head, channel)), divby=8), cute.make_layout(8))
                    cute.copy(copy_bf128, rvec, svec)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            tmem_after_sync()
            for v in cutlass.range(16, unroll_full=True):
                offset = (tid + v * 256) * 8
                h, d = offset // 512, offset % 512
                svec = cute.make_tensor(shared_out.iterator + cute.assume(shared_out.layout((h, d)), divby=8), cute.make_layout(8))
                gvec = cute.make_tensor(out.iterator + cute.assume(qi * 64 * 512 + offset, divby=8), cute.make_layout(8))
                cute.copy(copy_bf128, svec, gvec)
            tmem_before_sync()
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
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
