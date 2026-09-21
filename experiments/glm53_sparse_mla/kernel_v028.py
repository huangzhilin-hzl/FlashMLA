"""Iteration 028: independent MMA issuer with two S/P stages.

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
from cutlass._mlir.dialects import llvm


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


def make_kv_map(kv):
    status, desc = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv.data_ptr(),
        [cuda.cuuint64_t(576), cuda.cuuint64_t(kv.shape[0])], [cuda.cuuint64_t(576)],
        [cuda.cuuint32_t(64), cuda.cuuint32_t(1)], [cuda.cuuint32_t(1), cuda.cuuint32_t(1)],
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled: {status}")
    storage = torch.empty(128, dtype=torch.uint8, device=kv.device)
    status, = cuda.cuMemcpyHtoD(storage.data_ptr(), desc.getPtr(), 128)
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuMemcpyHtoD tensor map: {status}")
    return storage


class SparseMLA:
    def __init__(self, block_k=128):
        assert block_k == 128, "v028 requires --block-k 128"
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
        atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, fp8)
        qlayout = cute.tile_to_shape(atom, (64, 576), order=(0, 1))
        klayout = cute.tile_to_shape(atom, (self.block_k, 576), order=(0, 1))
        playout = cute.tile_to_shape(atom, (64, self.block_k), order=(0, 1))
        vlayout = cute.tile_to_shape(atom, (256, self.block_k), order=(0, 1))
        self.kernel(q, kv, idx, lens, out, tensor_map, qk, pv,
                    qlayout, klayout, playout, vlayout).launch(
                        grid=(q.shape[0], 1, 1), block=(288, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
               lens: cute.Tensor, out: cute.Tensor, tensor_map: cute.Tensor, qk: cute.TiledMma,
               pv: cute.TiledMma, qlayout: cute.ComposedLayout,
               klayout: cute.ComposedLayout, playout: cute.ComposedLayout,
               vlayout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        qi, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        alloc = utils.SmemAllocator()
        sq = alloc.allocate_tensor(cutlass.Float8E4M3FN, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        kbytes = cute.cosize(klayout.outer)
        sk_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * kbytes),
                                        byte_alignment=128, swizzle=klayout.inner)
        pbytes = cute.cosize(playout.outer)
        sp_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * pbytes),
                                        byte_alignment=128, swizzle=playout.inner)
        denom = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        valid_base = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(2 * self.block_k))
        qk_done = alloc.allocate_array(cutlass.Int64, 2)
        ready = alloc.allocate_array(cutlass.Int64, 2)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        full = alloc.allocate_array(cutlass.Int64, 2)
        empty = alloc.allocate_array(cutlass.Int64, 2)
        if tid == 0:
            for stage in cutlass.range(2, unroll_full=True):
                cute.arch.mbarrier_init(qk_done + stage, 1)
                cute.arch.mbarrier_init(ready + stage, 1)
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(512, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        store128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                       cutlass.Float8E4M3FN, num_bits_per_copy=128)
        load128 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float8E4M3FN, num_bits_per_copy=128)
        if (tid >= 128) & (tid < 256):
            load_tid = tid - 128
            for row_block in cutlass.range(2, unroll_full=True):
                h = load_tid // 4 + row_block * 32
                for col_block in cutlass.range(9, unroll_full=True):
                    d = (load_tid % 4) * 16 + col_block * 64
                    g = cute.make_tensor(q.iterator + cute.assume(qi * 64 * 576 + h * 576 + d, divby=16),
                                         cute.make_layout(16))
                    sh = cute.make_tensor(sq.iterator + cute.assume(sq.layout((h, d)), divby=16), cute.make_layout(16))
                    cute.copy(load128, g, sh)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        if (tid >= 128) & (tid < 256):
            load_tid = tid - 128
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = block % 2
                if block >= 2:
                    cute.arch.mbarrier_wait(empty + stage, ((block // 2) - 1) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                pos = block * self.block_k + load_tid
                slot = cutlass.Int32(-1)
                if pos < nvalid:
                    slot = idx[qi, pos]
                valid[load_tid] = slot
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if load_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, kbytes)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                with cute.arch.elect_one():
                    for group in cutlass.range(8, unroll_full=True):
                        row = (warp - 4) * 32 + group * 4
                        i0, i1 = valid[row], valid[row + 1]
                        i2, i3 = valid[row + 2], valid[row + 3]
                        raw = cute.recast_ptr(sk.iterator)
                        for col_block in cutlass.range(9, unroll_full=True):
                            dest = raw + cute.assume(col_block * self.block_k * 64 + row * 64, divby=128)
                            gather4(dest, tensor_map.iterator, col_block * 64,
                                    i0, i1, i2, i3, full + stage)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
        elif tid >= 256:
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            pm = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN, OperandMajorMode.K,
                OperandMajorMode.MN, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, 256))
            aq = qk.make_fragment_A(qk.get_slice(0).partition_A(sq))
            cs = qk.make_fragment_C(qk.partition_shape_C((64, self.block_k)))
            co_default = pm.make_fragment_C(pm.partition_shape_C((64, 512)))
            co_layout = cute.make_layout(co_default.shape,
                stride=(co_default.stride[0], co_default.stride[1], 256))
            to = cute.make_tensor(tp, co_layout)
            num_blocks = cute.ceil_div(lens[qi], self.block_k)
            for block in cutlass.range(num_blocks):
                if block == 0:
                    self.issue_qk(qk, aq, cs.layout, tp, sk_base, klayout,
                                  kbytes, full, qk_done, cutlass.Int32(0))
                if block + 1 < num_blocks:
                    self.issue_qk(qk, aq, cs.layout, tp, sk_base, klayout,
                                  kbytes, full, qk_done, block + 1)
                stage = block % 2
                cute.arch.mbarrier_wait(ready + stage, (block // 2) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                sv = cute.local_tile(sk_transposed, (512, self.block_k), (0, 0))
                bv = pm.make_fragment_B(pm.get_slice(0).partition_B(sv))
                sp = cute.make_tensor(sp_base.iterator + cute.assume(stage * pbytes, divby=128), playout.outer)
                ap = pm.make_fragment_A(pm.get_slice(0).partition_A(sp))
                cute.arch.fence_view_async_shared()
                pm.set(tcgen05.Field.ACCUMULATE, block > 0)
                for k in cutlass.range(cute.size(ap, mode=[2]), unroll_full=True):
                    cute.gemm(pm, to, ap[None, None, k], bv[None, None, k], to)
                    pm.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage)
        else:
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            sq_part = qk.get_slice(0).partition_A(sq)
            aq = qk.make_fragment_A(sq_part)
            cs = qk.make_fragment_C(qk.partition_shape_C((64, self.block_k)))
            co_default = pv.make_fragment_C(pv.partition_shape_C((64, 512)))
            co_layout = cute.make_layout(co_default.shape,
                stride=(co_default.stride[0], co_default.stride[1], 256))
            co = cute.make_tensor(co_default.iterator, co_layout)
            assert tcgen05.find_tmem_tensor_col_offset(cs) <= 128
            assert tcgen05.find_tmem_tensor_col_offset(co) <= 512
            # PTX Layout F (M64) supports lane alignments 0 and 16.
            # S occupies lanes [16:32] of each warp; O occupies [0:16].
            # They overlap columns but never the same physical cells.
            ts = cute.make_tensor(tp + (16 << 16), cs.layout)
            to = cute.make_tensor(tp, co.layout)
            ts2 = ts[((None, None), 0, 0)]
            to2 = to[((None, None), 0, 0)]
            # Each thread owns one head and half of the key columns. Lanes
            # separated by 16 own the same head and exchange max/sum with shuffles.
            scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(self.block_k // 2)),
                cutlass.Float32), ts2)
            ochunk_layout = cute.make_layout((to2.shape[0], 64),
                                             stride=(to2.stride[0], to2.stride[1]))
            ochunk = cute.make_tensor(tp, ochunk_layout)
            ocopy = tcgen05.make_tmem_copy(
                bw.get_tmem_load_op((64, 64, self.block_k), utils.LayoutEnum.ROW_MAJOR,
                    cutlass.Float32, cutlass.Float32, (64, 64), False), ochunk)
            st = scopy.get_slice(tid)
            ot = ocopy.get_slice(tid)
            src_s = st.partition_S(ts2)
            coords_s = st.partition_D(cute.make_identity_tensor((64, self.block_k)))
            rs = cute.make_fragment_like(coords_s, cutlass.Float32)
            coords_o = ot.partition_D(cute.make_identity_tensor((64, 64)))
            ro = cute.make_fragment_like(coords_o, cutlass.Float32)
            # This layout gives each lane the same head as its softmax data.
            # Preserve the original coalesced O layout for the final epilogue.
            ccopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            cstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            ct = ccopy.get_slice(tid)
            ccoords = ct.partition_D(cute.make_identity_tensor((64, 64)))
            rc = cute.make_fragment_like(ccoords, cutlass.Float32)
            rowmax = cutlass.Float32(-1.0e30)
            rowsum = cutlass.Float32(0.0)
            phase = cutlass.Int32(0)
            nvalid = lens[qi]
            num_blocks = cute.ceil_div(nvalid, self.block_k)
            for block in cutlass.range(num_blocks):
                stage = block % 2
                cute.arch.mbarrier_wait(qk_done + stage, (block // 2) % 2)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                sp = cute.make_tensor(sp_base.iterator + cute.assume(stage * pbytes, divby=128), playout.outer)
                src_s = st.partition_S(cute.make_tensor(tp + (16 << 16) + cute.assume(stage * 128, divby=128), ts2.layout))
                cute.copy(scopy, src_s, rs)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(rs), unroll_full=True):
                    h, col = coords_s[j]
                    val = cutlass.Float32(-1.0e30)
                    if valid[col] >= 0:
                        val = rs[j] * (0.0625 * math.log2(math.e))
                    rs[j] = val
                newmax = rs.load().reduce(cute.ReductionOp.MAX, rowmax, 0)
                newmax = cute.arch.fmax(newmax, cute.arch.shuffle_sync_bfly(newmax, offset=16))
                correction = cute.math.exp2(rowmax - newmax, fastmath=True)
                probs = cute.math.exp2(rs.load() - newmax, fastmath=True)
                blocksum = probs.reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)
                blocksum += cute.arch.shuffle_sync_bfly(blocksum, offset=16)
                rowsum = rowsum * correction + blocksum
                rowmax = newmax
                packed_p = cute.make_rmem_tensor(cute.size(rs), cutlass.Float8E4M3FN)
                packed_p.store((probs * 256.0).to(cutlass.Float8E4M3FN))
                for j in cutlass.range(cute.size(rs) // 16, unroll_full=True):
                    h, col = coords_s[j * 16]
                    rvec = cute.make_tensor(packed_p.iterator + j * 16, cute.make_layout(16))
                    svec = cute.make_tensor(sp.iterator + cute.assume(sp.layout((h, col)), divby=16),
                                            cute.make_layout(16))
                    cute.copy(store128, rvec, svec)
                if block > 0:
                    previous = block - 1
                    cute.arch.mbarrier_wait(empty + previous % 2, (previous // 2) % 2)
                skip_correction = cute.arch.vote_all_sync(correction == 1.0)
                if (block > 0) & (not skip_correction):
                    for tile in cutlass.range(8, unroll_full=True):
                        otile = cute.make_tensor(tp + cute.assume(tile * 64, divby=64), ochunk_layout)
                        src_o = ct.partition_S(otile)
                        dst_o = cstore.get_slice(tid).partition_D(otile)
                        cute.copy(ccopy, src_o, rc)
                        cute.arch.fence_view_async_tmem_load()
                        rc.store(rc.load() * correction)
                        cute.copy(cstore, rc, dst_o)
                        cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                cute.arch.fence_view_async_shared()
                if tid == 0:
                    cute.arch.mbarrier_arrive(ready + stage)
            last = num_blocks - 1
            cute.arch.mbarrier_wait(empty + last % 2, (last // 2) % 2)
            if tid % 32 < 16:
                denom[coords_s[0][0]] = rowsum
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            for tile in cutlass.range(8, unroll_full=True):
                src_o = ot.partition_S(cute.make_tensor(tp + cute.assume(tile * 64, divby=64), ochunk_layout))
                cute.copy(ocopy, src_o, ro)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(ro), unroll_full=True):
                    h, d = coords_o[j]
                    out[qi, h, tile * 64 + d] = (ro[j] / (denom[h] * 256.0)).to(cutlass.BFloat16)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp == 0:
                cute.arch.dealloc_tmem(tp, 512)


    @cute.jit
    def issue_qk(self, qk: cute.TiledMma, aq: cute.Tensor, slayout: cute.Layout,
                 tp: cute.Pointer, sk_base: cute.Tensor, klayout: cute.ComposedLayout,
                 kbytes: cutlass.Constexpr, full: cute.Pointer, done: cute.Pointer,
                 block: cutlass.Int32):
        qm = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN, OperandMajorMode.K,
            OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, self.block_k))
        stage = block % 2
        cute.arch.mbarrier_wait(full + stage, (block // 2) % 2)
        sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
        bk = qm.make_fragment_B(qm.get_slice(0).partition_B(sk))
        ts = cute.make_tensor(tp + (16 << 16) + cute.assume(stage * 128, divby=128), slayout)
        cute.arch.fence_view_async_shared()
        qm.set(tcgen05.Field.ACCUMULATE, False)
        for k in cutlass.range(cute.size(aq, mode=[2]), unroll_full=True):
            cute.gemm(qm, ts, aq[None, None, k], bk[None, None, k], ts)
            qm.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(done + stage)


def make_runner(inputs, block_k=128):
    q = inputs['query'].squeeze(1)
    kv = inputs['kv_cache'].view(-1, 576)
    idx = inputs['block_tables'].view(q.shape[0], -1)
    lens = inputs['seq_lens']
    out = torch.empty((q.shape[0], 64, 512), device=q.device, dtype=torch.bfloat16)
    tensor_map = make_kv_map(kv)
    args = [from_dlpack(t, assumed_align=16) for t in (q, kv, idx, lens, out, tensor_map)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(SparseMLA(block_k), *args, stream)
    def run():
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
    return run
