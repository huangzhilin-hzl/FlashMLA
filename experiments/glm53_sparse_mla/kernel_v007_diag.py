"""Diagnostic: inspect the 16x32bx2 TMEM score distribution.

One CTA owns one query and 256 output channels. QK is duplicated across the
two output CTAs as a deliberate initial resource tradeoff. No input expansion,
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


class SparseMLA:
    def __init__(self, block_k=64):
        self.block_k = block_k

    @cute.jit
    def __call__(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
                 lens: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
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
        self.kernel(q, kv, idx, lens, out, qk, pv,
                    qlayout, klayout, playout, vlayout).launch(
                        grid=(q.shape[0], 2, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
               lens: cute.Tensor, out: cute.Tensor, qk: cute.TiledMma,
               pv: cute.TiledMma, qlayout: cute.ComposedLayout,
               klayout: cute.ComposedLayout, playout: cute.ComposedLayout,
               vlayout: cute.ComposedLayout):
        tid, _, _ = cute.arch.thread_idx()
        qi, vi, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        alloc = utils.SmemAllocator()
        sq = alloc.allocate_tensor(cutlass.Float8E4M3FN, qlayout.outer, byte_alignment=128, swizzle=qlayout.inner)
        sk = alloc.allocate_tensor(cutlass.Float8E4M3FN, klayout.outer, byte_alignment=128, swizzle=klayout.inner)
        sp = alloc.allocate_tensor(cutlass.Float8E4M3FN, playout.outer, byte_alignment=128, swizzle=playout.inner)
        sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
        sv = cute.local_tile(sk_transposed, (256, self.block_k), (vi, 0))
        ss = alloc.allocate_tensor(cutlass.Float32, cute.make_layout((64, self.block_k)))
        alpha = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        denom = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        valid = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(self.block_k))
        bar = alloc.allocate_array(cutlass.Int64, 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(256, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                        ptr_to_buffer_holding_addr=holding)
        sq_part = qk.get_slice(0).partition_A(sq)
        sk_part = qk.get_slice(0).partition_B(sk)
        sp_part = pv.get_slice(0).partition_A(sp)
        sv_part = pv.get_slice(0).partition_B(sv)
        aq = qk.make_fragment_A(sq_part)
        bk = qk.make_fragment_B(sk_part)
        ap = pv.make_fragment_A(sp_part)
        bv = pv.make_fragment_B(sv_part)
        cs = qk.make_fragment_C(qk.partition_shape_C((64, self.block_k)))
        co = pv.make_fragment_C(pv.partition_shape_C((64, 256)))
        assert tcgen05.find_tmem_tensor_col_offset(cs) <= 64
        assert tcgen05.find_tmem_tensor_col_offset(co) <= 256
        # PTX Layout F (M64) supports lane alignments 0 and 16.
        # S occupies lanes [16:32] of each warp; O occupies [0:16].
        # They overlap columns but never the same physical cells.
        ts = cute.make_tensor(tp + (16 << 16), cs.layout)
        to = cute.make_tensor(tp, co.layout)
        ts2 = ts[((None, None), 0, 0)]
        to2 = to[((None, None), 0, 0)]
        scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(self.block_k // 2)),
            cutlass.Float32), ts2)
        print("S copy", scopy)
        for t in cutlass.range_constexpr(4):
            cc = scopy.get_slice(t * 16).partition_D(cute.make_identity_tensor((64, self.block_k)))
            print("Coordinates", t * 16, cc.layout, cc[0], cc[1], cc[15], cc[31])
        ocopy = tcgen05.make_tmem_copy(
            bw.get_tmem_load_op((64, 256, self.block_k), utils.LayoutEnum.ROW_MAJOR,
                cutlass.Float32, cutlass.Float32, (64, 256), False), to2)
        st = scopy.get_slice(tid)
        ot = ocopy.get_slice(tid)
        src_s = st.partition_S(ts2)
        dst_s = st.partition_D(ss)
        rs = cute.make_fragment_like(dst_s, cutlass.Float32)
        src_o = ot.partition_S(to2)
        coords_o = ot.partition_D(cute.make_identity_tensor((64, 256)))
        ro = cute.make_fragment_like(coords_o, cutlass.Float32)
        ostore = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(64)), cutlass.Float32), to2)
        dst_o = ostore.get_slice(tid).partition_D(to2)
        store128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                       cutlass.Float8E4M3FN, num_bits_per_copy=128)
        load128 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float8E4M3FN, num_bits_per_copy=128)
        for row_block in cutlass.range(2, unroll_full=True):
            h = tid // 4 + row_block * 32
            for col_block in cutlass.range(9, unroll_full=True):
                d = (tid % 4) * 16 + col_block * 64
                g = cute.make_tensor(q.iterator + cute.assume(qi * 64 * 576 + h * 576 + d, divby=16),
                                     cute.make_layout(16))
                sh = cute.make_tensor(sq.iterator + cute.assume(sq.layout((h, d)), divby=16), cute.make_layout(16))
                cute.copy(load128, g, sh)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        rowmax = cutlass.Float32(-1.0e30)
        rowsum = cutlass.Float32(0.0)
        phase = cutlass.Int32(0)
        nvalid = lens[qi]
        for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
            for j in cutlass.range(tid, self.block_k, 128):
                pos = block * self.block_k + j
                slot = cutlass.Int32(-1)
                if pos < nvalid:
                    slot = idx[qi, pos]
                valid[j] = slot
            cute.arch.barrier()
            for row_block in cutlass.range(self.block_k // 32, unroll_full=True):
                j = tid // 4 + row_block * 32
                slot = valid[j]
                for col_block in cutlass.range(9, unroll_full=True):
                    d = (tid % 4) * 16 + col_block * 64
                    sh = cute.make_tensor(sk.iterator + cute.assume(sk.layout((j, d)), divby=16), cute.make_layout(16))
                    if slot >= 0:
                        g = cute.make_tensor(kv.iterator + cute.assume(slot * 576 + d, divby=16), cute.make_layout(16))
                        cute.copy(load128, g, sh)
                    else:
                        for z in cutlass.range(16, unroll_full=True):
                            sh[z] = cutlass.Float8E4M3FN(0.0)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            if warp == 0:
                qk.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range(cute.size(aq, mode=[2]), unroll_full=True):
                    cute.gemm(qk, ts, aq[None, None, k], bk[None, None, k], ts)
                    qk.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(bar)
            cute.arch.mbarrier_wait(bar, phase)
            phase ^= 1
            cute.copy(scopy, src_s, rs)
            cute.arch.fence_view_async_tmem_load()
            for j in cutlass.range(cute.size(rs), unroll_full=True):
                dst_s[j] = rs[j]
            cute.arch.barrier()
            if tid < 64:
                rp = cute.make_rmem_tensor(self.block_k, cutlass.Float32)
                for j in cutlass.range(self.block_k, unroll_full=True):
                    v = cutlass.Float32(-1.0e30)
                    if valid[j] >= 0:
                        v = ss[tid, j] * (0.0625 * math.log2(math.e))
                    rp[j] = v
                newmax = rp.load().reduce(cute.ReductionOp.MAX, rowmax, 0)
                correction = cute.math.exp2(rowmax - newmax, fastmath=True)
                probs = cute.math.exp2(rp.load() - newmax, fastmath=True)
                blocksum = probs.reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0)
                rowsum = rowsum * correction + blocksum
                rowmax = newmax
                packed_p = cute.make_rmem_tensor(self.block_k, cutlass.Float8E4M3FN)
                packed_p.store((probs * 256.0).to(cutlass.Float8E4M3FN))
                for j in cutlass.range(self.block_k // 16, unroll_full=True):
                    rvec = cute.make_tensor(packed_p.iterator + j * 16, cute.make_layout(16))
                    svec = cute.make_tensor(sp.iterator + cute.assume(sp.layout((tid, j * 16)), divby=16),
                                            cute.make_layout(16))
                    cute.copy(store128, rvec, svec)
                alpha[tid] = correction
                denom[tid] = rowsum
            cute.arch.barrier()
            if block > 0:
                cute.copy(ocopy, src_o, ro)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(ro), unroll_full=True):
                    h, d = coords_o[j]
                    ro[j] = ro[j] * alpha[h]
                cute.copy(ostore, ro, dst_o)
                cute.arch.fence_view_async_tmem_store()
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            if warp == 0:
                pv.set(tcgen05.Field.ACCUMULATE, block > 0)
                for k in cutlass.range(cute.size(ap, mode=[2]), unroll_full=True):
                    cute.gemm(pv, to, ap[None, None, k], bv[None, None, k], to)
                    pv.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(bar)
            cute.arch.mbarrier_wait(bar, phase)
            phase ^= 1
            cute.arch.barrier()
        cute.copy(ocopy, src_o, ro)
        cute.arch.fence_view_async_tmem_load()
        for j in cutlass.range(cute.size(ro), unroll_full=True):
            h, d = coords_o[j]
            out[qi, h, vi * 256 + d] = (ro[j] / (denom[h] * 256.0)).to(cutlass.BFloat16)
        cute.arch.barrier()
        if warp == 0:
            cute.arch.dealloc_tmem(tp, 256)


def make_runner(inputs, block_k=64):
    q = inputs['query'].squeeze(1)
    kv = inputs['kv_cache'].view(-1, 576)
    idx = inputs['block_tables'].view(q.shape[0], -1)
    lens = inputs['seq_lens']
    out = torch.empty((q.shape[0], 64, 512), device=q.device, dtype=torch.bfloat16)
    args = [from_dlpack(t, assumed_align=16) for t in (q, kv, idx, lens, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(SparseMLA(block_k), *args, stream)
    def run():
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
    return run
