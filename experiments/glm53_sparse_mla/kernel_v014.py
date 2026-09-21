"""Iteration 014: specialize loader and compute warpgroups with a two-stage pipeline.

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
                        grid=(q.shape[0], 2, 1), block=(256, 1, 1), stream=stream)

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
        kbytes = cute.cosize(klayout.outer)
        sk_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * kbytes),
                                        byte_alignment=128, swizzle=klayout.inner)
        sp = alloc.allocate_tensor(cutlass.Float8E4M3FN, playout.outer, byte_alignment=128, swizzle=playout.inner)
        alpha = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
        # Alpha is needed inside the loop; denominators are needed only at exit.
        denom = alpha
        valid_base = alloc.allocate_tensor(cutlass.Int32, cute.make_layout(2 * self.block_k))
        bar = alloc.allocate_array(cutlass.Int64, 1)
        holding = alloc.allocate_array(cutlass.Int32, 1)
        full = alloc.allocate_array(cutlass.Int64, 2)
        empty = alloc.allocate_array(cutlass.Int64, 2)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            for stage in cutlass.range(2, unroll_full=True):
                cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init(empty + stage, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 0:
            cute.arch.alloc_tmem(256, holding)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.barrier()
        store128 = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                       cutlass.Float8E4M3FN, num_bits_per_copy=128)
        load128 = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float8E4M3FN, num_bits_per_copy=128)
        if tid >= 128:
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
        if tid >= 128:
            load_tid = tid - 128
            nvalid = lens[qi]
            for block in cutlass.range(cute.ceil_div(nvalid, self.block_k)):
                stage = block % 2
                if block >= 2:
                    cute.arch.mbarrier_wait(empty + stage, ((block // 2) - 1) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                self.gather(kv, idx, qi, block, nvalid, sk, valid, load128, store128, load_tid)
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if load_tid == 0:
                    cute.arch.mbarrier_arrive(full + stage)
        else:
            tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                                            ptr_to_buffer_holding_addr=holding)
            sq_part = qk.get_slice(0).partition_A(sq)
            sp_part = pv.get_slice(0).partition_A(sp)
            aq = qk.make_fragment_A(sq_part)
            ap = pv.make_fragment_A(sp_part)
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
            ostore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(16)), cutlass.Float32), ochunk)
            rowmax = cutlass.Float32(-1.0e30)
            rowsum = cutlass.Float32(0.0)
            phase = cutlass.Int32(0)
            nvalid = lens[qi]
            num_blocks = cute.ceil_div(nvalid, self.block_k)
            for block in cutlass.range(num_blocks):
                stage = block % 2
                cute.arch.mbarrier_wait(full + stage, (block // 2) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                sk_part = qk.get_slice(0).partition_B(sk)
                bk = qk.make_fragment_B(sk_part)
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                sv = cute.local_tile(sk_transposed, (256, self.block_k), (vi, 0))
                sv_part = pv.get_slice(0).partition_B(sv)
                bv = pv.make_fragment_B(sv_part)
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
                if tid % 32 < 16:
                    h = coords_s[0][0]
                    alpha[h] = correction
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if block > 0:
                    for tile in cutlass.range(4, unroll_full=True):
                        otile = cute.make_tensor(tp + cute.assume(tile * 64, divby=64), ochunk_layout)
                        src_o = ot.partition_S(otile)
                        dst_o = ostore.get_slice(tid).partition_D(otile)
                        cute.copy(ocopy, src_o, ro)
                        cute.arch.fence_view_async_tmem_load()
                        for j in cutlass.range(cute.size(ro), unroll_full=True):
                            h, d = coords_o[j]
                            ro[j] = ro[j] * alpha[h]
                        cute.copy(ostore, ro, dst_o)
                        cute.arch.fence_view_async_tmem_store()
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
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
                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                if tid == 0:
                    cute.arch.mbarrier_arrive(empty + stage)
            if tid % 32 < 16:
                denom[coords_s[0][0]] = rowsum
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            for tile in cutlass.range(4, unroll_full=True):
                src_o = ot.partition_S(cute.make_tensor(tp + cute.assume(tile * 64, divby=64), ochunk_layout))
                cute.copy(ocopy, src_o, ro)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(ro), unroll_full=True):
                    h, d = coords_o[j]
                    out[qi, h, vi * 256 + tile * 64 + d] = (ro[j] / (denom[h] * 256.0)).to(cutlass.BFloat16)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp == 0:
                cute.arch.dealloc_tmem(tp, 256)

    @cute.jit
    def gather(self, kv: cute.Tensor, idx: cute.Tensor, qi: cutlass.Int32,
               block: cutlass.Int32, nvalid: cutlass.Int32, sk: cute.Tensor,
               valid: cute.Tensor, load128: cute.CopyAtom, store128: cute.CopyAtom,
               tid: cutlass.Int32):
        for j in cutlass.range(tid, self.block_k, 128):
            pos = block * self.block_k + j
            slot = cutlass.Int32(-1)
            if pos < nvalid:
                slot = idx[qi, pos]
            valid[j] = slot
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
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
                    zero = cute.make_rmem_tensor(16, cutlass.Float8E4M3FN)
                    zero.fill(cutlass.Float8E4M3FN(0.0))
                    cute.copy(store128, zero, sh)
        cute.arch.cp_async_commit_group()


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
