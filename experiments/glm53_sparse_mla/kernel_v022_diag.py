"""Diagnostic: isolate v022 warp-local QK mapping.

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
from cutlass.cute.nvgpu.tcgen05.helpers import smem_descriptor_to_int


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
        assert block_k == 128, "v022 requires --block-k 128"
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
                        grid=(q.shape[0], 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, q: cute.Tensor, kv: cute.Tensor, idx: cute.Tensor,
               lens: cute.Tensor, out: cute.Tensor, qk: cute.TiledMma,
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
        sp = alloc.allocate_tensor(cutlass.Float8E4M3FN, playout.outer, byte_alignment=128, swizzle=playout.inner)
        denom = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(64))
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
            cute.arch.alloc_tmem(512, holding)
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
            for block in cutlass.range(1):
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
            # PTX Layout E / CUTLASS tmem_frg_ws<M64>: two N halves
            # occupy DP[0:64] and DP[64:128], sharing column addresses.
            # O uses columns [0:256]; S uses [256:320], all inside 512.
            # Each warp explicitly owns 16 consecutive heads. Load both
            # Layout-E N halves into the same thread's registers; otherwise
            # automatic tiling gives two heads per thread and needs a different
            # reduction scheme. The two lanes per head exchange via XOR16.
            head_base = warp * 16
            ts2 = cute.make_tensor(tp + cute.assume(256 + head_base * (1 << 16), divby=16),
                                   cute.make_layout((16, 64), stride=(1 << 16, 1)))
            scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)),
                cutlass.Float32), ts2)
            ochunk_layout = cute.make_layout((16, 64), stride=(1 << 16, 1))
            ochunk = cute.make_tensor(tp + cute.assume(head_base * (1 << 16), divby=16), ochunk_layout)
            ocopy = tcgen05.make_tmem_copy(
                bw.get_tmem_load_op((64, 64, self.block_k), utils.LayoutEnum.ROW_MAJOR,
                    cutlass.Float32, cutlass.Float32, (64, 64), False), ochunk)
            st = scopy.get_slice(tid % 32)
            ot = ocopy.get_slice(tid % 32)
            coords_s = st.partition_D(cute.make_identity_tensor((16, 64)))
            rs = cute.make_rmem_tensor(64, cutlass.Float32)
            coords_o = ot.partition_D(cute.make_identity_tensor((16, 64)))
            ro = cute.make_fragment_like(coords_o, cutlass.Float32)
            ccopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            cstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
                tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(32)), cutlass.Float32), ochunk)
            ct = ccopy.get_slice(tid % 32)
            ccoords = ct.partition_D(cute.make_identity_tensor((16, 64)))
            rc = cute.make_fragment_like(ccoords, cutlass.Float32)
            rowmax = cutlass.Float32(-1.0e30)
            rowsum = cutlass.Float32(0.0)
            phase = cutlass.Int32(0)
            nvalid = lens[qi]
            num_blocks = 1
            for block in cutlass.range(num_blocks):
                stage = block % 2
                cute.arch.mbarrier_wait(full + stage, (block // 2) % 2)
                sk = cute.make_tensor(sk_base.iterator + cute.assume(stage * kbytes, divby=128), klayout.outer)
                valid = cute.make_tensor(valid_base.iterator + stage * self.block_k, cute.make_layout(self.block_k))
                sk_transposed = cute.make_tensor(sk.iterator, cute.select(sk.layout, mode=[1, 0]))
                cute.arch.fence_view_async_shared()
                if warp == 0:
                    with cute.arch.elect_one():
                        for k in cutlass.range(18, unroll_full=True):
                            qa = cute.local_tile(sq, (64, 32), (0, k))
                            kb = cute.local_tile(sk, (128, 32), (0, k))
                            mma_ws(tp + 256, qa, kb, 128, False, k > 0)
                        tcgen05.commit(bar)
                cute.arch.mbarrier_wait(bar, phase)
                phase ^= 1
                for half in cutlass.range(2, unroll_full=True):
                    s_half = cute.make_tensor(ts2.iterator + half * (64 << 16), ts2.layout)
                    r_half = cute.make_tensor(rs.iterator + half * 32, cute.make_layout(coords_s.shape))
                    cute.copy(scopy, st.partition_S(s_half), r_half)
                cute.arch.fence_view_async_tmem_load()
                for j in cutlass.range(cute.size(rs), unroll_full=True):
                    h, col = coords_s[j % 32]
                    out[qi, head_base + h, col + (j // 32) * 64] = rs[j]
                if qi == 0:
                    if (tid == 0) | (tid == 16) | (tid == 32) | (tid == 64):
                        cute.printf("tid %d Hbase %d first %d,%d last %d,%d\n", tid, head_base,
                            coords_s[0][0], coords_s[0][1], coords_s[31][0], coords_s[31][1])
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            if warp == 0:
                cute.arch.dealloc_tmem(tp, 512)

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


def make_runner(inputs, block_k=128):
    q = inputs['query'].squeeze(1)
    kv = inputs['kv_cache'].view(-1, 576)
    idx = inputs['block_tables'].view(q.shape[0], -1)
    lens = inputs['seq_lens']
    out = torch.full((q.shape[0], 64, 512), float("nan"), device=q.device, dtype=torch.float32)
    args = [from_dlpack(t, assumed_align=16) for t in (q, kv, idx, lens, out)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(SparseMLA(block_k), *args, stream)
    def run():
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out
    return run
