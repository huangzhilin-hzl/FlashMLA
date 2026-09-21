"""Compile-time coordinate audit for v035/v036; does not launch GPU work."""
import argparse
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode


@cute.jit
def audit(groups: cutlass.Constexpr):
    qk = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32,
        tcgen05.CtaGroup.ONE, (64, 128))
    pv = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.MN, cutlass.Float32,
        tcgen05.CtaGroup.ONE, (64, 256), a_source=tcgen05.OperandSource.TMEM)
    cs = qk.make_fragment_C(qk.partition_shape_C((64, 128)))
    ap = pv.make_fragment_A(pv.partition_shape_A((64, 128)))
    co_default = pv.make_fragment_C(pv.partition_shape_C((64, 512)))
    co = cute.make_tensor(co_default.iterator, cute.make_layout(co_default.shape,
        stride=(co_default.stride[0], co_default.stride[1], 16 << 16)))
    assert tcgen05.find_tmem_tensor_col_offset(co) == 256
    full_s = cs[((None, None), 0, 0)]
    s2 = cute.make_tensor(full_s.iterator, cute.make_layout(
        (full_s.shape[0], 128 // groups), stride=(full_s.stride[0], full_s.stride[1])))
    pfirst = ap[((None, None), 0, 0)]
    p2 = cute.make_tensor(ap.iterator, cute.make_layout(
        (pfirst.shape[0], 128 // groups), stride=(pfirst.stride[0], pfirst.stride[1])))
    scopy = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(64 // groups)), cutlass.Float32), s2)
    pstore = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(16 // groups)), cutlass.Float8E4M3FN), p2)
    identity = cute.make_identity_tensor((64, 128 // groups))
    print("score_layout", cs.layout)
    print("probability_fragment", ap.layout)
    print("probability_view", p2.layout)
    print("output_layout", co.layout)
    for tid in cutlass.range_constexpr(128):
        source = scopy.get_slice(tid).partition_D(identity)
        dest = pstore.get_slice(tid).partition_S(identity)
        assert cute.size(source) == cute.size(dest) == 64 // groups
        for j in cutlass.range_constexpr(64 // groups):
            src = source[j]
            dst = dest[j]
            assert src[0] == dst[0] and src[1] == dst[1], "P packing changes logical element order"
    print("LAYOUT COORDINATE AUDIT PASS:", groups, "group(s), 128 threads x", 64 // groups, "values per group")


parser = argparse.ArgumentParser()
parser.add_argument("--compute-groups", type=int, choices=(1, 2), default=1)
args = parser.parse_args()
cute.compile(audit, args.compute_groups, options="--gpu-arch sm_103a")
