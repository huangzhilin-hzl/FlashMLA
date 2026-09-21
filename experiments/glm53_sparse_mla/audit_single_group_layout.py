"""Compile-time v076 audit: 64-score copies and eight-chunk vector epilogue."""
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode


@cute.jit
def audit():
    mma = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32,
        tcgen05.CtaGroup.ONE, (64, 128))
    placeholder = mma.make_fragment_C(mma.partition_shape_C((64, 128)))
    for width in cutlass.range_constexpr(32, 65, 32):
        physical = cute.make_tensor(placeholder.iterator,
            cute.make_layout((128, width), stride=(1 << 16, 1)))
        copy = tcgen05.make_tmem_copy(cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(width)), cutlass.Float32), physical)
        for tid in cutlass.range_constexpr(128):
            coords = copy.get_slice(tid).partition_D(cute.make_identity_tensor((128, width)))
            assert cute.size(coords) == width
            for j in cutlass.range_constexpr(width):
                assert coords[j][0] == tid and coords[j][1] == j
    print("PHYSICAL COPY COORDINATE AUDIT PASS")


scores, outputs = [], []
for dp in range(128):
    for j in range(64):
        head, key = dp % 64, (dp // 64) * 64 + j
        assert (dp, j) == (head + (key // 64) * 64, key % 64)
        scores.append((head, key))
    for tile in range(8):
        for j in range(32):
            head = dp % 64
            channel = (tile // 4) * 256 + (dp // 64) * 128 + (tile % 4) * 32 + j
            expected = (head + ((channel % 256) // 128) * 64,
                        (channel // 256) * 128 + channel % 128)
            assert (dp, tile * 32 + j) == expected
            outputs.append((head, channel))
assert len(scores) == len(set(scores)) == 64 * 128
assert len(outputs) == len(set(outputs)) == 64 * 512
cute.compile(audit, options="--gpu-arch sm_103a")
print("SINGLE-GROUP SCORE/OUTPUT COVERAGE AUDIT PASS")
