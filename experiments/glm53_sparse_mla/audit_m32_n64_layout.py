"""Compile-time audit of v082 M32/N64 split-head CTA mapping."""
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
    for width in cutlass.range_constexpr(16, 33, 16):
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


logical, output = [], []
for head_tile in range(2):
    for dp in range(128):
        for j in range(16):
            head = dp % 32
            key = (dp // 32) * 16 + j
            assert (dp, j) == (head + (key // 16) * 32, key % 16)
            logical.append((head_tile * 32 + head, key))
        for tile in range(4):
            for j in range(32):
                head = dp % 32
                channel = (tile // 2) * 256 + (dp // 32) * 64 + (tile % 2) * 32 + j
                expected = (head + ((channel % 256) // 64) * 32,
                            (channel // 256) * 64 + channel % 64)
                assert (dp, tile * 32 + j) == expected
                output.append((head_tile * 32 + head, channel))
assert len(logical) == len(set(logical)) == 64 * 64
assert len(output) == len(set(output)) == 64 * 512
assert 128 + 16 <= 256  # O then S, within each CTA's TMEM allocation.
cute.compile(audit, options="--gpu-arch sm_103a")
print("M32/N64 TWO-CTA SCORE/OUTPUT COVERAGE AUDIT PASS")
