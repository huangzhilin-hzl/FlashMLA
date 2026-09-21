"""Compile-time audit of v071 two-group Layout-G copies and staged-output mapping."""
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
    physical = cute.make_tensor(placeholder.iterator,
        cute.make_layout((128, 32), stride=(1 << 16, 1)))
    copy = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), cutlass.Float32), physical)
    for tid in cutlass.range_constexpr(128):
        coords = copy.get_slice(tid).partition_D(cute.make_identity_tensor((128, 32)))
        assert cute.size(coords) == 32
        for j in cutlass.range_constexpr(32):
            assert coords[j][0] == tid and coords[j][1] == j
    print("WS COPY AUDIT PASS: physical datapath = local thread id, column = element index")


logical = []
output = []
for group in range(2):
    for dp in range(128):
        for j in range(32):
            head = group * 32 + dp % 32
            key = (dp // 32) * 32 + j
            assert (dp, group * 32 + j) == (head % 32 + (key // 32) * 32, (head // 32) * 32 + key % 32)
            logical.append((head, key))
        for tile in range(4):
            for j in range(32):
                head = group * 32 + dp % 32
                channel = (tile // 2) * 256 + (dp // 32) * 64 + (tile % 2) * 32 + j
                expected = (head % 32 + ((channel % 256) // 64) * 32, (head // 32) * 128 + (channel // 256) * 64 + channel % 64)
                assert (dp, group * 128 + tile * 32 + j) == expected
                output.append((head, channel))
assert len(logical) == len(set(logical)) == 64 * 128
assert len(output) == len(set(output)) == 64 * 512
cute.compile(audit, options="--gpu-arch sm_103a")
print("LAYOUT G SCORE/OUTPUT AUDIT PASS: unique full coverage and physical mapping")
