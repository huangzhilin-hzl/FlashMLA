"""Compile-time audit of v038/v039 one-head Layout-E copies and corrected output mapping."""
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


# This maps the audited physical coordinates to logical (head, key) pairs.
logical = [(tid % 64, (tid // 64) * 64 + group * 32 + j)
           for group in range(2) for tid in range(128) for j in range(32)]
assert len(logical) == len(set(logical)) == 64 * 128
assert set(logical) == {(h, k) for h in range(64) for k in range(128)}
cute.compile(audit, options="--gpu-arch sm_103a")
print("WS LOGICAL COVERAGE PASS: every head/key element is covered exactly once")

# Each N256 PV tile maps channel n to DP=head+(n//128)*64, col=n%128.
# The epilogue reads 64 physical columns from both datapath halves.
output = []
for group in range(2):
    for tile in range(2):
        for h in range(64):
            for d in range(128):
                channel = group * 256 + tile * 64 + (d // 64) * 128 + d % 64
                read_phys = (h + (d // 64) * 64, group * 128 + tile * 64 + d % 64)
                actual_phys = (h + ((channel % 256) // 128) * 64, (channel // 256) * 128 + channel % 128)
                assert read_phys == actual_phys
                output.append((h, channel))
assert len(output) == len(set(output)) == 64 * 512
print("WS EPILOGUE AUDIT PASS: all output coordinates match physical N256 MMA layout")
