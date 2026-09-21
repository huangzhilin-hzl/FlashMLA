"""Compile-time audit of v056 four-group Layout-E copies and staged-output mapping."""
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
        cute.make_layout((128, 16), stride=(1 << 16, 1)))
    copy = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), cutlass.Float32), physical)
    for tid in cutlass.range_constexpr(128):
        coords = copy.get_slice(tid).partition_D(cute.make_identity_tensor((128, 16)))
        assert cute.size(coords) == 16
        for j in cutlass.range_constexpr(16):
            assert coords[j][0] == tid and coords[j][1] == j
    print("WS COPY AUDIT PASS: physical datapath = local thread id, column = element index")


# This maps the audited physical coordinates to logical (head, key) pairs.
logical = [(tid % 64, (tid // 64) * 64 + group * 16 + j)
           for group in range(4) for tid in range(128) for j in range(16)]
assert len(logical) == len(set(logical)) == 64 * 128
assert set(logical) == {(h, k) for h in range(64) for k in range(128)}
cute.compile(audit, options="--gpu-arch sm_103a")
print("WS LOGICAL COVERAGE PASS: every head/key element is covered exactly once")


output = []
for group in range(4):
    for dp in range(128):
        for tile in range(2):
            for j in range(32):
                head = dp % 64
                channel = (group // 2) * 256 + (group % 2) * 64 + (dp // 64) * 128 + tile * 32 + j
                read_phys = (dp, group * 64 + tile * 32 + j)
                mma_phys = (head + ((channel % 256) // 128) * 64, (channel // 256) * 128 + channel % 128)
                assert read_phys == mma_phys
                output.append((head, channel))
assert len(output) == len(set(output)) == 64 * 512
print("FOUR-GROUP OUTPUT AUDIT PASS: exact N256 physical mapping and unique full coverage")
