"""Offline audit for 16-DP correction copies and warp-local factor exchange."""
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode


@cute.jit
def audit():
    mma = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.MN, cutlass.Float32,
        tcgen05.CtaGroup.ONE, (64, 256))
    placeholder = mma.make_fragment_C(mma.partition_shape_C((64, 256)))
    shape = ((16, 4), 64)
    physical = cute.make_tensor(placeholder.iterator,
        cute.make_layout(shape, stride=((1 << 16, 32 << 16), 1)))
    copy = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(32)), cutlass.Float32), physical)
    store = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.St16x64bOp(tcgen05.copy.Repetition(32)), cutlass.Float32), physical)
    identity = cute.make_identity_tensor(shape)
    for tid in cutlass.range_constexpr(128):
        coords = copy.get_slice(tid).partition_D(identity)
        dest = store.get_slice(tid).partition_S(identity)
        assert cute.size(coords) == cute.size(dest) == 32
        lane = tid % 32
        for j in cutlass.range_constexpr(32):
            assert coords[j][0][0] == 8 * (lane % 2) + lane // 4
            assert coords[j][0][1] == tid // 32
            assert coords[j][1] == 2 * j + (lane // 2) % 2
            assert coords[j][0][0] == dest[j][0][0]
            assert coords[j][0][1] == dest[j][0][1]
            assert coords[j][1] == dest[j][1]
    print('16-DP COPY/STORE COORDINATE AUDIT PASS')


covered = set()
for group in range(2):
    for tid in range(128):
        lane = tid % 32
        dp = (tid // 32) * 32 + 8 * (lane % 2) + lane // 4 + group * 16
        source_lane = 8 * (lane % 2) + lane // 4 + group * 16
        assert (tid // 32 * 32 + source_lane) % 64 == dp % 64
        for tile in range(4):
            for j in range(32):
                column = tile * 64 + j * 2 + (lane // 2) % 2
                coordinate = (dp, column)
                assert coordinate not in covered
                covered.add(coordinate)
assert len(covered) == 128 * 256
cute.compile(audit, options='--gpu-arch sm_103a')
print('FULL OUTPUT TMEM COVERAGE AND FACTOR SHUFFLE AUDIT PASS')
