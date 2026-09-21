"""Print compile-time Layout-E epilogue register coordinates; no GPU launch."""
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode

@cute.jit
def inspect():
    mma = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32,
        tcgen05.CtaGroup.ONE, (64, 128))
    holder = mma.make_fragment_C(mma.partition_shape_C((64, 128)))
    tensor = cute.make_tensor(holder.iterator,
        cute.make_layout((64, (64, 2)), stride=(1 << 16, (1, 64 << 16))))
    copy = tcgen05.make_tmem_copy(bw.get_tmem_load_op(
        (64, 64, 128), utils.LayoutEnum.ROW_MAJOR, cutlass.Float32,
        cutlass.Float32, (64, 64), False), tensor)
    for tid in cutlass.range_constexpr(0, 128, 16):
        coords = copy.get_slice(tid).partition_D(cute.make_identity_tensor((64, 128)))
        print('THREAD', tid, 'SHAPE', coords.shape)
        for j in cutlass.range_constexpr(16):
            print(j, coords[j])

cute.compile(inspect, options='--gpu-arch sm_103a')
