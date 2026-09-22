"""Offline static CuTe audit of normal-M64 TMEM load/thread coordinates."""
import json
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode

records = []


@cute.jit
def audit():
    mma = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN, OperandMajorMode.K,
        OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, 64))
    acc = mma.make_fragment_C(mma.partition_shape_C((64, 64)))
    first = acc[((None, None), 0, 0)]
    layout = cute.make_layout((first.shape[0], 64),
                             stride=(first.stride[0], first.stride[1]))
    tile = cute.make_tensor(acc.iterator, layout)
    load = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32)),
        cutlass.Float32), tile)
    red = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.LdRed16x32bx2Op(tcgen05.copy.Repetition(32),
            redOp=tcgen05.copy.TmemLoadRedOp.MAX, nan=True, half_split_off=32),
        cutlass.Float32), tile)
    for tid in cutlass.range_constexpr(128):
        coords = load.get_slice(tid).partition_D(cute.make_identity_tensor((64, 64)))
        redcoords = red.get_slice(tid).partition_D(cute.make_identity_tensor((64, 64)))
        assert cute.size(coords) == 32
        row = []
        for j in cutlass.range_constexpr(32):
            assert coords[j] == redcoords[j]
            row.append(coords[j])
        records.append(row)


if __name__ == '__main__':
    cute.compile(audit, options='--gpu-arch sm_103a')
    flattened = [tuple(coord) for row in records for coord in row]
    assert len(flattened) == len(set(flattened)) == 4096
    for tid, row in enumerate(records):
        assert row == [(tid // 32 * 16 + tid % 16, tid % 32 // 16 * 32 + j)
                       for j in range(32)]
    print(json.dumps({'offline': True, 'mapping': records,
        'head': '(tid // 32) * 16 + tid % 16',
        'column': '((tid % 32) // 16) * 32 + j',
        'ld_and_ldred_coordinate_match': True,
        'explicit_ldred_half_split_off': 32,
        'scope': 'Static coordinates only; emitted LD.RED immediate also requires inspection.',
        'covered_unique_elements': len(flattened)}, indent=2))
