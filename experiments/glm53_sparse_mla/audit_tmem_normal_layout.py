"""Offline static CuTe audit of normal-M64 TMEM load/thread coordinates."""
import argparse
import json
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as bw
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode

records = []


@cute.jit
def audit(columns: cutlass.Constexpr):
    mma = bw.make_trivial_tiled_mma(cutlass.Float8E4M3FN, OperandMajorMode.K,
        OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (64, columns))
    acc = mma.make_fragment_C(mma.partition_shape_C((64, columns)))
    first = acc[((None, None), 0, 0)]
    layout = cute.make_layout((first.shape[0], columns),
                             stride=(first.stride[0], first.stride[1]))
    tile = cute.make_tensor(acc.iterator, layout)
    load = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(columns // 2)),
        cutlass.Float32), tile)
    red = tcgen05.make_tmem_copy(cute.make_copy_atom(
        tcgen05.copy.LdRed16x32bx2Op(tcgen05.copy.Repetition(columns // 2),
            redOp=tcgen05.copy.TmemLoadRedOp.MAX, nan=True, half_split_off=columns // 2),
        cutlass.Float32), tile)
    for tid in cutlass.range_constexpr(128):
        coords = load.get_slice(tid).partition_D(cute.make_identity_tensor((64, columns)))
        redcoords = red.get_slice(tid).partition_D(cute.make_identity_tensor((64, columns)))
        assert cute.size(coords) == columns // 2
        row = []
        for j in cutlass.range_constexpr(columns // 2):
            assert coords[j] == redcoords[j]
            row.append(coords[j])
        records.append(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--columns', type=int, choices=(64, 128), default=64)
    columns = parser.parse_args().columns
    cute.compile(audit, columns, options='--gpu-arch sm_103a')
    flattened = [tuple(coord) for row in records for coord in row]
    assert len(flattened) == len(set(flattened)) == 64 * columns
    for tid, row in enumerate(records):
        assert row == [(tid // 32 * 16 + tid % 16, tid % 32 // 16 * (columns // 2) + j)
                       for j in range(columns // 2)]
    print(json.dumps({'offline': True, 'columns': columns, 'mapping': records,
        'head': '(tid // 32) * 16 + tid % 16',
        'column': '((tid % 32) // 16) * (columns // 2) + j',
        'ld_and_ldred_coordinate_match': True,
        'explicit_ldred_half_split_off': columns // 2,
        'scope': 'Static coordinates only; emitted LD.RED immediate also requires inspection.',
        'covered_unique_elements': len(flattened)}, indent=2))
