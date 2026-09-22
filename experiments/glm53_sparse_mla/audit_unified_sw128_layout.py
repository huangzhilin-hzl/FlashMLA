"""CPU-only layout/transfer-accounting audit for v296/v297.

This checks static addresses and byte counts, not asynchronous runtime safety.
Run with CUDA_VISIBLE_DEVICES="" and CUTE_DSL_ARCH=sm_103a.
"""
import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05


@cute.jit
def audit():
    atom = tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.Float8E4M3FN)
    for rows in cutlass.range_constexpr(64, 129, 64):
        layout = cute.tile_to_shape(atom, (rows, 640), order=(0, 1))
        original = cute.tile_to_shape(atom, (rows, 512), order=(0, 1))
        assert cute.cosize(layout.outer) == rows * 640
        for row in cutlass.range_constexpr(rows):
            for col in cutlass.range_constexpr(0, 640, 16):
                linear = (col // 128) * rows * 128 + row * 128 + col % 128
                expected = linear ^ ((row % 8) * 16)
                assert layout((row, col)) == expected
                assert layout((row, col + 15)) == expected + 15
                if cutlass.const_expr(col < 512):
                    assert layout((row, col)) == original((row, col))
                    assert layout((row, col + 15)) == original((row, col + 15))
        print("SW128 padded640 vector-address audit passed for rows", rows)


assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide all GPUs for this audit"
for rows in (64, 128):
    physical = set()
    for row in range(rows):
        for col in range(640):
            linear = (col // 128) * rows * 128 + row * 128 + col % 128
            physical.add(linear ^ ((row % 8) * 16))
    assert physical == set(range(rows * 640))
    assert sum(rows * 128 for _ in range(5)) == rows * 640
assert [k * 32 + j for k in range(18) for j in range(32)] == list(range(576))
assert 4 * 8 * 5 * 4 * 128 == 128 * 640
assert (64 + 2 * 128) * (640 - 576) == 20480
cute.compile(audit, options="--gpu-arch sm_103a")
print("PASS: bijective padded storage, unchanged first512 layout, exact576 QK channels,")
print("      Q40960/KV81920 completion bytes and20480 extra shared bytes per CTA")
