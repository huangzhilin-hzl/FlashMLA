python3 - <<'PY'
from pathlib import Path
import ast,hashlib,json
root=Path('experiments/glm53_sparse_mla')
old=(root/'kernel_v284.py').read_text();s=old
def change(a,b,count=1):
 global s
 assert s.count(a)==count,(a,s.count(a),count)
 s=s.replace(a,b)
change('TENSOR_MAP_BYTES = 512','TENSOR_MAP_BYTES = 384')
change('''    specs = []
    for tensor, tile_rows in ((kv, 1), (q.view(-1, 576), 64)):
        for offset, width, box, swizzle in (
            (0, 576, 128, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B),
            (512, 64, 64, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B)):
            specs.append((tensor, tile_rows, offset, width, box, swizzle))
''','''    # Only Q uses five SW128 panels; KV keeps its original split layout.
    specs = [
        (kv, 1, 0, 576, 128, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B),
        (kv, 1, 512, 64, 64, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B),
        (q.view(-1, 576), 64, 0, 576, 128, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B),
    ]
''')
change('qlayout = cute.tile_to_shape(atom, (64, 512), order=(0, 1))','qlayout = cute.tile_to_shape(atom, (64, 640), order=(0, 1))')
change('        qtail_layout = cute.tile_to_shape(tail_atom, (64, 64), order=(0, 1))\n','')
change('qlayout, klayout, playout, vlayout, qtail_layout, ktail_layout).launch(','qlayout, klayout, playout, vlayout, ktail_layout).launch(')
change('''               vlayout: cute.ComposedLayout, qtail_layout: cute.ComposedLayout,
               ktail_layout: cute.ComposedLayout):''','''               vlayout: cute.ComposedLayout, ktail_layout: cute.ComposedLayout):''')
change('''        sq_tail = alloc.allocate_tensor(cutlass.Float8E4M3FN, qtail_layout.outer,
                                        byte_alignment=128, swizzle=qtail_layout.inner)
''','')
change('cute.local_tile(sq_tail, (64, 32), (0, k))','cute.local_tile(sq, (64, 32), (0, k + 16))')
change('mbarrier_arrive_and_expect_tx(qbar, 64 * 576)','mbarrier_arrive_and_expect_tx(qbar, 64 * 640)',2)
oldq='''                        for col_block in cutlass.range(4, unroll_full=True):
                            raw = cute.recast_ptr(sq.iterator)'''
newq=oldq.replace('range(4','range(5')
change(oldq,newq,2)
removed=[l for l in s.splitlines() if 'load_q_tile(cute.recast_ptr(sq_tail.iterator)' in l]
assert len(removed)==2
s=''.join(l for l in s.splitlines(keepends=True) if 'load_q_tile(cute.recast_ptr(sq_tail.iterator)' not in l)
assert 'sq_tail' not in s and 'qtail_layout' not in s
# KV producer source and full compute math remain text-identical, after Q allocations shift.
a=old[old.index('        if (warp >= 8) & (warp < 12):'):old.index('        elif warp == 12:')]
b=s[s.index('        if (warp >= 8) & (warp < 12):'):s.index('        elif warp == 12:')]
assert a==b
s='"""Iteration 298: Q-only SW128 padding within the prior196KiB carveout; based on v284.\n'+s[s.index('\n')+1:]
s=s.replace('"v284 requires','"v298 requires')
ast.parse(s);p=root/'kernel_v298.py';assert not p.exists();p.write_text(s)
proof={'parent':'v284','source_sha256':hashlib.sha256(s.encode()).hexdigest(),
 'producer_source_identical':True,'global_q_and_kv_stride':576,'q_shared_bytes':40960,
 'extra_shared_bytes':4096,'expected_dynamic_shared_bytes':199016,
 'expected_dynamic_plus_driver_bytes':200040,'prior_carveout_bytes':200704,
 'fits_prior_carveout_by_geometry':True,'q_tma_requests':5,'kv_gather4_requests_per_stage':160,
 'qk_k32_tiles':18,'tensor_map_bytes':384,'candidate_launched':False,
 'expected_runtime_carveout_pending':True}
dest=root/'artifacts/q_only_sw128_compile';dest.mkdir(parents=True,exist_ok=True)
(dest/'source_controls.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
PY
