python3 - <<'PY'
from pathlib import Path
import ast, hashlib, json
root=Path('experiments/glm53_sparse_mla')
proof={}
for parent,version in [('v284','v296'),('v287','v297')]:
 old=(root/f'kernel_{parent}.py').read_text()
 s=old;changes=[]
 def change(a,b,count=1):
  global s
  assert s.count(a)==count,(version,a,s.count(a),count)
  s=s.replace(a,b);changes.append((a,b,count))
 change('TENSOR_MAP_BYTES = 512','TENSOR_MAP_BYTES = 256')
 change('''    specs = []
    for tensor, tile_rows in ((kv, 1), (q.view(-1, 576), 64)):
        for offset, width, box, swizzle in (
            (0, 576, 128, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B),
            (512, 64, 64, cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B)):
            specs.append((tensor, tile_rows, offset, width, box, swizzle))
''','''    # Five 128-byte boxes cover 576 channels; TMA zero-fills the last 64 bytes.
    # Global rows and their 576-byte stride are unchanged.
    specs = [(tensor, tile_rows, 0, 576, 128,
              cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B)
             for tensor, tile_rows in ((kv, 1), (q.view(-1, 576), 64))]
''')
 change('qlayout = cute.tile_to_shape(atom, (64, 512), order=(0, 1))','qlayout = cute.tile_to_shape(atom, (64, 640), order=(0, 1))')
 change('klayout = cute.tile_to_shape(atom, (self.block_k, 512), order=(0, 1))','klayout = cute.tile_to_shape(atom, (self.block_k, 640), order=(0, 1))')
 change('''        qtail_layout = cute.tile_to_shape(tail_atom, (64, 64), order=(0, 1))
        ktail_layout = cute.tile_to_shape(tail_atom, (self.block_k, 64), order=(0, 1))
''','')
 change('qlayout, klayout, playout, vlayout, qtail_layout, ktail_layout).launch(','qlayout, klayout, playout, vlayout).launch(')
 change('''               vlayout: cute.ComposedLayout, qtail_layout: cute.ComposedLayout,
               ktail_layout: cute.ComposedLayout):''','''               vlayout: cute.ComposedLayout):''')
 change('''        sq_tail = alloc.allocate_tensor(cutlass.Float8E4M3FN, qtail_layout.outer,
                                        byte_alignment=128, swizzle=qtail_layout.inner)
''','')
 change('        ktbytes = cute.cosize(ktail_layout.outer)\n','')
 change('''        sk_tail_base = alloc.allocate_tensor(cutlass.Float8E4M3FN, cute.make_layout(2 * ktbytes),
                                             byte_alignment=128, swizzle=ktail_layout.inner)
''','')
 change('mbarrier_arrive_and_expect_tx(full + stage, kbytes + ktbytes)','mbarrier_arrive_and_expect_tx(full + stage, kbytes)')
 change('for col_block in cutlass.range(4, unroll_full=True):','for col_block in cutlass.range(5, unroll_full=True):',3 if version=='v296' else 4)
 change('''                            tail_raw = cute.recast_ptr(sk_tail_base.iterator) + cute.assume(stage * ktbytes + row * 64, divby=128)
                            gather4(tail_raw, tensor_map.iterator + 128, 0,
                                    i0, i1, i2, i3, full + stage)
''','')
 change('''                    sk_tail = cute.make_tensor(sk_tail_base.iterator + cute.assume(stage * ktbytes, divby=128), ktail_layout.outer)
''','')
 change('for k in cutlass.range(16, unroll_full=True):','for k in cutlass.range(18, unroll_full=True):')
 change('''                        for k in cutlass.range(2, unroll_full=True):
                            mma_ws(tp + 256,
                                cute.local_tile(sq_tail, (64, 32), (0, k)),
                                cute.local_tile(sk_tail, (128, 32), (0, k)),
                                128, False, True)
''','')
 qsites=2 if version=='v296' else 3
 change('mbarrier_arrive_and_expect_tx(qbar, 64 * 576)','mbarrier_arrive_and_expect_tx(qbar, 64 * 640)',qsites)
 change('load_q_tile(raw, tensor_map.iterator + 256,','load_q_tile(raw, tensor_map.iterator + 128,',qsites)
 lines=s.splitlines(keepends=True)
 removed=[(i,l) for i,l in enumerate(lines) if 'load_q_tile(cute.recast_ptr(sq_tail.iterator)' in l]
 assert len(removed)==qsites
 s=''.join(l for l in lines if 'load_q_tile(cute.recast_ptr(sq_tail.iterator)' not in l)
 assert 'sq_tail' not in s and 'sk_tail' not in s and 'ktbytes' not in s
 # Verify the exact set of edits by replaying the recorded transform.
 replay=old
 for a,b,count in changes:
  assert replay.count(a)==count
  replay=replay.replace(a,b)
 replay=''.join(l for l in replay.splitlines(keepends=True) if 'load_q_tile(cute.recast_ptr(sq_tail.iterator)' not in l)
 assert replay==s
 # Non-layout numerical helpers and runner are unchanged before version labels.
 before=ast.parse(old);after=ast.parse(s)
 unchanged=[]
 for node in before.body:
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name!='make_kv_map':
   other=next(n for n in after.body if isinstance(n,type(node)) and n.name==node.name)
   assert ast.dump(node)==ast.dump(other),node.name
   unchanged.append(node.name)
 s=f'"""Iteration {version[1:]}: unified SW128 Q/KV with padded640 shared rows; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 ast.parse(s)
 dest=root/f'kernel_{version}.py';assert not dest.exists();dest.write_text(s)
 proof[version]={'parent':parent,'sha256':hashlib.sha256(s.encode()).hexdigest(),
  'unchanged_top_level_functions':unchanged,'transform_checks_passed':True,
  'shared_q_bytes':64*640,'shared_k_stage_bytes':128*640,
  'extra_shared_bytes':(64+2*128)*(640-576),'qk_k32_tiles':18,'mathematical_qk_channels':576,
  'q_tma_requests_per_query':5,'gather4_requests_per_stage':4*8*5,
  'tensor_map_bytes':256,'candidate_launched':False}
dest=root/'artifacts/unified_sw128_compile';dest.mkdir(parents=True,exist_ok=True)
(dest/'source_controls.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
PY
