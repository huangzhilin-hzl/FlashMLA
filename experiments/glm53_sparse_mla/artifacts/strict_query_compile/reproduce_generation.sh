#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
s=(root/'kernel_v197.py').read_text()
s='"""Iteration 273: strict independent-role persistence; based on v197.\n'+s[s.index('\n')+1:]
s=s.replace('"v184 requires','"v273 requires')
s=s.replace('# Compile with CuTeDSL >=4.6.3 and inspect initial allocation before launch.','# Inspect initial allocation and role budgets on the selected compiler before launch.')
s=s.replace('One CTA owns one query and all 512 output channels. Both PV N tiles reuse','Each CTA processes assigned queries and all 512 output channels. Both PV N tiles reuse')
assert s.count('grid=(q.shape[0], 1, 1)')==1
s=s.replace('grid=(q.shape[0], 1, 1)','grid=(min(q.shape[0], 148), 1, 1)')
s=s.replace('qi, _, _ = cute.arch.block_idx()','cta_id, _, _ = cute.arch.block_idx()')
start=s.index('        store128 =')
end=s.index('\n\n\n\ndef make_runner',start)
body=s[start:end]
old='''            if warp == 0:
                cute.arch.dealloc_tmem(tp, 512)'''
assert body.count(old)==1
body=body.replace(old,'')
assert body.count('stage = block % 2')==3
body=body.replace('stage = block % 2','stage = (tile_base + block) % 2')
body=body.replace('if block >= 2:','if tile_base + block >= 2:')
body=body.replace('((block // 2) - 1) % 2','(((tile_base + block) // 2) - 1) % 2')
assert body.count('(block // 2) % 2')==2
body=body.replace('(block // 2) % 2','((tile_base + block) // 2) % 2')
for barrier in ['p_ready','qk_done','pv_done']:
 old=f'mbarrier_wait({barrier}, block % 2)'
 assert body.count(old)==1
 body=body.replace(old,f'mbarrier_wait({barrier}, (tile_base + block) % 2)')
assert body.count('mbarrier_wait(qbar, 0)')==2
body=body.replace('mbarrier_wait(qbar, 0)','mbarrier_wait(qbar, query_phase)')
qload_start=body.index('        if warp == 0:')
producer_start=body.index('        if (warp >= 8) & (warp < 12):')
issuer_start=body.index('        elif warp == 12:')
compute_start=body.index('        elif warp < 8:')
setup=body[:qload_start]
qload=body[qload_start:producer_start]
qload='\n'.join('        '+line if line else '' for line in qload.rstrip().splitlines())+'\n'
parts=[body[producer_start:issuer_start],body[issuer_start:compute_start],body[compute_start:]]
generated=setup+'''        # Independent role loops retain query and tile phase counters across rows.
        # Next Q publication follows the previous compute epilogue rendezvous.
'''
for index,part in enumerate(parts):
 header,content=part.split('\n',1)
 generated+=header+'\n'
 if index==2:
  old='            cute.arch.setmaxregister_increase(176)\n'
  assert content.count(old)==1
  content=content.replace(old,'')
  generated+=old
 generated+='''            tile_base = cutlass.Int32(0)
            query_phase = cutlass.Int32(0)
            for qi in cutlass.range(cta_id, q.shape[0], min(q.shape[0], 148), unroll=1):
'''
 if index==2:generated+=qload
 generated+='\n'.join('    '+line if line else '' for line in content.rstrip().splitlines())+'\n'
 generated+='''                tile_base += cute.ceil_div(lens[qi], self.block_k)
                query_phase ^= 1
'''
generated+='''        if warp == 0:
            final_tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                ptr_to_buffer_holding_addr=holding)
            cute.arch.dealloc_tmem(final_tp, 512)
'''
s=s[:start]+generated+s[end:]
ast.parse(s)
assert s.count('cute.arch.setmaxregister_increase(176)')==1
assert s.count('cute.arch.setmaxregister_decrease(64)')==1
assert s.count('cute.arch.barrier()')==1
assert s.count('for qi in cutlass.range(')==3
assert s.count('cute.arch.dealloc_tmem(')==1
p=root/'kernel_v273.py'
assert not p.exists()
p.write_text(s)
print('v273',hashlib.sha256(s.encode()).hexdigest())
PY

