#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
parent=(root/'kernel_v190.py').read_text()
for version,grid,why in [('v267','cute.ceil_div(q.shape[0], 2)','two queries per CTA'),('v268','min(q.shape[0], 148)','one persistent CTA per B300 SM')]:
 s=parent
 s=f'"""Iteration {version[1:]}: {why}; based on v190.\n'+s[s.index('\n')+1:]
 s=s.replace('"v190 requires',f'"{version} requires')
 assert s.count('grid=(q.shape[0], 1, 1)')==1
 s=s.replace('grid=(q.shape[0], 1, 1)',f'grid=({grid}, 1, 1)')
 assert s.count('qi, _, _ = cute.arch.block_idx()')==1
 s=s.replace('qi, _, _ = cute.arch.block_idx()','cta_id, _, _ = cute.arch.block_idx()')
 begin=s.index('        store128 =')
 end=s.index('\n\n\n\ndef make_runner',begin)
 work=s[begin:end]
 old='''            if warp == 0:
                cute.arch.dealloc_tmem(tp, 512)'''
 assert work.count(old)==1
 work=work.replace(old,'')
 assert work.count('stage = block % 2')==3
 work=work.replace('stage = block % 2','stage = (tile_base + block) % 2')
 assert work.count('if block >= 2:')==1
 work=work.replace('if block >= 2:','if tile_base + block >= 2:')
 work=work.replace('((block // 2) - 1) % 2','(((tile_base + block) // 2) - 1) % 2')
 assert work.count('(block // 2) % 2')==2
 work=work.replace('(block // 2) % 2','((tile_base + block) // 2) % 2')
 for barrier in ['p_ready','qk_done','pv_done']:
  old=f'mbarrier_wait({barrier}, block % 2)'
  assert work.count(old)==1
  work=work.replace(old,f'mbarrier_wait({barrier}, (tile_base + block) % 2)')
 assert work.count('mbarrier_wait(qbar, 0)')==2
 work=work.replace('mbarrier_wait(qbar, 0)','mbarrier_wait(qbar, query_phase)')
 work='\n'.join('    '+line if line else '' for line in work.rstrip().splitlines())
 s=s[:begin]+f'''        # Retain TMEM and barrier allocations; phase counts continue across queries.
        tile_base = cutlass.Int32(0)
        query_phase = cutlass.Int32(0)
        for qi in cutlass.range(cta_id, q.shape[0], {grid}, unroll=1):
'''+work+'''
            # Every role drains before the next query overwrites Q/shared state.
            tmem_before_sync()
            cute.arch.barrier()
            tmem_after_sync()
            tile_base += cute.ceil_div(lens[qi], self.block_k)
            query_phase ^= 1
        if warp == 0:
            final_tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                ptr_to_buffer_holding_addr=holding)
            cute.arch.dealloc_tmem(final_tp, 512)
'''+s[end:]
 s=s.replace('One CTA owns one query and all 512 output channels. Both PV N tiles reuse','Each CTA processes its assigned queries and all 512 output channels. Both PV N tiles reuse')
 ast.parse(s)
 assert s.count('cute.arch.dealloc_tmem(')==1
 assert s.count('cute.arch.alloc_tmem(')==1
 assert s.count('cute.arch.mbarrier_init(qk_done, 1)')==1
 path=root/f'kernel_{version}.py'
 assert not path.exists()
 path.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
# Query partition for odd, small and full batches; phase schedule spans query edges.
for batch in [1,2,3,147,148,149,512,8192]:
 for grid in [(batch+1)//2,min(batch,148)]:
  assert sorted(q for cta in range(grid) for q in range(cta,batch,grid))==list(range(batch))
for lengths in [[16,16],[1,3,2,16],[0,1,0,7,16]]:
 base=0
 for count in lengths:
  for block in range(count):
   absolute=base+block
   assert (absolute%2,absolute//2%2)==(absolute%2,(absolute//2)&1)
  base+=count
print('static query coverage and phase accounting PASS; runtime remains unqualified')
PY

