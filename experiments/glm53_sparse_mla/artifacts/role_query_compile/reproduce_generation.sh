#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version in [('v269','v271'),('v270','v272')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: independent role query loops; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 start=s.index('        # Retain TMEM and barrier allocations; phase counts continue across queries.')
 loop_start=s.index('        for qi in cutlass.range(',start)
 body_start=s.index('\n',loop_start)+1
 for_line=s[loop_start:body_start].strip()
 end=s.index('        if warp == 0:\n            final_tp',body_start)
 body=s[body_start:end]
 tail=body.index('            # Every role drains before the next query overwrites Q/shared state.')
 body=body[:tail]
 qload_start=body.index('            if warp == 0:')
 producer_start=body.index('            if (warp >= 8) & (warp < 12):')
 issuer_start=body.index('            elif warp == 12:')
 compute_start=body.index('            elif warp < 8:')
 setup=body[:qload_start]
 setup='\n'.join(line[4:] if line.startswith('    ') else line for line in setup.rstrip().splitlines())+'\n'
 qload=body[qload_start:producer_start]
 qload='\n'.join('    '+line if line else '' for line in qload.rstrip().splitlines())+'\n'
 parts=[body[producer_start:issuer_start],body[issuer_start:compute_start],body[compute_start:]]
 generated=setup+'''        # Roles advance independently; existing tile and Q barriers carry phases.
        # The compute-only end barrier protects per-query reduction/epilogue reuse.
'''
 for index,part in enumerate(parts):
  header,content=part.split('\n',1)
  header=header[4:]
  generated+=header+'\n'
  if index==2:
   old='''                # Equal-count repeat is legal; CTA rendezvous separates queries.
                cute.arch.setmaxregister_increase(192)
'''
   assert content.count(old)==1
   content=content.replace(old,'')
   generated+='            cute.arch.setmaxregister_increase(192)\n'
  generated+='''            tile_base = cutlass.Int32(0)
            query_phase = cutlass.Int32(0)
            '''+for_line+'\n'
  if index==2:
   generated+=qload
  generated+=content.rstrip()+'\n'
  generated+='''                tile_base += cute.ceil_div(lens[qi], self.block_k)
                query_phase ^= 1
'''
 s=s[:start]+generated+s[end:]
 ast.parse(s)
 assert s.count('cute.arch.setmaxregister_increase(192)')==1
 assert s.count('cute.arch.barrier()')==1
 assert s.count('for qi in cutlass.range(')==3
 assert s.count('cute.arch.dealloc_tmem(')==1
 assert s.count('cute.arch.alloc_tmem(')==1
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY

