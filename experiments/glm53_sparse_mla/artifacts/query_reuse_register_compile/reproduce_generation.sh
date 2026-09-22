#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version in [('v267','v269'),('v268','v270')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: query reuse with donor64/compute192; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 assert s.count('block=(416, 1, 1), stream=stream')==1
 s=s.replace('block=(416, 1, 1), stream=stream','block=(512, 1, 1), min_blocks_per_mp=1, stream=stream')
 anchor='''        # Retain TMEM and barrier allocations; phase counts continue across queries.'''
 assert s.count(anchor)==1
 s=s.replace(anchor,'''        # Complete warpgroups: two compute, one producer, one issuer/idle.
        # Initial128*512 = final256*64 + 256*192 = 65536 registers.
        if warp >= 8:
            cute.arch.setmaxregister_decrease(64)
'''+anchor)
 anchor='''            elif warp < 8:
                cute.arch.mbarrier_wait(qbar, query_phase)'''
 assert s.count(anchor)==1
 s=s.replace(anchor,'''            elif warp < 8:
                # Equal-count repeat is legal; CTA rendezvous separates queries.
                cute.arch.setmaxregister_increase(192)
                cute.arch.mbarrier_wait(qbar, query_phase)''')
 ast.parse(s)
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY

