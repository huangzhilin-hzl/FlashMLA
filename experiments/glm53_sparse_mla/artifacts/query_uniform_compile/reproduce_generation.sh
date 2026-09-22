#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version in [('v272','v276'),('v273','v277')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: uniform persistent query state; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 anchor='        cta_id, _, _ = cute.arch.block_idx()\n'
 assert s.count(anchor)==1
 s=s.replace(anchor,anchor+'        cta_id = cute.arch.make_warp_uniform(cta_id)\n')
 loop='            for qi in cutlass.range(cta_id, q.shape[0], min(q.shape[0], 148), unroll=1):\n'
 assert s.count(loop)==3
 s=s.replace(loop,loop+'                qi = cute.arch.make_warp_uniform(qi)\n')
 anchor='''                tile_base += cute.ceil_div(lens[qi], self.block_k)
                query_phase ^= 1'''
 assert s.count(anchor)==3
 s=s.replace(anchor,'''                tile_base = cute.arch.make_warp_uniform(tile_base + cute.ceil_div(lens[qi], self.block_k))
                query_phase = cute.arch.make_warp_uniform(query_phase ^ 1)''')
 ast.parse(s)
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY

