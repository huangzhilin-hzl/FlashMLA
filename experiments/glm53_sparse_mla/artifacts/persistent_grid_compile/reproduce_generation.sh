#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for version,ctas in [('v274',296),('v275',592)]:
 s=(root/'kernel_v272.py').read_text()
 s=f'"""Iteration {version[1:]}: persistent grid with {ctas} CTAs; based on v272.\n'+s[s.index('\n')+1:]
 s=s.replace('"v272 requires',f'"{version} requires')
 assert s.count('min(q.shape[0], 148)')==4
 s=s.replace('min(q.shape[0], 148)',f'min(q.shape[0], {ctas})')
 ast.parse(s)
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
 for batch in [1,2,3,148,149,296,297,512,513,592,593,8192]:
  grid=min(batch,ctas)
  assert sorted(q for cta in range(grid) for q in range(cta,batch,grid))==list(range(batch))
print('query partition covers small, odd and full batches exactly once')
PY

