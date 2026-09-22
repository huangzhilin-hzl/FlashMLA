#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json,hashlib,re
root=Path('artifacts/persistent_grid_compile')
dest=Path('artifacts/snapshots/persistent_grid_compile')
assert not dest.exists()
for v in ['v274','v275']:
 p=root/v
 ptx=next(p.glob('*.ptx')).read_text()
 sass=(p/'sass.txt').read_text()
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+192;',ptx)
 assert 'REG:128 STACK:8' in (p/'resources.txt').read_text()
 assert sass.count('USETMAXREG')==2
 assert ptx.count('tcgen05.mma.ws')==26 and ptx.count('tcgen05.commit.')==2
 shutil.copy2('kernel_'+v+'.py',root/v/('kernel_'+v+'.py'))
n=0
for p in root.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.ptx','.json','.log','.txt'}:
  out=dest/p.relative_to(root)
  out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,out)
  n+=1
print(json.dumps({'files':n}))
PY

