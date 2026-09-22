#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
source=Path('artifacts/index_cache_compile')
dest=Path('artifacts/snapshots/index_cache_compile')
assert not dest.exists()
n=0
for p in source.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.ptx','.log','.txt','.json'}:
  out=dest/p.relative_to(source)
  out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,out)
  n+=1
for v in ['v263','v264']:
 shutil.copy2('kernel_'+v+'.py',dest/v/('kernel_'+v+'.py'))
 n+=1
print('snapshot_files',n)
PY
