#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json,hashlib
root=Path('artifacts/query_reuse_compile')
dest=Path('artifacts/snapshots/query_reuse_compile')
assert not dest.exists()
for v in ['v267','v268']:
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

