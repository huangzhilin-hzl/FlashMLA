#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/strict_persistent_query_source')
assert not dest.exists()
dest.mkdir(parents=True)
n=0
for p in Path('artifacts').glob('v273_source*'):
 if p.is_file() and p.suffix in {'.csv','.json','.txt','.log'}:
  shutil.copy2(p,dest/p.name)
  n+=1
print(json.dumps({'files':n}))
PY

