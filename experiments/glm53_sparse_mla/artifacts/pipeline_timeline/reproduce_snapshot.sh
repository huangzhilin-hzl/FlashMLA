#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/pipeline_timeline')
assert not dest.exists()
counts={}
for directory in ['pipeline_timeline_compile','pipeline_timeline_guarded','pipeline_timeline_full']:
 n=0
 for p in (Path('artifacts')/directory).rglob('*'):
  if p.is_file() and p.suffix in {'.py','.json','.log','.txt','.ptx'}:
   out=dest/directory/p.relative_to(Path('artifacts')/directory)
   out.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(p,out)
   n+=1
 counts[directory]=n
print(json.dumps(counts))
PY
