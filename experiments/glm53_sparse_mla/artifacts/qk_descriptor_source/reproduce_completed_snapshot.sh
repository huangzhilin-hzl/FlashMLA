#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/qk_descriptor_completed')
assert not dest.exists()
counts={}
for version in ['v260','v262']:
 (dest/version).mkdir(parents=True,exist_ok=True)
 for p in Path('artifacts').glob(version+'_*'):
  if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'} and '_source' not in p.name:
   shutil.copy2(p,dest/version/p.name)
   counts[version]=counts.get(version,0)+1
(dest/'qk_descriptor_source').mkdir(parents=True,exist_ok=True)
for p in Path('artifacts').glob('v262_source*'):
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  shutil.copy2(p,dest/'qk_descriptor_source'/p.name)
  counts['source']=counts.get('source',0)+1
print(json.dumps(counts))
PY
