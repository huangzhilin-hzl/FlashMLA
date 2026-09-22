#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/strict_query_extended')
assert not dest.exists()
counts={}
for directory in ['strict_query_validation','strict_query_round_robin']:
 n=0
 for p in (Path('artifacts')/directory).rglob('*'):
  if p.is_file() and p.suffix in {'.py','.json','.log','.txt','.csv'}:
   out=dest/directory/p.relative_to(Path('artifacts')/directory)
   out.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(p,out)
   n+=1
 counts[directory]=n
print(json.dumps(counts))
for p in Path('artifacts/strict_query_validation').glob('*event100.json'):
 r=json.loads(p.read_text())
 print(p.name,[(x['case'],x['cache'],x['median_us']) for x in r['benchmarks']])
PY

