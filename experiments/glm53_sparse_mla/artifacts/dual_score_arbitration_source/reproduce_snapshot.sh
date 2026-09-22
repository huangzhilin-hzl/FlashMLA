#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,io,json,shutil
lines=Path('artifacts/v250_source.csv').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[1:]))))
picked=[]
for i,r in enumerate(rows):
 if 'UTCMMA' in r.get('Source',''):
  picked.append({k:r.get(k) for k in ('Address','Source','Instructions Executed','Thread Instructions Executed')})
print(json.dumps(picked,indent=2))
Path('artifacts/v250_qk_branch_counts_raw.json').write_text(json.dumps(picked,indent=2)+'\n')
snap=Path('artifacts/snapshots/dual_score_arbitration_source');assert not snap.exists();snap.mkdir(parents=True)
for name in ('v250_source.csv','v250_source_summary.json','v250_source.log','v250_source_preflight.log','v250_qk_branch_counts_raw.json'):
 shutil.copy2(Path('artifacts')/name,snap/name)
# Remove only non-text profiler report from the new version snapshot.
p=Path('artifacts/snapshots/v250/v250_ncu.ncu-rep')
if p.exists():p.unlink()
print('source snapshot files',sum(p.is_file() for p in snap.iterdir()))
PY
