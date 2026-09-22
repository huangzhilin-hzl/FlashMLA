#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,csv,io,json
snap=Path('artifacts/snapshots/kv_prefetch_compile');assert not snap.exists();(snap/'v254').mkdir(parents=True)
for p in Path('artifacts/kv_prefetch_compile/v254').iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v254'/p.name)
shutil.copy2('kernel_v254.py',snap/'v254'/'kernel_v254.py')
shutil.copy2('artifacts/v254_compile_preflight.log',snap/'v254_compile_preflight.log')
snap=Path('artifacts/snapshots/kv_prefetch_source');assert not snap.exists();snap.mkdir(parents=True)
for n in ('v254_source.csv','v254_source_summary.json','v254_source.log','v254_source_preflight.log','v254_cache_comparison.json'):shutil.copy2(Path('artifacts')/n,snap/n)
lines=Path('artifacts/v254_source.csv').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[1:]))))
pfs=[{k:r.get(k) for k in ('Address','Source','Instructions Executed','Thread Instructions Executed','Predicated-On Thread Instructions Executed')} for r in rows if 'CCTL.E.PF2' in r.get('Source','')]
assert len(pfs)==5
(snap/'prefetch_execution.json').write_text(json.dumps(pfs,indent=2)+'\n')
snap=Path('artifacts/snapshots/kv_prefetch_cache');assert not snap.exists();snap.mkdir(parents=True)
for p in Path('artifacts/kv_prefetch_cache').iterdir():
 if p.suffix in ('.csv','.json','.log'):shutil.copy2(p,snap/p.name)
p=Path('artifacts/snapshots/v254/v254_ncu.ncu-rep')
if p.exists():p.unlink()
print('prefetch executions',pfs)
PY
