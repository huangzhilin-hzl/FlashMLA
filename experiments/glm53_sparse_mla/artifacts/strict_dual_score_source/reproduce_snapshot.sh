#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,io,json,shutil
lines=Path('artifacts/v257_source.csv').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[1:]))))
r=[x for x in rows if 'UTCQMMA.WS' in x.get('Source','')]
assert len(r)==70
groups={}
for name,group in [('bootstrap',r[:18]),('early',r[18:36]),('pv',r[36:52]),('fallback',r[52:])]:
 vals=[int(x['Predicated-On Thread Instructions Executed'].replace(',','')) for x in group]
 assert len(set(vals))==1
 assert all(int(x['Instructions Executed'].replace(',',''))==v for x,v in zip(group,vals))
 groups[name]={'count':vals[0],'static_sites':len(group)}
assert groups['bootstrap']['count']==8192 and groups['pv']['count']==131072
assert groups['early']['count']+groups['fallback']['count']==122880
d={'groups':groups,'early_fraction':groups['early']['count']/122880}
Path('artifacts/v257_qk_branch_counts.json').write_text(json.dumps({'summary':d,'rows':r},indent=2)+'\n')
snap=Path('artifacts/snapshots/strict_dual_score_source');assert not snap.exists();snap.mkdir(parents=True)
for n in ('v257_source.csv','v257_source_summary.json','v257_source.log','v257_source_preflight.log','v257_qk_branch_counts.json'):shutil.copy2(Path('artifacts')/n,snap/n)
snap=Path('artifacts/snapshots/strict_dual_score_compile');assert not snap.exists();(snap/'v257').mkdir(parents=True)
for p in Path('artifacts/strict_dual_score_compile/v257').iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v257'/p.name)
shutil.copy2('kernel_v257.py',snap/'v257'/'kernel_v257.py')
shutil.copy2('artifacts/v257_compile_preflight.log',snap/'v257_compile_preflight.log')
p=Path('artifacts/snapshots/v257/v257_ncu.ncu-rep')
if p.exists():p.unlink()
print(json.dumps(d,indent=2))
PY
