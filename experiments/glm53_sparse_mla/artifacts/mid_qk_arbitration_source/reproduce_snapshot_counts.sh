#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,io,json,shutil
lines=Path('artifacts/v253_source.csv').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[1:]))))
r=[x for x in rows if 'UTCQMMA.WS' in x.get('Source','')]
assert len(r)==62
groups={}
for name,group in [('bootstrap',r[:18]),('early_prefix',r[18:26]),('early_tail',r[26:36]),('pv',r[36:44]),('fallback_prefix',r[44:52]),('fallback_tail',r[52:])]:
 counts={}
 for key in ('Instructions Executed','Thread Instructions Executed','Predicated-On Thread Instructions Executed'):
  v=[int(x[key].replace(',','')) for x in group];assert len(set(v))==1
  counts[key]=v[0]
 groups[name]={'counts':counts,'static_mma_sites':len(group),'first_address':group[0]['Address'],'last_address':group[-1]['Address']}
def actual(name):return groups[name]['counts']['Predicated-On Thread Instructions Executed']
assert actual('bootstrap')==8192 and actual('pv')==131072
assert actual('early_prefix')+actual('fallback_prefix')==122880
assert actual('early_tail')+actual('fallback_tail')==122880
partial=actual('early_prefix')-actual('early_tail')
d={'groups':groups,'paths':{'full_early':actual('early_tail'),'partial_early':partial,'full_fallback':actual('fallback_prefix')},'note':'Use predicated-on thread counts: ten early-tail native MMAs are predicated, so warp issue counts overstate actual early completion.'}
p=Path('artifacts/v253_qk_branch_counts.json');p.write_text(json.dumps({'summary':d,'rows':r},indent=2)+'\n')
print(json.dumps(d,indent=2))
snap=Path('artifacts/snapshots/mid_qk_arbitration_source');assert not snap.exists();snap.mkdir(parents=True)
for name in ('v253_source.csv','v253_source_summary.json','v253_source.log','v253_source_preflight.log','v253_qk_branch_counts.json'):
 shutil.copy2(Path('artifacts')/name,snap/name)
c=Path('artifacts/dual_score_arbitration_compile/v253')
snap=Path('artifacts/snapshots/mid_qk_arbitration_compile');assert not snap.exists();(snap/'v253').mkdir(parents=True)
for p in c.iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v253'/p.name)
shutil.copy2('kernel_v253.py',snap/'v253'/'kernel_v253.py')
shutil.copy2('artifacts/v253_compile_preflight.log',snap/'v253_compile_preflight.log')
p=Path('artifacts/snapshots/v253/v253_ncu.ncu-rep')
if p.exists():p.unlink()
PY
