#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,io,json,shutil
lines=Path('artifacts/v250_source.csv').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(lines[1:]))))
r=[x for x in rows if 'UTCQMMA.WS' in x.get('Source','')]
assert len(r)==62
def n(x): return int(x['Instructions Executed'].replace(',',''))
groups={}
for name,group in [('bootstrap',r[:18]),('early',r[18:36]),('pv',r[36:44]),('fallback',r[44:])]:
 counts=[n(x) for x in group];assert len(set(counts))==1
 groups[name]={'count':counts[0],'static_mma_sites':len(group),'first_address':group[0]['Address'],'last_address':group[-1]['Address']}
assert groups['bootstrap']['count']==8192
assert groups['early']['count']+groups['fallback']['count']==8192*15
assert groups['pv']['count']==8192*16
d={'groups':groups,'early_fraction_of_nonbootstrap_qk':groups['early']['count']/(8192*15),'note':'Compiler source-order blocks verified against native instruction groups; source-profile execution regime only.'}
p=Path('artifacts/v250_qk_branch_counts_raw.json');p.write_text(json.dumps({'rows':r,'summary':d},indent=2)+'\n')
shutil.copy2(p,Path('artifacts/snapshots/dual_score_arbitration_source')/p.name)
print(json.dumps(d,indent=2))
PY
