#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
all_data={}
for v in ('v252','v253'):
 p=Path('artifacts/dual_score_arbitration_compile')/v
 sass=(p/'sass.txt').read_text();ptx=list(p.glob('*.ptx'))[0].read_text()
 d={'static_native_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',sass))//2,'LDL':len(re.findall(r'\bLDL\b',sass)),'STL':len(re.findall(r'\bSTL\b',sass)),'ELECT':len(re.findall(r'\bELECT\b',sass)),'VOTE':len(re.findall(r'\bVOTE\b',sass)),'BRA_U_ANY':sass.count('BRA.U.ANY'),'MMA':sass.count('UTCQMMA.WS'),'commit_sites':ptx.count('tcgen05.commit'),'elections':[x.strip() for x in ptx.splitlines() if 'elect.sync' in x]}
 all_data[v]=d
assert all_data['v253']['LDL']==all_data['v253']['STL']==0
assert all_data['v253']['MMA']==62
assert all_data['v253']['BRA_U_ANY']<=all_data['v252']['BRA_U_ANY']
Path('artifacts/v253_compile_control.json').write_text(json.dumps(all_data,indent=2)+'\n')
print(json.dumps(all_data,indent=2))
PY
