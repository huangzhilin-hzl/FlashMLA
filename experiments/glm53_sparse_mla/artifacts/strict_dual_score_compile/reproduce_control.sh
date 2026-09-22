#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
p=Path('artifacts/strict_dual_score_compile/v257');s=(p/'sass.txt').read_text();t=list(p.glob('*.ptx'))[0].read_text()
d={'static_native_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,'LDL':len(re.findall(r'\bLDL\b',s)),'STL':len(re.findall(r'\bSTL\b',s)),'MMA':s.count('UTCQMMA.WS'),'commit_sites':t.count('tcgen05.commit'),'elections':[x.strip() for x in t.splitlines() if 'elect.sync' in x],'register_budgets':[x.strip() for x in t.splitlines() if 'setmaxnreg' in x],'collector_fill_sites':len(re.findall(r'collector::b[0-3]::fill',t)),'collector_lastuse_sites':len(re.findall(r'collector::b[0-3]::lastuse',t))}
assert d['LDL']==d['STL']==0
assert d['MMA']==70 and d['commit_sites']==5
assert d['collector_fill_sites']==d['collector_lastuse_sites']==8
(p/'resource_summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
PY
