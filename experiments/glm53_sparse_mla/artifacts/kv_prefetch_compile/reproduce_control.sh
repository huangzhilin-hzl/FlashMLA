#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
p=Path('artifacts/kv_prefetch_compile/v254');s=(p/'sass.txt').read_text();t=list(p.glob('*.ptx'))[0].read_text()
d={'static_native_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,'LDL':len(re.findall(r'\bLDL\b',s)),'STL':len(re.findall(r'\bSTL\b',s)),'PTX_prefetch_sites':t.count('prefetch.global.L2'),'prefetch_native':[x.strip() for x in s.splitlines() if 'CCTL' in x or 'PF2' in x]}
assert d['LDL']==d['STL']==0
assert d['PTX_prefetch_sites']==5 and len(d['prefetch_native'])==5
(p/'resource_summary.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
PY
