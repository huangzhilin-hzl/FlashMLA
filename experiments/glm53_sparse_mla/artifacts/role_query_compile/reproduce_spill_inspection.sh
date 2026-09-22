#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
p=Path('artifacts/role_query_compile/v271')
lines=(p/'sass.txt').read_text().splitlines()
sites=[]
for i,line in enumerate(lines):
 if re.search(r'\b(?:LDL|STL)\b',line):
  sites.append({'instruction':line.strip(),'context':[x.strip() for x in lines[max(0,i-6):i+5] if not re.match(r'^\s*/\* 0x',x)]})
(p/'local_memory_sites.json').write_text(json.dumps(sites,indent=2)+'\n')
print(json.dumps(sites[:6],indent=2))
print('Remaining local sites')
for x in sites[6:]:print(x['instruction'].split(';')[0])
PY

