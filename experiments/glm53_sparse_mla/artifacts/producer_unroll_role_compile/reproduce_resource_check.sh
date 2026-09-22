#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re
p=Path('artifacts/producer_unroll_role_compile/v240')
s=(p/'sass.txt').read_text()
ptx=next(p.glob('*.ptx')).read_text()
r=(p/'resources.txt').read_text()
sha=hashlib.sha256(Path('kernel_v240.py').read_bytes()).hexdigest()
assert sha=='586f4a8b8aca56d9ace295d881d1065cbb0272194c16bfc0e776ce416543459c'
assert 'REG:128 STACK:0' in r
assert not re.search(r'\b(?:LDL|STL)\b',s)
for mode,count in [('dec',104),('dec',56),('inc',176)]:
 assert re.search(r'setmaxnreg\.'+mode+r'\.sync\.aligned\.u32\s+'+str(count)+r'\s*;',ptx)
d=dict(kernel_sha256=sha,registers=128,stack_bytes=0,static_instructions=len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,local_load_sites=0,local_store_sites=0,setmax_sass=[x.strip() for x in s.splitlines() if 'SETMAX' in x],ur_transfers=[x.strip() for x in s.splitlines() if 'MOV.SPILL' in x or 'R2UR.FILL' in x])
(p/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({k:v for k,v in d.items() if k!='ur_transfers'},indent=2))
print('UR/R transfer sites:',len(d['ur_transfers']))
PY
