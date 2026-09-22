#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json,hashlib,shutil
base=Path('artifacts/producer_unroll_followup_compile')
summary={}
for v in ('v237','v238','v239'):
 p=base/v
 s=(p/'sass.txt').read_text()
 ptx=next(p.glob('*.ptx')).read_text()
 lines=s.splitlines()
 local=[x.strip() for x in lines if re.search(r'\b(?:LDL|STL)\b',x)]
 alloc=[x.strip() for x in lines if 'SETMAX' in x]
 instr=re.findall(r'/\* 0x[0-9a-f]{16} \*/',s)
 d=dict(kernel_sha256=hashlib.sha256(Path('kernel_'+v+'.py').read_bytes()).hexdigest(),resources=(p/'resources.txt').read_text().strip(),static_instructions=len(instr)//2,local_load_sites=len([x for x in local if 'LDL' in x]),local_store_sites=len([x for x in local if 'STL' in x]),setmax_sass=alloc,local_sass=local)
 summary[v]=d
 print(v,d['static_instructions'],d['local_load_sites'],d['local_store_sites'],alloc)
 if v=='v239':
  print('\n'.join(local[:12]+local[-12:]))
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
snap=Path('artifacts/snapshots/producer_unroll_followup_compile')
assert not snap.exists()
for src in base.rglob('*'):
 if src.is_file() and src.suffix in ('.json','.log','.ptx','.txt'):
  dest=snap/src.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for v in ('v237','v238','v239'):
 shutil.copy2(Path('artifacts')/(v+'_compile_preflight.log'),snap/(v+'_preflight.log'))
PY
