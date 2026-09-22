#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re,shutil
base=Path('artifacts/single_pv_completion_compile')
expected={'v246':('ef6b3e6b989d7ee7b1ec66d39ac07ff6f34369b850ba2075b3d5aa90f2ea3b5a',111),'v247':('a5f28f2331d57b974fb60835c956b9ead26a411f4b47eadc7663ac68cb580872',128)}
summary={}
for v,(sha,regs) in expected.items():
 p=base/v;s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text();r=(p/'resources.txt').read_text()
 assert hashlib.sha256(Path('kernel_'+v+'.py').read_bytes()).hexdigest()==sha
 assert f'REG:{regs} STACK:0' in r
 assert not re.search(r'\b(?:LDL|STL)\b',s)
 commits=[x.strip() for x in ptx.splitlines() if 'tcgen05.commit.' in x]
 assert len(commits)==2,commits
 summary[v]=dict(kernel_sha256=sha,registers=regs,stack_bytes=0,static_instructions=len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,local_load_sites=0,local_store_sites=0,commits=commits)
 elections=[x.strip() for x in ptx.splitlines() if 'elect.sync' in x]
 assert len(elections)==4 and all(x.endswith('-1;') for x in elections),elections
 summary[v]['elections']=elections
 print(v,summary[v])
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
snap=Path('artifacts/snapshots/single_pv_completion_compile');assert not snap.exists()
for p in base.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.ptx','.txt','.log'):
  dest=snap/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2('artifacts/single_pv_completion_compile_preflight.log',snap/'preflight.log')
PY
