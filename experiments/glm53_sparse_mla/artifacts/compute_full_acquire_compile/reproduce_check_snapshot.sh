#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re,shutil
base=Path('artifacts/compute_full_acquire_compile')
expected={'v248':('ae18a0da015f7fea921b3448b5e7ff06757034b5c0a0823651dcfd9b43fa94d7',115),'v249':('32010a1b3fb22ea1451c1588abc1a65feaa773e8541a8f3b3a1ec5ef94261081',128)}
summary={}
for v,(sha,regs) in expected.items():
 p=base/v;s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text();r=(p/'resources.txt').read_text()
 assert hashlib.sha256(Path('kernel_'+v+'.py').read_bytes()).hexdigest()==sha
 assert f'REG:{regs} STACK:0' in r
 assert not re.search(r'\b(?:LDL|STL)\b',s)
 commits=[x.strip() for x in ptx.splitlines() if 'tcgen05.commit.' in x]
 assert len(commits)==2,commits
 summary[v]=dict(kernel_sha256=sha,registers=regs,stack_bytes=0,static_instructions=len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,local_load_sites=0,local_store_sites=0,commits=commits)
 assert ptx.count('mbarrier.try_wait')==7
 summary[v]['wait_sites']=ptx.count('mbarrier.try_wait')
 elections=[x.strip() for x in ptx.splitlines() if 'elect.sync' in x]
 assert len(elections)==4 and all(x.endswith('-1;') for x in elections),elections
 summary[v]['elections']=elections
 print(v,summary[v])
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
snap=Path('artifacts/snapshots/compute_full_acquire_compile');assert not snap.exists()
for p in base.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.ptx','.txt','.log'):
  dest=snap/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2('artifacts/compute_full_acquire_compile_preflight.log',snap/'preflight.log')
PY
