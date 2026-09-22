#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,re,shutil
base=Path('artifacts/pv_completion_release_compile')
expected={'v242':('48af11d0f1d5f301942c54c4cd7da00b5e6c6bd10282eec8332f69091a309325',123),'v243':('010e20171ac7fec56ea033a745b8a20bbde3731276bf4154639f706c07746796',128)}
summary={}
for v,(sha,regs) in expected.items():
 p=base/v;s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text();r=(p/'resources.txt').read_text()
 assert hashlib.sha256(Path('kernel_'+v+'.py').read_bytes()).hexdigest()==sha
 assert f'REG:{regs} STACK:0' in r
 assert not re.search(r'\b(?:LDL|STL)\b',s)
 commits=[x.strip() for x in ptx.splitlines() if 'tcgen05.commit.' in x]
 assert len(commits)==3,commits
 summary[v]=dict(kernel_sha256=sha,registers=regs,stack_bytes=0,static_instructions=len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,local_load_sites=0,local_store_sites=0,commits=commits)
 print(v,summary[v])
(base/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
snap=Path('artifacts/snapshots/pv_completion_release_compile');assert not snap.exists()
for p in base.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.ptx','.txt','.log'):
  dest=snap/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2('artifacts/pv_completion_release_compile_preflight.log',snap/'preflight.log')
PY
