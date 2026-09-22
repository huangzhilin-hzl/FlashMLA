#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json,shutil,hashlib
p=Path('artifacts/strict_query_compile/v273')
ptx=next(p.glob('*.ptx')).read_text()
sass=(p/'sass.txt').read_text()
assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+176;',ptx)
assert 'REG:128 STACK:8' in (p/'resources.txt').read_text()
assert sass.count('USETMAXREG')==2
assert ptx.count('tcgen05.mma.ws')==34
assert ptx.count('tcgen05.commit.')==2
lines=sass.splitlines()
sites=[{'instruction':line.strip(),'context':[x.strip() for x in lines[max(0,i-6):i+5] if not re.match(r'^\s*/\* 0x',x)]} for i,line in enumerate(lines) if re.search(r'\b(?:LDL|STL)\b',line)]
(p/'local_memory_sites.json').write_text(json.dumps(sites,indent=2)+'\n')
print(json.dumps(sites[:2],indent=2))
r={'threads':512,'initial_registers':128,'stack_bytes':8,'donor_registers':64,'compute_registers':176,'initial_pool':65536,'final_pool':61440,'source_sha256':hashlib.sha256(Path('kernel_v273.py').read_bytes()).hexdigest(),'candidate_launched':False}
Path('artifacts/strict_query_compile/control.json').write_text(json.dumps(r,indent=2)+'\n')
shutil.copy2('kernel_v273.py',p/'kernel_v273.py')
dest=Path('artifacts/snapshots/strict_query_compile')
assert not dest.exists()
n=0
for f in Path('artifacts/strict_query_compile').rglob('*'):
 if f.is_file() and f.suffix in {'.py','.json','.ptx','.log','.txt'}:
  out=dest/f.relative_to('artifacts/strict_query_compile')
  out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(f,out)
  n+=1
print(json.dumps({'control':r,'snapshot_files':n}))
PY

