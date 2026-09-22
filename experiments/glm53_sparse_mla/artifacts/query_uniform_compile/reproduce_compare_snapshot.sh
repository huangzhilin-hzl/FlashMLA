#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json,re,shutil,hashlib
root=Path('artifacts/query_uniform_compile')
result={}
for parent,version,parent_dir in [('v272','v276','role_query_compile'),('v273','v277','strict_query_compile')]:
 p=root/version
 ref=Path('artifacts')/parent_dir/parent
 old=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',(ref/'sass.txt').read_text())
 new=re.findall(r'/\* (0x[0-9a-f]{16}) \*/',(p/'sass.txt').read_text())
 ptx=next(p.glob('*.ptx')).read_text()
 compute=192 if version=='v276' else 176
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+str(compute)+';',ptx)
 changes=[{'word_index':i,'parent':a,'candidate':b} for i,(a,b) in enumerate(zip(old,new)) if a!=b]
 result[version]={'parent':parent,'native_encoding_identical':old==new,'parent_encoding_words':len(old),'candidate_encoding_words':len(new),'different_aligned_words':len(changes),'first_different_words':changes[:12],'resources':(p/'resources.txt').read_text(),'source_sha256':hashlib.sha256(Path('kernel_'+version+'.py').read_bytes()).hexdigest(),'candidate_launched':False}
 shutil.copy2('kernel_'+version+'.py',p/('kernel_'+version+'.py'))
print(json.dumps(result,indent=2))
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
dest=Path('artifacts/snapshots/query_uniform_compile')
assert not dest.exists()
n=0
for p in root.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.ptx','.json','.log','.txt'}:
  out=dest/p.relative_to(root)
  out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,out)
  n+=1
print(json.dumps({'snapshot_files':n}))
PY

