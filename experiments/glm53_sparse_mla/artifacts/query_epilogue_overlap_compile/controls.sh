set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json,hashlib,shutil
root=Path('artifacts/query_epilogue_overlap_compile')
out={}
for v in ['v284','v285']:
 p=root/v;s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text()
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+('192' if v=='v284' else '176')+';',ptx)
 lines=s.splitlines()
 sites=['\n'.join(lines[max(0,i-4):i+5]) for i,l in enumerate(lines) if re.search(r'\b(?:LDL|STL)\b',l)]
 (p/'local_sites.txt').write_text('\n\n'.join(sites)+'\n')
 shutil.copy2(f'kernel_{v}.py',p/f'kernel_{v}.py')
 out[v]={'source_sha256':hashlib.sha256(Path(f'kernel_{v}.py').read_bytes()).hexdigest(),'local_sites':len(sites),'guarded_runtime_next':True}
(root/'control_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
PY
