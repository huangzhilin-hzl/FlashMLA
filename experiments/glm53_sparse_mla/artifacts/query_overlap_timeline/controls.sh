set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
root=Path('artifacts/query_overlap_timeline/compile')
summary=json.loads((root/'summary.json').read_text())
checks={}
for v in ['v278','v284','v281','v285']:
 p=root/v;ptx=next(p.glob('*.ptx')).read_text();sass=(p/'sass.txt').read_text()
 fast=v in ['v278','v284']
 assert 'REG:128' in summary[v]['resources']
 assert ('STACK:8' if fast else 'STACK:0') in summary[v]['resources']
 assert ptx.count('tcgen05.mma.ws')==(26 if fast else 34)
 assert ptx.count('tcgen05.commit.')==2
 assert ptx.count('%globaltimer')==2
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+('192' if fast else '176')+';',ptx)
 lines=sass.splitlines()
 snippets=['\n'.join(lines[max(0,i-8):i+13]) for i,line in enumerate(lines) if 'GLOBALTIMER' in line]
 (p/'timestamp_sites.txt').write_text('\n\n'.join(snippets)+'\n')
 checks[v]={'same_register_stack_and_local_site_counts_as_parent':True,'timestamp_ptx_sites':2,'native_timer_sites':len(snippets)}
(root/'control_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
PY
