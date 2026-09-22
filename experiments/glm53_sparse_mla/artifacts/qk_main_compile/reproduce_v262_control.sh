#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
p=Path('artifacts/qk_main_compile/v262')
s=(p/'sass.txt').read_text()
ptx=next(p.glob('*.ptx')).read_text()
keys=['UTCQMMA.WS','MOV.SPILL','R2UR.FILL','R2UR','UIADD3','ULOP3','LDL','STL','BRA.U.ANY']
r={'instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,
 'resources':(p/'resources.txt').read_text(),'counts':{k:s.count(k) for k in keys},
 'ptx_mma':ptx.count('tcgen05.mma.ws'),'ptx_commits':ptx.count('tcgen05.commit.')}
assert r['ptx_mma']==34 and r['ptx_commits']==2
assert not r['counts']['LDL'] and not r['counts']['STL']
(p/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
PY
