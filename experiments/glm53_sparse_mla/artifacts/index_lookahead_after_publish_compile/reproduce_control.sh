#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
p=Path('artifacts/index_lookahead_compile/v256');s=(p/'sass.txt').read_text();ptx=list(p.glob('*.ptx'))[0].read_text()
native=[x.strip().split('/* 0x')[0].strip() for x in s.splitlines() if re.search(r'/\*[0-9a-f]+\*/',x)]
d={'static_native_instructions':len(native),'LDL':len(re.findall(r'\bLDL\b',s)),'STL':len(re.findall(r'\bSTL\b',s)),'prefetch_sites':ptx.count('prefetch.global'),'load_and_barrier_sites':[x for x in native if 'LDG.E' in x or 'BAR.SYNC' in x]}
assert d['LDL']==d['STL']==d['prefetch_sites']==0
(p/'resource_summary.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
# Include producer index loads, publication barriers and the first gather.
start=next(i for i,x in enumerate(native) if 'LDG.E' in x and i>30)
selected=[]
for i,x in enumerate(native):
 if 'LDG.E' in x:
  selected.append({'load':x,'context':native[max(0,i-4):i+7]})
(p/'load_context.json').write_text(json.dumps(selected,indent=2)+'\n')
print('load contexts',json.dumps(selected[:5],indent=2))
PY
