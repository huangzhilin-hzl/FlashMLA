#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
result={}
for v in ['v263','v264']:
 p=Path('artifacts/index_cache_compile')/v
 lines=(p/'sass.txt').read_text().splitlines()
 ptx=next(p.glob('*.ptx')).read_text()
 scalar=[(i,s) for i,s in enumerate(lines) if re.search(r'\bLDG\.',s)]
 chosen=[(i,s) for i,s in scalar if 'desc[UR30]' not in s]
 assert ptx.count('ld.global.L2::cache_hint.b32')==1
 assert len(chosen)==1,(v,chosen)
 i,line=chosen[0]
 reg=int(re.search(r'desc\[UR(\d+)\]',line).group(1))
 policy_pattern=rf'\b(?:UMOV|ULOP3)[^;]*\bUR(?:{reg}|{reg+1})\b'
 r={'hint_sites':1,'scalar_loads':[s.strip() for _,s in scalar],
    'hint_load_context':[s.strip() for s in lines[max(0,i-28):i+6]],
    'policy_setup':[s.strip() for s in lines[:i] if re.search(policy_pattern,s)]}
 result[v]=r
 print(v,'hinted load',line.strip())
 print('policy setup',r['policy_setup'][:8])
Path('artifacts/index_cache_compile/policy_encoding.json').write_text(json.dumps(result,indent=2)+'\n')
PY
