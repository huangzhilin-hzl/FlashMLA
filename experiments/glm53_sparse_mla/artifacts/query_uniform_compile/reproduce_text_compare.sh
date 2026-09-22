#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
result={}
for a,b,d in [('v272','v276','role_query_compile'),('v273','v277','strict_query_compile')]:
 old=(Path('artifacts')/d/a/'sass.txt').read_text()
 new=(Path('artifacts/query_uniform_compile')/b/'sass.txt').read_text()
 pattern=r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;\s*/\*'
 oldlines=re.findall(pattern,old)
 newlines=re.findall(pattern,new)
 assert len(oldlines)==len(newlines)
 diffs=[{'pc':x[0],'parent':x[1],'candidate':y[1]} for x,y in zip(oldlines,newlines) if x!=y]
 result[b]={'text_differences':diffs,'note':'Raw instruction encodings differ at ten words; do not label binary-identical.'}
 print(b,json.dumps(diffs,indent=2))
Path('artifacts/query_uniform_compile/text_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
PY

