#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re
for v in ['v265','v266']:
 p=Path('artifacts/eight_producer_timeline_compile')/v
 lines=(p/'sass.txt').read_text().splitlines()
 print(v)
 for i,line in enumerate(lines):
  if re.search(r'\b(?:LDL|STL)\b',line):
   print('\n'.join(lines[max(0,i-8):i+9]))
 print((p/'compile.log').read_text()[-2800:])
PY

