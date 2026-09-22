set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import os,time
for p in sorted(Path('artifacts').glob('v289_*')):
 if p.is_file() and p.suffix=='.log':
  print(p.name,p.stat().st_size,'mtime',time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime(p.stat().st_mtime)))
  print('\n'.join(p.read_text(errors='replace').splitlines()[-12:]))
PY
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
ps -eo pid,ppid,etime,args | grep -E 'v289|compute-sanitizer' | grep -v grep || true
