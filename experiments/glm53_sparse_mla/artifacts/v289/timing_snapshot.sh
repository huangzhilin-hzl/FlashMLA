set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
root=Path('artifacts');dest=root/'snapshots/v289_timing'
assert not dest.exists();dest.mkdir(parents=True)
n=0
for pattern in ['v289_full_event.*','v289_ncu*']:
 for p in root.glob(pattern):
  if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
   shutil.copy2(p,dest/p.name);n+=1
shutil.copy2(root/'v289_gpu_preflight.log',dest/'v289_timing_preflight.log')
print('additive timing files',n+1)
PY
