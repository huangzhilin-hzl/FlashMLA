set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
root=Path('artifacts/snapshots/v289_qualification_before_busy')
assert not root.exists();root.mkdir(parents=True)
n=0
for p in Path('artifacts').glob('v289_*'):
 if p.is_file() and p.suffix in {'.json','.log','.txt'}:
  shutil.copy2(p,root/p.name);n+=1
print('snapshot files',n)
PY
nvidia-smi -i GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2 --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi -i GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
