set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
root=Path('artifacts');dest=root/'snapshots/v291_qualification_before_busy'
assert not dest.exists();dest.mkdir(parents=True)
for p in root.glob('v291_*'):
 if p.is_file() and p.suffix in {'.json','.log','.txt'}:shutil.copy2(p,dest/p.name)
shutil.copy2(root/'v291_gpu_preflight.log',root/'v291_gpu_preflight_busy.log')
shutil.copy2(root/'v291_gpu_preflight_busy.log',dest/'v291_gpu_preflight_busy.log')
print('files',sum(p.is_file() for p in dest.iterdir()))
PY
nvidia-smi -i GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2 --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
