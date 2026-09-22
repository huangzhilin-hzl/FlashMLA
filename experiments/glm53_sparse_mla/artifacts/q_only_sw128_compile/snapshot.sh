set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
root=Path('artifacts/q_only_sw128_compile')
shutil.copy2('audit_unified_sw128_layout.py',root/'audit_unified_sw128_layout.py')
dest=Path('artifacts/snapshots/q_only_sw128_compile')
assert not dest.exists()
n=0
for p in root.rglob('*'):
 if p.is_file() and p.suffix in {'.py','.ptx','.json','.log','.txt','.sh'}:
  target=dest/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,target);n+=1
print('CPU-only compile snapshot files',n)
PY
