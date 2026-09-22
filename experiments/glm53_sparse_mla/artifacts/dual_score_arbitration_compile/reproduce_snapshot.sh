#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
snap=Path('artifacts/snapshots/dual_score_arbitration_compile')
assert not snap.exists()
d=snap/'v250';d.mkdir(parents=True)
src=Path('artifacts/dual_score_arbitration_compile/v250')
for p in src.iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'): shutil.copy2(p,d/p.name)
shutil.copy2('kernel_v250.py',d/'kernel_v250.py')
shutil.copy2('artifacts/v250_compile_preflight.log',snap/'v250_compile_preflight.log')
print('snapshot files',sum(p.is_file() for p in snap.rglob('*')))
PY
