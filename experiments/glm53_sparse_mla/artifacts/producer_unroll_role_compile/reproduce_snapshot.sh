#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
base=Path('artifacts/producer_unroll_role_compile')
snap=Path('artifacts/snapshots/producer_unroll_role_compile')
assert not snap.exists()
for p in base.rglob('*'):
 if p.is_file() and p.suffix in ('.json','.ptx','.txt','.log'):
  dest=snap/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2('artifacts/v240_compile_preflight.log',snap/'preflight.log')
print('snapshot files',sum(p.is_file() for p in snap.rglob('*')))
PY
