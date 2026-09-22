#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
dest=Path('artifacts/snapshots/index_cache_v263')
assert not dest.exists()
dest.mkdir(parents=True)
for p in Path('artifacts').glob('v263_*'):
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  shutil.copy2(p,dest/p.name)
print('files',len(list(dest.iterdir())))
PY
