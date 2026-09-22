#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/index_cache_completed')
assert not dest.exists()
(dest/'v264').mkdir(parents=True)
for p in Path('artifacts').glob('v264_*'):
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  shutil.copy2(p,dest/'v264'/p.name)
cache=dest/'index_cache_traffic'
cache.mkdir()
for p in Path('artifacts/index_cache_traffic').iterdir():
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  shutil.copy2(p,cache/p.name)
print(json.dumps({p.name:len(list(p.iterdir())) for p in dest.iterdir()}))
PY
