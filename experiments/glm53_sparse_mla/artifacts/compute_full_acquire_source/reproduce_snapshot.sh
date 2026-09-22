#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
snap=Path('artifacts/snapshots/compute_full_acquire_source');assert not snap.exists();snap.mkdir(parents=True)
for name in ('v249_source.csv','v249_source_summary.json','v249_source.log','v249_source_preflight.log'):
 shutil.copy2(Path('artifacts')/name,snap/name)
print('snapshot files',sum(p.is_file() for p in snap.iterdir()))
PY
