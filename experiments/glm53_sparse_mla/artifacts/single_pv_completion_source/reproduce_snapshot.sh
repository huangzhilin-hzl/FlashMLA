#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
snap=Path('artifacts/snapshots/single_pv_completion_source');assert not snap.exists();snap.mkdir(parents=True)
for name in ('v247_source.csv','v247_source_summary.json','v247_source.log','v247_source_preflight.log'):
 shutil.copy2(Path('artifacts')/name,snap/name)
print('snapshot files',sum(p.is_file() for p in snap.iterdir()))
PY
