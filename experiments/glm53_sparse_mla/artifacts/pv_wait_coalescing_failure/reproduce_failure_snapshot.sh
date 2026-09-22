#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json,hashlib
snap=Path('artifacts/snapshots/pv_wait_coalescing_failure');assert not snap.exists();snap.mkdir(parents=True)
files=list(Path('artifacts').glob('v244_*'))
files=[p for p in files if p.is_file() and p.suffix in ('.log','.json')]
for p in files: shutil.copy2(p,snap/p.name)
for v in ('v244','v245'):shutil.copy2('kernel_'+v+'.py',snap/('kernel_'+v+'.py'))
(snap/'status.json').write_text(json.dumps(dict(v244='rejected: synccheck Missing wait, exit86; no uninstrumented equivalence or timing',v245='rejected same protocol; compiled only, no launch',reference='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-primary-phase'),indent=2)+'\n')
print('snapshot files',sum(p.is_file() for p in snap.iterdir()))
PY
