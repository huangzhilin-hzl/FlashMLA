#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
snap=Path('artifacts/snapshots/index_lookahead_before_publish_compile');assert not snap.exists();(snap/'v255').mkdir(parents=True)
for p in Path('artifacts/index_lookahead_compile/v255').iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v255'/p.name)
shutil.copy2('kernel_v255.py',snap/'v255'/'kernel_v255.py')
shutil.copy2('artifacts/v255_compile_preflight.log',snap/'v255_compile_preflight.log')
snap=Path('artifacts/snapshots/index_lookahead_before_publish_source');assert not snap.exists();snap.mkdir(parents=True)
for n in ('v255_source.csv','v255_source_summary.json','v255_source.log','v255_source_preflight.log'):shutil.copy2(Path('artifacts')/n,snap/n)
p=Path('artifacts/snapshots/v255/v255_ncu.ncu-rep')
if p.exists():p.unlink()
print('snapshots ready')
PY
