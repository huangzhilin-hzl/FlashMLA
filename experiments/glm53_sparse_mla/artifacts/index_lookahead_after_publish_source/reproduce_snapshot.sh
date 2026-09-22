#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
snap=Path('artifacts/snapshots/index_lookahead_after_publish_compile');assert not snap.exists();(snap/'v256').mkdir(parents=True)
for p in Path('artifacts/index_lookahead_compile/v256').iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v256'/p.name)
shutil.copy2('kernel_v256.py',snap/'v256'/'kernel_v256.py')
shutil.copy2('artifacts/v256_compile_preflight.log',snap/'v256_compile_preflight.log')
snap=Path('artifacts/snapshots/index_lookahead_after_publish_source');assert not snap.exists();snap.mkdir(parents=True)
for n in ('v256_source.csv','v256_source_summary.json','v256_source.log','v256_source_preflight.log'):shutil.copy2(Path('artifacts')/n,snap/n)
p=Path('artifacts/snapshots/v256/v256_ncu.ncu-rep')
if p.exists():p.unlink()
print('snapshots ready')
PY
