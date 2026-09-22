#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/qk_descriptor_controls')
assert not dest.exists()
counts={}
for name in ['qk_panel_compile','qk_main_compile']:
 n=0
 source=Path('artifacts')/name
 for p in source.rglob('*'):
  if p.is_file() and p.suffix in {'.py','.ptx','.json','.log','.txt'}:
   out=dest/name/p.relative_to(source)
   out.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(p,out)
   n+=1
 counts[name]=n
for version,category in [('v258','qk_panel_compile'),('v259','qk_panel_compile'),('v260','qk_main_compile'),('v261','qk_main_compile'),('v262','qk_main_compile')]:
 shutil.copy2(Path('kernel_'+version+'.py'),dest/category/version/('kernel_'+version+'.py'))
for name in ['v258','qk_descriptor_source']:
 (dest/name).mkdir(parents=True,exist_ok=True)
for p in Path('artifacts').glob('v258_*'):
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  target='qk_descriptor_source' if '_source' in p.name else 'v258'
  shutil.copy2(p,dest/target/p.name)
  counts[target]=counts.get(target,0)+1
print(json.dumps(counts))
PY
