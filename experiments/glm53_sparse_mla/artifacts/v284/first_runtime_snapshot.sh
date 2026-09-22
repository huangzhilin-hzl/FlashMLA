set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
dest=Path('artifacts/snapshots/v284_first_runtime')
assert not dest.exists()
dest.mkdir(parents=True)
n=0
for p in Path('artifacts').glob('v284_*'):
 if p.is_file() and p.suffix in {'.json','.log','.csv','.txt'}:
  shutil.copy2(p,dest/p.name)
  n+=1
r=json.loads(Path('artifacts/v284_full_event.json').read_text())
print({'files':n,'medians':[{k:x[k] for k in ['case','median_us']} for x in r['benchmarks']]})
PY
