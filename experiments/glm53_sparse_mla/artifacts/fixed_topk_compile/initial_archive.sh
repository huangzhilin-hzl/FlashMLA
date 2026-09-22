set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil
src=Path('artifacts/fixed_topk_compile')
dest=Path('artifacts/fixed_topk_compile_precomment')
assert src.exists() and not dest.exists()
for v in ['v294','v295']:
 shutil.copy2(f'kernel_{v}.py',src/f'kernel_{v}.py')
src.rename(dest)
print('preserved initial CPU compile and source before comment clarification')
PY
