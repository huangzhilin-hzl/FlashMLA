set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v227_compile_preflight.log 2>&1
mkdir -p artifacts/normal_pipeline_compile/v227_corrected_sync
cd artifacts/normal_pipeline_compile/v227_corrected_sync
PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python /tmp/glm53_sparse_mla_dev/compile_offline.py --initialize-cuda --kernel-version v227 --block-k 64 > compile.log 2>&1
cubin=( *.cubin )
cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
cuobjdump --dump-sass "${cubin[0]}" > sass.txt
cat resources.txt

/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re
matches=re.findall(r'tcgen05.ld.red[^;]+;',next(Path('.').glob('*.ptx')).read_text())
assert len(matches)==1 and matches[0].endswith(', 32;'),matches
print('LD.RED half split immediate32 verified')
PY
