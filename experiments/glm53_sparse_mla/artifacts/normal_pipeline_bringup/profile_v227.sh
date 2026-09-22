set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
import json,csv,re
from pathlib import Path
d=json.load(open('artifacts/v227_full_accuracy_seed1234.json'))
a=d['cases']['v227']['examples'];b=d['cases']['trtllm']['examples']
c=lambda x:[(e['row'],e['head'],e['channel']) for e in x]
print('failure_coords_identical',c(a)==c(b),c(a),c(b))
rows=list(csv.DictReader(open('artifacts/v227_ncu_raw.csv')))
r=rows[1]
print({k:v for k,v in r.items() if k.startswith(('launch__occupancy','sm__ctas_active','launch__shared_mem','launch__barrier','launch__waves','sm__maximum_warps'))})
ptx=next(Path('artifacts/normal_pipeline_compile/v227_corrected_sync').glob('*.ptx')).read_text()
for x in re.findall(r'tcgen05.mma[^;]+;',ptx)[:2]:print(x)
PY
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v227_source_preflight.log 2>&1
ncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --section SourceCounters --section WarpStateStats --force-overwrite -o artifacts/v227_unlocked_source /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version v227 --block-k 64 > artifacts/v227_unlocked_source.log 2>&1
ncu --import artifacts/v227_unlocked_source.ncu-rep --page source --print-source sass --csv > artifacts/v227_unlocked_source.csv
/opt/sglang/bin/python summarize_source_sass.py artifacts/v227_unlocked_source.csv artifacts/v227_unlocked_source_summary.json
mkdir -p artifacts/snapshots/normal_pipeline_source
cp artifacts/v227_unlocked_source* artifacts/v227_source_preflight.log artifacts/snapshots/normal_pipeline_source/
