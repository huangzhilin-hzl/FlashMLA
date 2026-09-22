#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/persistent_grid_round_robin
/opt/sglang/bin/python check_gpu_idle.py > artifacts/persistent_grid_round_robin/preflight.log 2>&1
timeout --kill-after=10s 300s /opt/sglang/bin/python -u bench_round_robin.py --kernel-versions v272 v274 v275 --rounds 5 --check-rows 512 --warmup-iters 20 --repeat-iters 100 --endpoint-telemetry off --output-json artifacts/persistent_grid_round_robin/persistent_grid_round_robin.json > artifacts/persistent_grid_round_robin/run.log 2>&1
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json,statistics
d=json.loads(Path('artifacts/persistent_grid_round_robin/persistent_grid_round_robin.json').read_text())
r=d['benchmarks']
summary={}
for cache in ['warm','cold']:
 summary[cache]={}
 for name in ['trtllm/native','cute-v272/native','cute-v274/native','cute-v275/native']:
  selected=[x for x in r if x['case']==name and x['cache']==cache]
  ratios=[next(y['median_us'] for y in r if y['case']=='cute-v272/native' and y['cache']==cache and y['round']==x['round'])/x['median_us'] for x in selected]
  summary[cache][name]={'medians_us':[x['median_us'] for x in selected],'v272_over_candidate':ratios}
Path('artifacts/persistent_grid_round_robin/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY

