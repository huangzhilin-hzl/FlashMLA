set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_pv_warm_recheck
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_pv_warm_recheck/preflight.log 2>&1
for round in 0 1 2; do
 if [[ "$round" == 1 ]]; then versions="v284 v286"; else versions="v286 v284"; fi
 for version in $versions; do
  timeout --kill-after=10s 240s /opt/sglang/bin/python -u bench.py --backends trtllm cute --kernel-version "$version" --block-k 128 --scope native --check-rows 512 --warmup-iters 20 --repeat-iters 100 --timing cuda-graph --cache warm --output-json "artifacts/query_pv_warm_recheck/${round}_${version}.json" > "artifacts/query_pv_warm_recheck/${round}_${version}.log" 2>&1
  tail -5 "artifacts/query_pv_warm_recheck/${round}_${version}.log"
 done
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json
root=Path('artifacts/query_pv_warm_recheck')
summary=[]
for i in range(3):
 r={'round':i,'order':['v284','v286'] if i==1 else ['v286','v284'],'medians':{}}
 for v in ['v284','v286']:
  d=json.loads((root/f'{i}_{v}.json').read_text())
  r['medians'][v]={x['case']:x['median_us'] for x in d['benchmarks']}
 r['v284_over_v286']=r['medians']['v284']['cute-v284/native']/r['medians']['v286']['cute-v286/native']
 summary.append(r)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
