set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_overlap_timeline/full
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_overlap_timeline/full/preflight.log 2>&1
timeout --kill-after=10s 300s /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 8192 --chunk 3 --repeat 3 --output-dir artifacts/query_overlap_timeline/full > artifacts/query_overlap_timeline/full/run.log 2>&1
tail -13 artifacts/query_overlap_timeline/full/run.log
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,math,statistics
root=Path('artifacts/query_overlap_timeline')
d=json.loads((root/'full/query_trace.json').read_text())
result={}
for v,record in d['versions'].items():
 assert record['instrumented_sha256']==hashlib.sha256((root/f'compile/{v}/kernel_{v}_query_trace.py').read_bytes()).hexdigest()
 assert record['original_sha256']==hashlib.sha256(Path(f'kernel_{v}.py').read_bytes()).hexdigest()
 deltas=[];quantum=0
 for run in record['runs']:
  assert run['bitwise_mismatches']==0 and run['finite']
  ts=run['timestamps_ns'];grid=min(len(ts),148)
  assert all(a>0 and b>=a for a,b in ts)
  assert run['summary']['query_pairs']==len(ts)-grid
  values=[b for row in ts for b in row]
  for a,b in zip(values,values[1:]):quantum=math.gcd(quantum,abs(b-a))
  deltas.extend(ts[q-grid][1]-ts[q][0] for q in range(grid,len(ts)))
 result[v]={'pairs':len(deltas),'next_qk_before_prior_epilogue':sum(x>0 for x in deltas),'equal':sum(x==0 for x in deltas),'median_delta_ns':statistics.median(deltas),'min_delta_ns':min(deltas),'max_delta_ns':max(deltas),'observed_timestamp_difference_gcd_ns':quantum,'notice':'Instrumented ordering observations; not production saved-time estimates.'}
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
print('raw bytes',(root/'full/query_trace.json').stat().st_size)
PY
