#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/kv_prefetch_cache
/opt/sglang/bin/python check_gpu_idle.py > artifacts/kv_prefetch_cache/preflight.log 2>&1
for version in v190 v254; do
 ncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,lts__t_requests.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum --force-overwrite -o artifacts/kv_prefetch_cache/${version} /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version "$version" --block-k 128 > artifacts/kv_prefetch_cache/${version}.log 2>&1
 ncu --import artifacts/kv_prefetch_cache/${version}.ncu-rep --page raw --csv > artifacts/kv_prefetch_cache/${version}.csv
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,json
d={}
for v in ('v190','v254'):
 units,row,*_=list(csv.DictReader((Path('artifacts/kv_prefetch_cache')/(v+'.csv')).open()))
 d[v]={k:{'value':row[k],'unit':units[k]} for k in row if k.startswith(('gpu__','dram__','lts__'))}
Path('artifacts/kv_prefetch_cache/summary.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
PY
