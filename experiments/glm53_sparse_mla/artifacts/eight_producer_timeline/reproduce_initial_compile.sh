#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
for version in v265 v266; do
  mkdir -p artifacts/eight_producer_timeline_compile/$version
  PYTHONPATH=/tmp/glm53_sparse_mla_dev timeout --kill-after=10s 180s /opt/sglang/bin/python -u probe_pipeline_timeline.py --versions "$version" --compile-only --output-dir "artifacts/eight_producer_timeline_compile/$version" > "artifacts/eight_producer_timeline_compile/$version/compile.log" 2>&1
  (
    cd "artifacts/eight_producer_timeline_compile/$version"
    cubin=( *.cubin )
    cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
    cuobjdump --dump-sass "${cubin[0]}" > sass.txt
    cat resources.txt
  )
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
r={}
for v in ['v265','v266']:
 p=Path('artifacts/eight_producer_timeline_compile')/v
 sass=(p/'sass.txt').read_text()
 ptx=next(p.glob('*.ptx')).read_text()
 r[v]={'resources':(p/'resources.txt').read_text(),'LDL_STL':len(re.findall(r'\b(?:LDL|STL)\b',sass)),'globaltimer':ptx.count('%globaltimer'),'shared_u64_stores':ptx.count('st.shared.u64'),'native_timer':len(re.findall(r'\bCS2R\b',sass))}
 print(v,r[v])
Path('artifacts/eight_producer_timeline_compile/resources_summary.json').write_text(json.dumps(r,indent=2)+'\n')
PY

