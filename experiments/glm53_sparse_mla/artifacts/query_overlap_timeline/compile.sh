set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_overlap_timeline/compile
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_overlap_timeline/compile/preflight.log 2>&1
for version in v278 v284 v281 v285; do
 dir=artifacts/query_overlap_timeline/compile/$version
 mkdir -p "$dir"
 /opt/sglang/bin/python -u probe_query_overlap.py --versions "$version" --compile-only --output-dir "$dir" > "$dir/compile.log" 2>&1
 (
  cd "$dir"
  cubin=( *.cubin )
  cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
  cuobjdump --dump-sass "${cubin[0]}" > sass.txt
  cat resources.txt
 )
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json,re
root=Path('artifacts/query_overlap_timeline/compile')
result={}
for v in ['v278','v284','v281','v285']:
 p=root/v;s=(p/'sass.txt').read_text()
 result[v]={'resources':(p/'resources.txt').read_text(),'LDL':len(re.findall(r'\bLDL\b',s)),'STL':len(re.findall(r'\bSTL\b',s)),'static_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2}
print(json.dumps(result,indent=2))
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
