set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
root=Path('artifacts/fixed_topk_compile_precomment')
out={}
for v,parent,folder in [('v294','v284','query_epilogue_overlap_compile'),('v295','v287','query_pv_overlap_compile')]:
 a=(root/(v+'_dynamic')/'sass.txt').read_text()
 b=(Path('artifacts')/folder/parent/'sass.txt').read_text()
 words=lambda s:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',s)
 aa,bb=words(a),words(b)
 out[v]={'parent':parent,'encoding_words':[len(aa),len(bb)],'dynamic_native_identical':aa==bb,'mismatched_words':sum(x!=y for x,y in zip(aa,bb))}
 assert len(aa)==len(bb)
print(json.dumps(out))
(root/'dynamic_parent_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
PY
nvidia-smi -i GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2 --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
