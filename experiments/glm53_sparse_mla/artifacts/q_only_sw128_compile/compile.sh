set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=""
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/q_only_sw128_compile
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json,os
Path('artifacts/q_only_sw128_compile/offline_mode.json').write_text(json.dumps({
 'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
 'initialize_cuda':False,'kernel_launch':False,
 'reason':'min_blocks_per_mp=1 does not take CuTeDSL device-attribute carveout branch; explicitly hide all devices and compile fake tensors.'
},indent=2)+'\n')
PY
for version in v298; do
  mkdir -p artifacts/q_only_sw128_compile/$version
  (
    cd artifacts/q_only_sw128_compile/$version
    PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python /tmp/glm53_sparse_mla_dev/compile_offline.py --kernel-version "$version" --block-k 128 > compile.log 2>&1
    cubin=( *.cubin )
    cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
    cuobjdump --dump-sass "${cubin[0]}" > sass.txt
    cat resources.txt
  )
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
result={}
for version in ['v298']:
 p=Path('artifacts/q_only_sw128_compile')/version
 s=(p/'sass.txt').read_text()
 ptx=next(p.glob('*.ptx')).read_text()
 keys=['UTCQMMA.WS','MOV.SPILL','R2UR.FILL','R2UR','UIADD3','ULOP3','LDL','STL','BRA.U.ANY']
 r={'instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,
    'resources':(p/'resources.txt').read_text(),'counts':{k:s.count(k) for k in keys},
    'ptx_mma':ptx.count('tcgen05.mma.ws'),'ptx_commits':ptx.count('tcgen05.commit.')}
 result[version]=r
 print(version,r)
Path('artifacts/q_only_sw128_compile/summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY

