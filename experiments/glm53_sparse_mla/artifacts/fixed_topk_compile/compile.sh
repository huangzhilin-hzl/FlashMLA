set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=""
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/fixed_topk_compile
for version in v294 v295; do
 for mode in full dynamic; do
  mkdir -p "artifacts/fixed_topk_compile/${version}_${mode}"
  (
   cd "artifacts/fixed_topk_compile/${version}_${mode}"
   MLA_COMPILE_VERSION="$version" MLA_COMPILE_MODE="$mode" PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python - <<'PY' > compile.log 2>&1
import importlib,os,runpy,sys,json
from pathlib import Path
version=os.environ['MLA_COMPILE_VERSION']
full=os.environ['MLA_COMPILE_MODE']=='full'
module=importlib.import_module('kernel_'+version)
original=module.SparseMLA
module.SparseMLA=lambda block_k:original(block_k,full_topk=full)
mode={'version':version,'full_topk':full,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'initialize_cuda':False,'candidate_launched':False}
Path('mode.json').write_text(json.dumps(mode,indent=2)+'\n')
print(json.dumps(mode),flush=True)
sys.argv=['compile_offline.py','--kernel-version',version,'--block-k','128']
runpy.run_path('/tmp/glm53_sparse_mla_dev/compile_offline.py',run_name='__main__')
PY
   cubin=( *.cubin )
   cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
   cuobjdump --dump-sass "${cubin[0]}" > sass.txt
   cat resources.txt
  )
 done
done
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json
root=Path('artifacts/fixed_topk_compile');result={}
for p in sorted(root.iterdir()):
 if not p.is_dir():continue
 s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text()
 keys=['UTCQMMA.WS','MOV.SPILL','R2UR.FILL','R2UR','UIADD3','ULOP3','LDL','STL','BRA.U.ANY']
 r={'mode':json.loads((p/'mode.json').read_text()),'instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',s))//2,
    'resources':(p/'resources.txt').read_text(),'counts':{k:s.count(k) for k in keys},
    'ptx_mma':ptx.count('tcgen05.mma.ws'),'ptx_commits':ptx.count('tcgen05.commit.')}
 result[p.name]=r;print(p.name,r)
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
