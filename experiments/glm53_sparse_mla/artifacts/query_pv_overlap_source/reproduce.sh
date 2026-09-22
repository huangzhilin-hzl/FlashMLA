set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,collections,json,re,shutil,hashlib
root=Path('artifacts/query_pv_overlap_source')
assert not root.exists();root.mkdir()
for v,command in json.loads("{\"v286\":\"set -euo pipefail\\ncd /tmp/glm53_sparse_mla_dev\\nexport CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2\\nexport CUTE_DSL_ARCH=sm_103a\\n/opt/sglang/bin/python check_gpu_idle.py > artifacts/v286_source_preflight.log 2>&1\\nncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --section SourceCounters --section WarpStateStats --force-overwrite -o artifacts/v286_source /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version v286 --block-k 128 > artifacts/v286_source.log 2>&1\\nncu --import artifacts/v286_source.ncu-rep --page source --print-source sass --csv > artifacts/v286_source.csv\\n/opt/sglang/bin/python summarize_source_sass.py artifacts/v286_source.csv artifacts/v286_source_summary.json\\n\",\"v287\":\"set -euo pipefail\\ncd /tmp/glm53_sparse_mla_dev\\nexport CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2\\nexport CUTE_DSL_ARCH=sm_103a\\n/opt/sglang/bin/python check_gpu_idle.py > artifacts/v287_source_preflight.log 2>&1\\nncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --section SourceCounters --section WarpStateStats --force-overwrite -o artifacts/v287_source /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version v287 --block-k 128 > artifacts/v287_source.log 2>&1\\nncu --import artifacts/v287_source.ncu-rep --page source --print-source sass --csv > artifacts/v287_source.csv\\n/opt/sglang/bin/python summarize_source_sass.py artifacts/v287_source.csv artifacts/v287_source_summary.json\\n\"}").items():(root/(v+'_profile.sh')).write_text(command+'\n')
for parent,candidate in [('v284','v286'),('v285','v287')]:
 counts={}
 sources={}
 for v in [parent,candidate]:
  path=Path(f'artifacts/{v}_source.csv')
  sources[v]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
  c=collections.Counter()
  with path.open() as f:
   next(f)
   for row in csv.DictReader(f):
    s=row['Source'].strip()
    if s.startswith('@'):s=s.split(None,1)[1]
    if not s:continue
    value=row.get('Instructions Executed','').replace(',','')
    if value:c[s.split()[0]]+=int(float(value))
  counts[v]=c
 d={k:counts[candidate][k]-counts[parent][k] for k in counts[candidate].keys()|counts[parent].keys()}
 result={'totals':{v:sum(c.values()) for v,c in counts.items()},'counts':counts,'deltas':d,'sources':sources}
 (root/f'{parent}_{candidate}_opcode_counts.json').write_text(json.dumps(result,indent=2)+'\n')
 print(parent,candidate,result['totals'],'largest',sorted(d.items(),key=lambda kv:abs(kv[1]),reverse=True)[:12])
 for p in Path('artifacts').glob(candidate+'_source*'):
  if p.is_file() and p.suffix in {'.json','.csv','.log','.txt'}:shutil.copy2(p,root/p.name)
dest=Path('artifacts/snapshots/query_pv_overlap_source')
assert not dest.exists();shutil.copytree(root,dest)
print('snapshot files',sum(p.is_file() for p in dest.rglob('*')))
PY
