set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,json
root=Path('artifacts');dest=root/'snapshots/v294_qualification_before_busy'
assert not dest.exists();dest.mkdir(parents=True)
shutil.copy2(root/'v294_gpu_preflight.log',root/'v294_gpu_preflight_busy.log')
for p in root.glob('v294_*'):
 if p.is_file() and p.suffix in {'.json','.log','.txt'}:shutil.copy2(p,dest/p.name)
proof={}
for suffix,expected in [('fixed_memcheck',True),('masked_full_synccheck',True),('dynamic_mode_synccheck',False),('equivalence',True),('short_equivalence',False)]:
 d=json.loads((root/f'v294_{suffix}.json').read_text());v=d['versions']['v294']
 assert v['fixed_topk_2048']==expected and all(r['pass'] for r in v['repeats'])
 proof[suffix]={'full_topk':expected,'repeats':len(v['repeats']),'elements_per_repeat':v['repeats'][0]['elements']}
(dest/'mode_verification_summary.json').write_text(json.dumps(proof,indent=2)+'\n')
print('files',sum(p.is_file() for p in dest.iterdir()))
print(json.dumps(proof))
PY
