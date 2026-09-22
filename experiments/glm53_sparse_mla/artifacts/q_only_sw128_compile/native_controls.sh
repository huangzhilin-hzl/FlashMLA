set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json,hashlib,shutil
root=Path('artifacts/q_only_sw128_compile');out={}
for version,parent,parent_folder in [('v298','v284','query_epilogue_overlap_compile')]:
 p=root/version;source=Path(f'kernel_{version}.py')
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 s=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text()
 parent_s=(Path('artifacts')/parent_folder/parent/'sass.txt').read_text()
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+('192' if version=='v298' else '176')+';',ptx)
 meta=[json.loads(l) for l in (p/'compile.log').read_text().splitlines() if l.startswith('{')]
 assert any(d.get('sha256')==sha and d.get('cuda_context_initialized') is False for d in meta)
 lines=s.splitlines();local=['\n'.join(lines[max(0,i-4):i+5]) for i,l in enumerate(lines) if re.search(r'\b(?:LDL|STL)\b',l)]
 (p/'local_sites.txt').write_text('\n\n'.join(local)+'\n')
 shutil.copy2(source,p/source.name)
 def counts(text):
  return {k:len(re.findall(r'\b'+re.escape(k)+r'\b',text)) for k in ['UTCQMMA.WS','MOV.SPILL','R2UR.FILL','R2UR','UIADD3','ULOP3','LDL','STL']}
 out[version]={'sha256':sha,'parent':parent,'counts':counts(s),'parent_counts':counts(parent_s),
  'local_sites':len(local),'ptx_mma':ptx.count('tcgen05.mma.ws'),'ptx_commits':ptx.count('tcgen05.commit.'),
  'bounds_and_role_budgets_verified':True,'runtime_launched':False}
 # Retain descriptor and transaction instructions for review, without assigning latency.
 selected=[l for l in ptx.splitlines() if any(x in l for x in ['mbarrier.arrive.expect_tx','cp.async.bulk.tensor','tcgen05.mma.ws','tcgen05.commit'])]
 (p/'ptx_pipeline_sites.txt').write_text('\n'.join(selected)+'\n')
(root/'native_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
