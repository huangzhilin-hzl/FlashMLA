set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import ast,hashlib,json,re,shutil
root=Path('artifacts/fixed_topk_compile')
old=Path('artifacts/fixed_topk_compile_precomment')
out={}
words=lambda s:re.findall(r'/\* (0x[0-9a-f]{16}) \*/',s)
for version,parent,parent_folder in [('v294','v284','query_epilogue_overlap_compile'),('v295','v287','query_pv_overlap_compile')]:
 source=Path(f'kernel_{version}.py')
 assert ast.dump(ast.parse(source.read_text()))==ast.dump(ast.parse((old/source.name).read_text()))
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 for mode in ['full','dynamic']:
  p=root/(version+'_'+mode)
  sass=(p/'sass.txt').read_text();ptx=next(p.glob('*.ptx')).read_text()
  assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
  assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
  assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+('192' if version=='v294' else '176')+';',ptx)
  metadata=[json.loads(l) for l in (p/'compile.log').read_text().splitlines() if l.startswith('{')]
  assert any(d.get('sha256')==sha and d.get('cuda_context_initialized') is False for d in metadata)
  assert words(sass)==words((old/p.name/'sass.txt').read_text())
  lines=sass.splitlines()
  local=['\n'.join(lines[max(0,i-4):i+5]) for i,l in enumerate(lines) if re.search(r'\b(?:LDL|STL)\b',l)]
  (p/'local_sites.txt').write_text('\n\n'.join(local)+'\n')
  shutil.copy2(source,p/source.name)
  r={'source_sha256':sha,'comment_only_change_native_identical':True,
     'local_sites':len(local),'scalar_global_ldg_sites':len(re.findall(r'\bLDG\.',sass)),
     'actual_mode':json.loads((p/'mode.json').read_text())}
  if mode=='dynamic':
   other=Path('artifacts')/parent_folder/parent/'sass.txt'
   assert words(sass)==words(other.read_text())
   r.update({'dynamic_native_identical_to_parent':parent,'encoding_words':len(words(sass))})
  out[p.name]=r
(root/'control_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
