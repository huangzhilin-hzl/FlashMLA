#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import re,json
root=Path('experiments/glm53_sparse_mla/artifacts/role_query_compile')
r={}
for v in ['v271','v272']:
 p=root/v
 ptx=next(p.glob('*.ptx')).read_text()
 sass=(p/'sass.txt').read_text()
 resources=(p/'resources.txt').read_text()
 assert re.search(r'\.reqntid\s+512,\s*1,\s*1',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+64;',ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+192;',ptx)
 assert 'REG:128 STACK:8' in resources
 assert sass.count('USETMAXREG')==2
 assert Path(f'experiments/glm53_sparse_mla/kernel_{v}.py').read_bytes()==(p/f'kernel_{v}.py').read_bytes()
 r[v]={'threads':512,'initial_registers':128,'stack_bytes':8,'donor_registers':64,'compute_registers':192,'initial_and_final_pool':65536,'native_redistribution_sites':2,'source_matches_compiled_archive':True}
(root/'control.json').write_text(json.dumps(r,indent=2)+'\n')
(root/'control_invocation_note.json').write_text(json.dumps({'initial_assertion':'failed on an extra tab before the setmaxnreg immediate','correction':'whitespace-insensitive regular expression; native immediates64/192 independently inspected','runtime_effect':'no source or binary changes; the explicit text-format assertion was after initial guarded qualification'},indent=2)+'\n')
print(r)
PY

