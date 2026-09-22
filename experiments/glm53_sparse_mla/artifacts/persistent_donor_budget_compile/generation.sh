python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version,donor,compute in [('v280','v282',72,184),('v281','v283',80,176)]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: donor{donor}/compute{compute} register allocation; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 assert s.count('setmaxregister_decrease(64)')==1
 s=s.replace('setmaxregister_decrease(64)',f'setmaxregister_decrease({donor})')
 old_compute=192 if parent=='v280' else 176
 assert s.count(f'setmaxregister_increase({old_compute})')==1
 s=s.replace(f'setmaxregister_increase({old_compute})',f'setmaxregister_increase({compute})')
 lines=s.splitlines()
 for i,line in enumerate(lines):
  if '# compute registers. Final budget is ' in line:
   lines[i]=f'        # compute registers. Final budget is 256*{donor} + 256*{compute} = 65536.'
  if '# Initial128*512 =' in line:
   lines[i]=f'        # Initial128*512 = final256*{donor} + 256*{compute} = 65536 registers.'
 s='\n'.join(lines)+'\n'
 assert (donor+compute)*256==65536
 ast.parse(s)
 p=root/f'kernel_{version}.py';assert not p.exists();p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY
