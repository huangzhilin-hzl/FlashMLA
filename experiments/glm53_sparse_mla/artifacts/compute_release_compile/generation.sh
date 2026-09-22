python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
old="""        if warp == 0:
            final_tp = cute.arch.retrieve_tmem_ptr(cutlass.Float32, alignment=16,
                ptr_to_buffer_holding_addr=holding)
            cute.arch.dealloc_tmem(final_tp, 512)"""
new='\n'.join('    '+line for line in old.splitlines())
for parent,version in [('v272','v278'),('v273','v279')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: compute-role TMEM release; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 assert s.count(old)==1
 s=s.replace(old,new)
 ast.parse(s)
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY
