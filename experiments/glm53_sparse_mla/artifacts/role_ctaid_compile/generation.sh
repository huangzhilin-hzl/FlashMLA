python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
helper='''@dsl_user_op
def role_cta_id(*, loc=None, ip=None):
    # Read inside each role so a common initial query index need not survive setup.
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [],
        "mov.u32 $0, %ctaid.x;", "=r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


'''
for parent,version in [('v278','v280'),('v279','v281')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: role-local CTA index read; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 assert s.count('        cta_id, _, _ = cute.arch.block_idx()\n')==1
 s=s.replace('        cta_id, _, _ = cute.arch.block_idx()\n','')
 assert s.count('for qi in cutlass.range(cta_id,')==3
 s=s.replace('for qi in cutlass.range(cta_id,','for qi in cutlass.range(role_cta_id(),')
 needle='@dsl_user_op\ndef tmem_before_sync'
 assert s.count(needle)==1
 s=s.replace(needle,helper+needle)
 ast.parse(s)
 p=root/f'kernel_{version}.py'
 assert not p.exists()
 p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
PY
