python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version in [('v290','v292'),('v291','v293')]:
 old=(root/f'kernel_{parent}.py').read_text()
 start=old.index('                    # Acquired P-ready drains old Q readers and all compute Qbar waiters.\n')
 stop=old.index('                    with cute.arch.elect_one():\n                        for ntile',start)
 chunk=old[start:stop]
 moved=chunk.replace('# The issuer loads next Q before submitting the current final PV.',
                     '# Submit current PV first, then load next Q while it is in flight.')
 marker='                        tcgen05.commit(pv_done)\n'
 assert old.count(marker)==1
 s=old[:start]+old[stop:];s=s.replace(marker,marker+moved)
 restored=s.replace(moved,'')
 ready='''                    cute.arch.mbarrier_wait(p_ready, (tile_base + block) % 2)
                    tmem_after_sync()
                    cute.arch.fence_view_async_shared()
'''
 assert restored.count(ready)==1
 restored=restored.replace(ready,ready+chunk)
 assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(old))
 s=f'"""Iteration {version[1:]}: issuer next-Q prefetch after PV commit; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 ast.parse(s)
 p=root/f'kernel_{version}.py';assert not p.exists();p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
print('AST inverse restores each parent exactly before label changes')
PY
