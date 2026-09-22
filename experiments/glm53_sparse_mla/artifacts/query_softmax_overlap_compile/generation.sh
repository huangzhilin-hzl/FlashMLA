python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
for parent,version in [('v286','v288'),('v287','v289')]:
 old=(root/f'kernel_{parent}.py').read_text()
 start=old.index('                    # Last QK has consumed Q; prefetch may overlap the final PV.\n')
 stop=old.index('                    cute.arch.mbarrier_wait(pv_done,',start)
 chunk=old[start:stop]
 moved=chunk.replace('# Last QK has consumed Q; prefetch may overlap the final PV.',
                     '# Last QK is complete; all compute Qbar waiters have joined.\n                    # Prefetch next Q during final softmax/correction and PV.')
 marker='''                    partial_max[cgroup * 128 + coords_s[0][0]] = newmax
                    tmem_before_sync()
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    tmem_after_sync()
'''
 assert old.count(marker)==1
 s=old[:start]+old[stop:]
 s=s.replace(marker,marker+moved)
 restored=s.replace(moved,'')
 pv='                    cute.arch.mbarrier_wait(pv_done, (tile_base + block) % 2)\n'
 assert restored.count(pv)==1
 restored=restored.replace(pv,chunk+pv)
 assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(old))
 s=f'"""Iteration {version[1:]}: next Q after last max rendezvous; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 ast.parse(s)
 p=root/f'kernel_{version}.py';assert not p.exists();p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
print('AST inverse restores each parent exactly before label changes')
PY
