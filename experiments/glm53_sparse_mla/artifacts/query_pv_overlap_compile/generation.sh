python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
condition='if (warp == 0) & (qi + min(q.shape[0], 148) < q.shape[0]):'
for parent,version in [('v284','v286'),('v285','v287')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: prefetch next Q before final PV wait; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 begin=s.index('                '+condition+'\n')
 end=s.index('                head = coords_s[0][0] % 64\n',begin)
 old=s[begin:end]
 assert len(old.splitlines())==7
 fallback=old.replace(condition,'if (num_blocks <= 0) & (warp == 0) & (qi + min(q.shape[0], 148) < q.shape[0]):')
 inner='\n'.join('    '+line for line in old.splitlines())+'\n'
 inner=inner.replace(condition,'if (block + 1 == num_blocks) & (warp == 0) & (qi + min(q.shape[0], 148) < q.shape[0]):')
 assert s.count(old)==1;s=s.replace(old,fallback)
 marker='                    cute.arch.mbarrier_wait(pv_done, (tile_base + block) % 2)\n'
 assert s.count(marker)==1
 s=s.replace(marker,'                    # Last QK has consumed Q; prefetch may overlap the final PV.\n'+inner+marker)
 s=s.replace('# Old QK/PV are complete; next Q uses disjoint storage from O.', '# Preserve next-Q publication when the key loop has no iterations.')
 ast.parse(s)
 p=root/f'kernel_{version}.py';assert not p.exists();p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
for blocks in [0,1,2,15,16,17]:
 for has_next in [False,True]:
  loads=sum(1 for block in range(blocks) if block+1==blocks and has_next)
  loads+=int(blocks<=0 and has_next)
  assert loads==int(has_next)
print('next-Q publication occurs exactly once, including zero-iteration control flow')
PY
