python3 - <<'PY'
from pathlib import Path
import ast,hashlib
root=Path('experiments/glm53_sparse_mla')
qload='''                if warp == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(qbar, 64 * 576)
                        for col_block in cutlass.range(4, unroll_full=True):
                            raw = cute.recast_ptr(sq.iterator) + cute.assume(col_block * 64 * 128, divby=128)
                            load_q_tile(raw, tensor_map.iterator + 256, col_block * 128, qi * 64, qbar)
                        load_q_tile(cute.recast_ptr(sq_tail.iterator), tensor_map.iterator + 384, 0, qi * 64, qbar)
'''
initial=qload.replace('if warp == 0:', 'if (warp == 0) & (qi < min(q.shape[0], 148)):')
prefetch=qload.replace('if warp == 0:', 'if (warp == 0) & (qi + min(q.shape[0], 148) < q.shape[0]):')
prefetch=prefetch.replace('qi * 64','(qi + min(q.shape[0], 148)) * 64')
for parent,version in [('v278','v284'),('v281','v285')]:
 s=(root/f'kernel_{parent}.py').read_text()
 s=f'"""Iteration {version[1:]}: next-query Q load before output epilogue; based on {parent}.\n'+s[s.index('\n')+1:]
 s=s.replace(f'"{parent} requires',f'"{version} requires')
 assert s.count(qload)==1;s=s.replace(qload,initial)
 marker='\n                head = coords_s[0][0] % 64\n'
 assert s.count(marker)==1
 s=s.replace(marker,'\n                # Old QK/PV are complete; next Q uses disjoint storage from O.\n'+prefetch+marker[1:])
 s=s.replace('# Next Q publication follows the previous compute epilogue rendezvous.', '# Next Q overlaps the prior epilogue; P-ready protects output reuse.')
 ast.parse(s)
 p=root/f'kernel_{version}.py';assert not p.exists();p.write_text(s)
 print(version,hashlib.sha256(s.encode()).hexdigest())
for batch in [1,2,148,149,296,297,512,513,8192]:
 grid=min(batch,148);loads=[]
 for cta in range(grid):
  loads.append(cta)
  for qi in range(cta,batch,grid):
   if qi+grid<batch:loads.append(qi+grid)
 assert sorted(loads)==list(range(batch))
print('each Q loaded once for small, odd and full query partitions')
PY
