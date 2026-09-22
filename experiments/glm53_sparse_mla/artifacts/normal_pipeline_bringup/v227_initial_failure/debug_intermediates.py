import os,sys,json,hashlib
from pathlib import Path
import torch
import benchmark_source as source
p=Path('artifacts/v227_initial_failure')
s=Path('kernel_v227.py').read_text()
s=s.replace("dtype=torch.bfloat16)","dtype=torch.float32)").replace("out = torch.empty((q.shape[0], 64, 512)", "out = torch.empty((q.shape[0], 64, 2048)")
s=s.replace("                newmax = cute.arch.fmax(loaded_max[0], rowmax)", """                for j in cutlass.range(32, unroll_full=True):
                    out[qi, head, coords[j][1]] = rs[j]
                out[qi, head, 128 + ctid % 32 // 16] = loaded_max[0]
                newmax = cute.arch.fmax(loaded_max[0], rowmax)""")
s=s.replace("                rowsum = rowsum * correction + sum_tree[0]", """                for j in cutlass.range(32, unroll_full=True):
                    out[qi, head, 64 + coords[j][1]] = probs[j]
                out[qi, head, 132 + ctid % 32 // 16] = newmax
                out[qi, head, 136 + ctid % 32 // 16] = sum_tree[0]
                rowsum = rowsum * correction + sum_tree[0]""")
s=s.replace("                packed_p.store(probs.to(cutlass.Float8E4M3FN))", """                packed_p.store(probs.to(cutlass.Float8E4M3FN))
                for j in cutlass.range(32, unroll_full=True):
                    out[qi, head, 192 + coords[j][1]] = packed_p[j].to(cutlass.Float32)""")
a=s.index('                    packed_out.store(')
b=s.index('            tmem_before_sync()',a)
s=s[:a]+"""                    for j in cutlass.range(32, unroll_full=True):
                        channel = plane * 256 + tile * 64 + coords[j][1]
                        out[qi, head, 512 + channel] = rc[j]
                        out[qi, head, 1024 + channel] = rc[j] * norm
"""+s[b:]
(p/'debug_kernel.py').write_text(s)
sys.path.insert(0,str(p.resolve()))
import debug_kernel
source.LOCAL_TOKENS=2
source.make_sparse_indices.__defaults__=(2,source.TOPK)
inputs=source.make_inputs(3,0,1234)
inputs['seq_lens'].fill_(64)
inputs['block_tables'].view(2,-1)[:,64:]=-1
torch.backends.cuda.matmul.allow_tf32=False
run=debug_kernel.make_runner(inputs,64)
actual=run().clone()
torch.cuda.synchronize()
records=[]
for row in range(2):
 q=inputs['query'][row].reshape(64,576).float()
 slots=inputs['block_tables'].view(2,-1)[row,:64].long()
 kv=inputs['kv_cache'].reshape(-1,576)[slots].float()
 scores=q@kv.T
 probs=torch.exp2((scores-scores.max(-1,keepdim=True).values)*(.0625*__import__('math').log2(__import__('math').e)))*448
 staged_p=actual[row,:,192:256]
 pv=staged_p@kv[:,:512]
 r={'row':row,'score_maxerr':(actual[row,:,:64]-scores).abs().max().item(),
    'prob_maxerr':(actual[row,:,64:128]-probs).abs().max().item(),
    'pv_maxerr_from_actual_p':(actual[row,:,512:1024]-pv).abs().max().item(),
    'loaded_max_err':(actual[row,:,128:130]-scores.reshape(64,2,32).max(-1).values).abs().max().item(),
    'head0_scores':actual[row,0,:64].tolist(),'head0_score_ref':scores[0].tolist(),
    'head0_prob':actual[row,0,64:128].tolist(),'head0_prob_ref':probs[0].tolist(),
    'head0_actual_pv':actual[row,0,512:520].tolist(),'head0_pv_ref':pv[0,:8].tolist()}
 records.append(r)
 print(json.dumps({k:v for k,v in r.items() if not k.startswith('head0')}),flush=True)
torch.save({'intermediates':actual.cpu(),'inputs':{k:v.cpu() for k,v in inputs.items() if isinstance(v,torch.Tensor)}},p/'debug_tensors.pt')
(p/'debug_intermediates.json').write_text(json.dumps(records,indent=2)+'\n')
