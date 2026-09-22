import json,hashlib
from pathlib import Path
import torch
import benchmark_source as source
import kernel_v227
source.LOCAL_TOKENS=2
source.make_sparse_indices.__defaults__=(2,source.TOPK)
inputs=source.make_inputs(3,0,1234)
torch.backends.cuda.matmul.allow_tf32=False
original=inputs['block_tables'].clone()
report=[]
for length in (64,128,256,2048):
 inputs['block_tables'].copy_(original)
 inputs['seq_lens'].fill_(length)
 inputs['block_tables'].view(2,-1)[:,length:]=-1
 run=kernel_v227.make_runner(inputs,64)
 actual=run().float().reshape(2,64,512)
 ref=source.reference_rows(inputs,[0,1])
 err=(actual-ref).abs()
 bad=~torch.isclose(actual,ref,atol=.01,rtol=.05)
 row={'length':length,'mismatches':bad.sum().item(),'maxabs':err.max().item(),
 'actual_absmax':actual.abs().max().item(),'ref_absmax':ref.abs().max().item(),
 'head_rmse':err.square().mean(-1).sqrt().tolist(),
 'actual0':actual[0,:4,:8].tolist(),'ref0':ref[0,:4,:8].tolist()}
 report.append(row)
 print(json.dumps({k:v for k,v in row.items() if k not in ['head_rmse','actual0','ref0']}),flush=True)
Path('artifacts/v227_initial_failure/debug_stages.json').write_text(json.dumps(report,indent=2)+'\n')
