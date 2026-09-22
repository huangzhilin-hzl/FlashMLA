"""Bounded unprofiled output-equivalence diagnostic; no timing or FP32 claim."""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import torch
import benchmark_source as source

p=argparse.ArgumentParser()
p.add_argument('--sizes',type=int,nargs='+',default=[2,148,296,8192])
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--baseline',default='v232')
p.add_argument('--candidate',default='v233')
p.add_argument('--output-json',required=True)
a=p.parse_args()
d={'purpose':__doc__,'args':vars(a),'kernel_sha256':{},'records':[], 'benchmark_sha256':hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(), 'guardrails':os.getenv('MLA_TMEM_GUARDRAILS')=='1'}
mods={n:importlib.import_module(f'kernel_{n}') for n in [a.baseline,a.candidate]}
for n,m in mods.items():d['kernel_sha256'][n]=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
with torch.inference_mode():
 for size in a.sizes:
  source.LOCAL_TOKENS=size
  source.make_sparse_indices.__defaults__=(size,source.TOPK)
  inp=source.make_inputs(3,0,1234)
  runners={n:m.make_runner(inp,128) for n,m in mods.items()}
  ref=runners[a.baseline]().clone()
  for repeat in range(a.repeats):
   out=runners[a.candidate]();bad=out.view(torch.int16)!=ref.view(torch.int16)
   row_bad=bad.reshape(size,-1).sum(1)
   rec={'size':size,'repeat':repeat,'finite':bool(out.isfinite().all().item()),
        'mismatches':int(bad.sum().item()),
        'plane_mismatches':[int(bad[:,:,i*256:(i+1)*256].sum().item()) for i in range(2)],
        'max_abs':float((out.float()-ref.float()).abs().max().item()),
        'bad_rows':row_bad.nonzero().flatten().cpu().tolist(),
        'bad_counts_by_row':row_bad.cpu().tolist()}
   d['records'].append(rec)
   print(json.dumps({k:v for k,v in rec.items() if k not in ['bad_rows','bad_counts_by_row']}),flush=True)
  del runners,ref,out,bad,row_bad,inp
  torch.cuda.empty_cache()
Path(a.output_json).write_text(json.dumps(d,indent=2)+'\n')
