"""Report masked-input precision for every backend before failing the audit.

Use the same two-row input and original tolerance as validate_masks.py.
This is an accuracy-only diagnostic, not a latency or pass override.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import benchmark_source as source

parser = argparse.ArgumentParser()
parser.add_argument('--kernel-versions', nargs='+', required=True)
parser.add_argument('--output-json', required=True)
args = parser.parse_args()
source.LOCAL_TOKENS = 2
source.make_sparse_indices.__defaults__ = (2, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
inputs = source.make_inputs(3, 0, 1234)
indices = inputs['block_tables'].view(2, -1)
indices[0, [0, 31, 32, 63, 64, 127, 128, 511, 1023, 2047]] = -1
indices[1, 129:] = -1
indices[1, [31, 32, 127]] = -1
inputs['seq_lens'][1] = 129
q = inputs['query'].squeeze(1).float()
kv = inputs['kv_cache'].view(-1, 576).float()
references, counts = [], []
for row in range(2):
    slots = indices[row, :int(inputs['seq_lens'][row])]
    slots = slots[slots >= 0].long()
    counts.append(slots.numel())
    selected = kv[slots]
    references.append(torch.softmax((q[row] @ selected.T) * 0.0625, -1) @ selected[:, :512])
reference = torch.stack(references)
runners = {'trtllm': source.make_trtllm_case(inputs).run}
for version in args.kernel_versions:
    runners[version] = importlib.import_module(f'kernel_{version}').make_runner(inputs, 128)
outputs = {name: runner().reshape(2, 64, 512).float().clone() for name, runner in runners.items()}
report = {'purpose': 'accuracy only; unchanged masked-input tolerance',
          'benchmark_sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
          'valid_counts': counts, 'atol': 0.01, 'rtol': 0.05, 'cases': {}}
for name, actual in outputs.items():
    error = (actual-reference).abs()
    bad = (~torch.isfinite(actual)) | (error > 0.01 + 0.05*reference.abs())
    coords = bad.nonzero().cpu().tolist()
    stats = {'checked_elements': reference.numel(), 'mismatches': len(coords),
             'max_abs': error.max().item(), 'rmse': error.square().mean().sqrt().item(),
             'relative_rmse': (error.square().sum()/reference.square().sum()).sqrt().item(),
             'pass': not coords,
             'failures': [{'row': r, 'head': h, 'channel': c, 'actual': actual[r,h,c].item(),
                           'reference': reference[r,h,c].item()} for r,h,c in coords]}
    if name != 'trtllm':
        stats['kernel_sha256'] = hashlib.sha256(Path(f'kernel_{name}.py').read_bytes()).hexdigest()
        stats['unequal_vs_trtllm'] = (actual != outputs['trtllm']).sum().item()
        stats['same_failures_as_trtllm'] = stats['failures'] == report['cases']['trtllm']['failures']
    report['cases'][name] = stats
    print(name, {k:v for k,v in stats.items() if k != 'failures'}, flush=True)
Path(args.output_json).write_text(json.dumps(report, indent=2)+'\n')
raise SystemExit(0 if all(case['pass'] for case in report['cases'].values()) else 1)
