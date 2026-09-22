"""Audit exact BF16 output equivalence after scheduling-only changes.

This compares kernels, not mathematical accuracy. A match inherits every
baseline numerical limitation; it does not establish a FP32-reference pass.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
import benchmark_source as source

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='v054')
parser.add_argument('--candidate', required=True)
parser.add_argument('--block-k', type=int, choices=(64, 128, 256), default=128)
parser.add_argument('--local-tokens', type=int, default=8192)
parser.add_argument('--chunk', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--output-json', required=True)
args = parser.parse_args()
source.LOCAL_TOKENS = args.local_tokens
source.make_sparse_indices.__defaults__ = (args.local_tokens, source.TOPK)
result = {'purpose': 'kernel equivalence only; not FP32-reference accuracy or timing',
          'args': vars(args), 'kernel_sha256': {}, 'repeats': []}
with torch.inference_mode():
    inputs = source.make_inputs(args.chunk, 0, args.seed)
    runners = {}
    for version in (args.baseline, args.candidate):
        module = importlib.import_module(f'kernel_{version}')
        runners[version] = module.make_runner(inputs, args.block_k)
        result['kernel_sha256'][version] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    expected = runners[args.baseline]().clone()
    for repeat in range(args.repeats):
        actual = runners[args.candidate]()
        bad = actual.view(torch.int16) != expected.view(torch.int16)
        stats = {'repeat': repeat, 'finite': bool(torch.isfinite(actual).all().item()),
                 'elements': actual.numel(), 'bitwise_mismatches': int(bad.sum().item()),
                 'max_abs_difference': (actual.float() - expected.float()).abs().max().item()}
        stats['pass'] = stats['finite'] and stats['bitwise_mismatches'] == 0
        result['repeats'].append(stats)
        print(json.dumps(stats), flush=True)
Path(args.output_json).write_text(json.dumps(result, indent=2) + '\n')
raise SystemExit(0 if all(r['pass'] for r in result['repeats']) else 1)
