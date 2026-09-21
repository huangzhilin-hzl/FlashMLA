"""Check every requested row in batches against the unchanged FP32 reference.

This is an accuracy audit, not a performance benchmark. Failures are recorded
for every backend before returning a nonzero exit code; tolerances are fixed.
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
parser.add_argument('--include-trtllm', action='store_true')
parser.add_argument('--block-k', type=int, choices=(64, 128, 256), default=128)
parser.add_argument('--local-tokens', type=int, default=8192)
parser.add_argument('--chunk', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch-rows', type=int, default=256)
parser.add_argument('--output-json', required=True)
args = parser.parse_args()
source.LOCAL_TOKENS = args.local_tokens
source.make_sparse_indices.__defaults__ = (args.local_tokens, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
result = {
    'purpose': 'accuracy only; no latency claim', 'args': vars(args),
    'benchmark_sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
    'atol': 0.01, 'rtol': 0.05, 'cases': {},
}
with torch.inference_mode():
    inputs = source.make_inputs(args.chunk, 0, args.seed)
    runners = {}
    if args.include_trtllm:
        runners['trtllm'] = source.make_trtllm_case(inputs).run
    for version in args.kernel_versions:
        module = importlib.import_module(f'kernel_{version}')
        runners[version] = module.make_runner(inputs, args.block_k)
        result['cases'][version] = {'kernel_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    outputs = {name: run().reshape(args.local_tokens, 64, 512) for name, run in runners.items()}
    for name, output in outputs.items():
        result['cases'].setdefault(name, {}).update({
            'finite': bool(torch.isfinite(output).all().item()), 'mismatches': 0,
            'max_abs': 0.0, 'squared_error_sum': 0.0, 'examples': [],
        })
    reference_sq_sum = 0.0
    count = 0
    for start in range(0, args.local_tokens, args.batch_rows):
        stop = min(start + args.batch_rows, args.local_tokens)
        rows = list(range(start, stop))
        reference = source.reference_rows(inputs, rows)
        reference_sq_sum += reference.square().sum(dtype=torch.float64).item()
        count += reference.numel()
        for name, output in outputs.items():
            actual = output[start:stop].float()
            error = actual - reference
            stats = result['cases'][name]
            stats['max_abs'] = max(stats['max_abs'], error.abs().max().item())
            stats['squared_error_sum'] += error.square().sum(dtype=torch.float64).item()
            bad = ~torch.isclose(actual, reference, atol=0.01, rtol=0.05, equal_nan=False)
            mismatches = int(bad.sum().item())
            stats['mismatches'] += mismatches
            remaining = 10 - len(stats['examples'])
            if mismatches and remaining > 0:
                positions = bad.nonzero()[:remaining].tolist()
                for row, head, channel in positions:
                    ref = reference[row, head, channel].item()
                    value = actual[row, head, channel].item()
                    stats['examples'].append({'row': start + row, 'head': head, 'channel': channel,
                        'actual': value, 'reference': ref, 'abs_error': abs(value-ref),
                        'allowed_error': 0.01 + 0.05 * abs(ref)})
        if stop % 1024 == 0 or stop == args.local_tokens:
            print(json.dumps({'checked_rows': stop, 'mismatches': {k:v['mismatches'] for k,v in result['cases'].items()}}), flush=True)
    for name, stats in result['cases'].items():
        stats['checked_rows'] = args.local_tokens
        stats['checked_elements'] = count
        stats['rmse'] = (stats['squared_error_sum'] / count) ** 0.5
        stats['relative_rmse'] = (stats['squared_error_sum'] / max(reference_sq_sum, 1e-24)) ** 0.5
        stats['pass'] = stats['finite'] and stats['mismatches'] == 0
        if name != 'trtllm' and 'trtllm' in outputs:
            stats['unequal_vs_trtllm'] = int((outputs[name] != outputs['trtllm']).sum().item())
    Path(args.output_json).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
raise SystemExit(0 if all(v['pass'] for v in result['cases'].values()) else 1)
