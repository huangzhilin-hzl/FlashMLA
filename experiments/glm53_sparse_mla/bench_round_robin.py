"""Rotate backend order while reusing the unchanged benchmark timing function.

This supplements, and does not replace, the original paired benchmark records.
Every measurement remains one launch per event pair, optionally via CUDA Graph.
"""
import argparse
import hashlib
import importlib
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import benchmark_source as source

parser = argparse.ArgumentParser()
parser.add_argument('--kernel-versions', nargs='+', required=True)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--check-rows', type=int, default=512)
parser.add_argument('--warmup-iters', type=int, default=20)
parser.add_argument('--repeat-iters', type=int, default=100)
parser.add_argument('--endpoint-telemetry', choices=('on', 'off'), default='off',
                    help='Opt in to nvidia-smi queries; their idle gaps alter the operating regime')
parser.add_argument('--output-json', required=True)
args = parser.parse_args()
source.LOCAL_TOKENS = 8192
source.make_sparse_indices.__defaults__ = (8192, source.TOPK)
torch.backends.cuda.matmul.allow_tf32 = False
inputs = source.make_inputs(3, 0, 1234)
cases = [source.make_trtllm_case(inputs)]
for version in args.kernel_versions:
    runner = importlib.import_module(f'kernel_{version}').make_runner(inputs, 128)
    cases.append(source.Case(f'cute-{version}/native', runner, 'FP8 Q/KV; block_k128; preallocated BF16 output'))
check_args = SimpleNamespace(rank=0, check_rows=args.check_rows, check_atol=0.01, check_rtol=0.05)
checks = source.check_cases(cases, inputs, check_args)
timing_args = SimpleNamespace(warmup_iters=args.warmup_iters, repeat_iters=args.repeat_iters, timing='cuda-graph')
properties = torch.cuda.get_device_properties(0)
l2_bytes = getattr(properties, 'L2_cache_size', 0) or 128*1024*1024
flush_bytes = max(2*l2_bytes, 256*1024*1024)
def gpu_snapshot():
    # Endpoint metadata only; this does not sample frequency inside a kernel.
    if args.endpoint_telemetry == 'off':
        return None
    result = subprocess.run(['nvidia-smi', '-i', 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2',
                             '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return result.stdout.strip()

records = []
for cache in ('warm', 'cold'):
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device='cuda') if cache == 'cold' else None
    for round_index in range(args.rounds):
        offset = round_index % len(cases)
        ordered = cases[offset:] + cases[:offset]
        for order_index, case in enumerate(ordered):
            before = gpu_snapshot()
            samples = source.measure_case(case, timing_args, flush)
            after = gpu_snapshot()
            record = {'round': round_index, 'order': order_index, 'case': case.name,
                      'cache': cache, 'gpu_before': before, 'gpu_after': after, 'median_us': statistics.median(samples),
                      'p05_us': float(np.percentile(samples, 5)),
                      'p95_us': float(np.percentile(samples, 95)), 'samples_us': samples}
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'samples_us'}), flush=True)
    del flush
report = {'purpose': 'rotating backend order; unchanged source.measure_case timing',
          'args': vars(args), 'gpu': properties.name, 'torch': torch.__version__,
          'benchmark_sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
          'kernel_sha256': {v: hashlib.sha256(Path(f'kernel_{v}.py').read_bytes()).hexdigest() for v in args.kernel_versions},
          'correctness': checks, 'benchmarks': records}
Path(args.output_json).write_text(json.dumps(report, indent=2)+'\n')
