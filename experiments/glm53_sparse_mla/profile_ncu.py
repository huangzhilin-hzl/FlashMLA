"""Profile exactly one warmed attention invocation, excluding data preparation.

Use with ncu --profile-from-start off. NCU durations are never timing results.
"""
import argparse
import importlib
import torch
import benchmark_source as source

parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['trtllm', 'cute'], required=True)
parser.add_argument('--kernel-version', default='v001')
parser.add_argument('--block-k', type=int, default=64)
parser.add_argument('--local-tokens', type=int, default=8192)
args = parser.parse_args()
source.LOCAL_TOKENS = args.local_tokens
source.make_sparse_indices.__defaults__ = (source.LOCAL_TOKENS, source.TOPK)
with torch.inference_mode():
    inputs = source.make_inputs(3, 0, 1234)
    if args.backend == 'trtllm':
        run = source.make_trtllm_case(inputs).run
    else:
        kernel = importlib.import_module(f'kernel_{args.kernel_version}')
        run = kernel.make_runner(inputs, args.block_k)
    for _ in range(10):
        output = run()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    output = run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
