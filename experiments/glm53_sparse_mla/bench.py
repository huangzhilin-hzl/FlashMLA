"""Run the user benchmark unchanged, with an additional CuTeDSL candidate."""
import argparse
import importlib
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys
import benchmark_source as source

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--kernel-version', default='v001')
parser.add_argument('--local-tokens', type=int, default=8192)
parser.add_argument('--block-k', type=int, default=64)
probe, rest = parser.parse_known_args()
source.LOCAL_TOKENS = probe.local_tokens
source.make_sparse_indices.__defaults__ = (source.LOCAL_TOKENS, source.TOPK)
source.BACKENDS = (*source.BACKENDS, 'cute')
original_make_cases = source.make_cases
original_configuration = source.configuration

def configuration(inputs, cases, args):
    config = original_configuration(inputs, cases, args)
    kernel_path = Path(__file__).with_name(f'kernel_{probe.kernel_version}.py')
    config['candidate'] = dict(
        version=probe.kernel_version, block_k=probe.block_k,
        file_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        cutlass_dsl=version('nvidia-cutlass-dsl'),
        original_benchmark_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
    )
    return config

def make_cases(inputs, args):
    backends = args.backends
    args.backends = [b for b in backends if b != 'cute']
    cases = original_make_cases(inputs, args)
    args.backends = backends
    if 'cute' in backends:
        kernel = importlib.import_module(f'kernel_{probe.kernel_version}')
        run = kernel.make_runner(inputs, probe.block_k)
        cases.append(source.Case(f'cute-{probe.kernel_version}/native', run,
                     f'FP8 Q/KV; preallocated BF16 output; block_k={probe.block_k}'))
    return cases

source.make_cases = make_cases
source.configuration = configuration
source.main(rest)
