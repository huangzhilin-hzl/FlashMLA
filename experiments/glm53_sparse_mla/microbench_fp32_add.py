"""GPU1-only diagnostic of scalar/packed FP32 recurrence scheduling.

This is not MLA timing. Per-SM cycle counts include loop/warp scheduling and
fixed timestamp/checksum overhead; they are not an isolated ISA latency claim.
Distinct input seeds prevent independent recurrences from being merged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

# CuTe records its dump working directory at import time. Normalize the CLI
# output path and enter it before importing the compiler runtime.
if __name__ == '__main__' and '--help' not in sys.argv and '-h' not in sys.argv:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--output-dir', type=Path)
    initial, _ = bootstrap.parse_known_args()
    if initial.output_dir is not None:
        initial_directory = initial.output_dir.resolve()
        initial_directory.mkdir(parents=True, exist_ok=True)
        os.chdir(initial_directory)
        sys.argv += ['--output-dir', str(initial_directory)]

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


GPU_UUID = 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'
UNROLL = 8


@dsl_user_op
def timed_recurrences(inputs, pairs, packed, loops, *, loc=None, ip=None):
    lines = ['{ .reg .u64 t0,t1; .reg .u32 counter; .reg .pred again;',
             f'.reg .f32 a<{pairs}>, b<{pairs}>;',
             f'.reg .b64 state<{pairs}>, increment;']
    if packed:
        lines.append('mov.b64 increment, {$2, $3};')
    for i in range(pairs):
        if packed:
            lines.append(f'mov.b64 state{i}, {{${4 + 2*i}, ${5 + 2*i}}};')
        else:
            lines += [f'mov.f32 a{i}, ${4 + 2*i};', f'mov.f32 b{i}, ${5 + 2*i};']
    lines += [f'mov.u32 counter, {loops};', 'bar.warp.sync 0xffffffff;',
              'mov.u64 t0, %clock64;', 'RECURRENCE_LOOP:']
    for _ in range(UNROLL):
        for i in range(pairs):
            if packed:
                lines.append(f'add.rn.f32x2 state{i}, state{i}, increment;')
            else:
                lines += [f'add.rn.f32 a{i}, a{i}, $2;',
                          f'add.rn.f32 b{i}, b{i}, $3;']
    lines += ['sub.u32 counter, counter, 1;',
              'setp.ne.u32 again, counter, 0;', '@again bra.uni RECURRENCE_LOOP;']
    if packed:
        for i in range(pairs):
            lines.append(f'mov.b64 {{a{i}, b{i}}}, state{i};')
    lines.append('add.rn.f32 $1, a0, b0;')
    for i in range(1, pairs):
        lines += [f'add.rn.f32 $1, $1, a{i};', f'add.rn.f32 $1, $1, b{i};']
    lines += ['bar.warp.sync 0xffffffff;', 'mov.u64 t1, %clock64;',
              'sub.u64 $0, t1, t0;', '}']
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64(), T.f32()]),
        [cutlass.Float32(inputs[i]).ir_value(loc=loc, ip=ip)
         for i in range(2 + 2*pairs)], '\n'.join(lines),
        ','.join(['=&l', '=&f'] + ['f'] * (2 + 2*pairs)),
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (cutlass.Int64(llvm.extractvalue(T.i64(), result, [0], loc=loc, ip=ip)),
            cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)))


class RecurrenceProbe:
    def __init__(self, pairs, packed, loops, threads):
        self.pairs, self.packed = pairs, packed
        self.loops, self.threads = loops, threads

    @cute.jit
    def __call__(self, inputs: cute.Tensor, cycles: cute.Tensor,
                 checksums: cute.Tensor, stream: cuda.CUstream):
        self.kernel(inputs, cycles, checksums).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, inputs: cute.Tensor, cycles: cute.Tensor, checksums: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        elapsed, checksum = timed_recurrences(inputs, self.pairs, self.packed, self.loops)
        cycles[tid] = elapsed
        checksums[tid] = checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, choices=(1, 4, 8), required=True)
    parser.add_argument('--threads', type=int, choices=(32, 128, 256), required=True)
    parser.add_argument('--packed', action='store_true')
    parser.add_argument('--loops', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        parser.error('select the authorized GPU1 UUID before running')
    if not 1 <= args.loops <= 4096 or not 1 <= args.repeats <= 50:
        parser.error('loops must be1..4096 and repeats1..50')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'result.json').exists():
        parser.error('choose a fresh output directory')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    os.chdir(output)
    # Powers of two make this bounded recurrence exactly representable in FP32.
    host_inputs = [2**-12, 2**-11] + [i*0.25 for i in range(2*args.pairs)]
    inputs = torch.tensor(host_inputs, device='cuda', dtype=torch.float32)
    cycles = torch.empty(args.threads, device='cuda', dtype=torch.int64)
    checksums = torch.empty(args.threads, device='cuda', dtype=torch.float32)
    tensors = [from_dlpack(x, assumed_align=16) for x in (inputs, cycles, checksums)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(RecurrenceProbe(args.pairs, args.packed, args.loops, args.threads),
                            *tensors, stream, options='--gpu-arch sm_103a --keep-cubin --keep-ptx')
    for _ in range(3):
        compiled(*tensors, stream)
    torch.cuda.synchronize()
    records = []
    steps = args.loops * UNROLL
    expected = sum(host_inputs[2+i] + steps*host_inputs[i % 2]
                   for i in range(2*args.pairs))
    for _ in range(args.repeats):
        compiled(*tensors, stream)
        torch.cuda.synchronize()
        observed = checksums.cpu().tolist()
        assert all(x == expected for x in observed), (observed, expected)
        elapsed = cycles.cpu().tolist()
        assert all(x > 0 for x in elapsed)
        records.append({'cycles_by_thread': elapsed,
                        'median_cycles': statistics.median(elapsed)})
    cubins = list(output.glob('*.cubin'))
    assert len(cubins) == 1, cubins
    for flag, name in [('--dump-resource-usage', 'resources.txt'), ('--dump-sass', 'sass.txt')]:
        (output/name).write_text(subprocess.check_output(['cuobjdump', flag, str(cubins[0])], text=True))
    median = statistics.median(x['median_cycles'] for x in records)
    report = {'purpose': __doc__, 'gpu_uuid': GPU_UUID, 'source_sha256': source_sha,
              'pairs': args.pairs, 'packed': args.packed, 'threads': args.threads,
              'loops': args.loops, 'unroll': UNROLL, 'steps_per_chain': steps,
              'repeats': args.repeats, 'checksum': expected, 'records': records,
              'median_cycles': median, 'cycles_per_step': median/steps,
              'cycles_per_scalar_add_per_thread': median/(steps*args.pairs*2)}
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records', 'purpose')}), flush=True)


if __name__ == '__main__':
    main()
