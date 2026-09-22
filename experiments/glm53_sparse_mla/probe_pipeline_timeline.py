"""Diagnostic timestamps for selected CTA pipelines; never a speed benchmark.

Generate instrumented copies of qualified kernels without changing originals.
Timestamp writes use shared memory; a final CTA barrier precedes export. Those
changes can perturb scheduling/resources, so compare outputs and report limits.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EVENTS = {
    'compute_qk_wait': 0, 'compute_qk_ready': 1, 'scores_loaded': 2,
    'probability_high_ready': 3, 'compute_pv_wait': 4, 'compute_pv_ready': 5,
    'probability_published': 6, 'issuer_p_wait': 7, 'issuer_p_ready': 8,
    'pv_issue': 9, 'pv_commit': 10, 'qk_issue': 11, 'qk_commit': 12,
    'producer_empty_wait': 13, 'producer_empty_ready': 14,
    'producer_gathers_issued': 15, 'compute_full_wait': 16,
    'compute_full_ready': 17,
}
NEVENT = len(EVENTS)
NKEY_TILES = 16


def expr(text):
    return ast.parse(text).body[0]


def call(node, suffix):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func).endswith(suffix))


def marker(event, block='block', leader=None):
    text = f'trace_mark(trace_smem, qi, {block}, {EVENTS[event]})'
    return expr(f'if tid == {leader}:\n    {text}' if leader is not None else text)


class Instrument(ast.NodeTransformer):
    """Transform only the device kernel; preserve all original math expressions."""
    def __init__(self, issuer_tid):
        self.role = None
        self.issuer_elected = False
        self.issuer_tid = issuer_tid
        self.counts = {}

    def visit_If(self, node):
        previous = self.role
        test = ast.unparse(node.test)
        if 'warp >= 8' in test and f'warp < {self.issuer_tid // 32}' in test:
            self.role = 'producer'
        elif test == f'warp == {self.issuer_tid // 32}':
            self.role = 'issuer'
        elif test == 'warp < 8':
            self.role = 'compute'
        node.body = self.statements(node.body)
        self.role = previous
        node.orelse = self.statements(node.orelse)
        return node

    def visit_For(self, node):
        node.body = self.statements(node.body)
        node.orelse = self.statements(node.orelse)
        if self.role == 'producer' and ast.unparse(node.target) == 'block':
            node.body.append(marker('producer_gathers_issued', leader=256))
        return node

    def visit_While(self, node):
        node.body = self.statements(node.body)
        node.orelse = self.statements(node.orelse)
        return node

    def visit_With(self, node):
        previous = self.issuer_elected
        if self.role == 'issuer' and any('elect_one' in ast.unparse(item.context_expr) for item in node.items):
            self.issuer_elected = True
        node.body = self.statements(node.body)
        self.issuer_elected = previous
        return node

    def statements(self, nodes):
        result = []
        for index, original in enumerate(nodes):
            before, after = [], []
            if call(original, 'mbarrier_wait'):
                target = ast.unparse(original.value.args[0])
                phase = ast.unparse(original.value.args[1])
                if self.role == 'compute':
                    if target.startswith('qk_done'):
                        before = [marker('compute_qk_wait', leader=0)]
                        after = [marker('compute_qk_ready', leader=0)]
                    elif target.startswith('full'):
                        before = [marker('compute_full_wait', leader=0)]
                        after = [marker('compute_full_ready', leader=0)]
                    elif target == 'pv_done':
                        owner = ('num_blocks - 1' if 'num_blocks' in phase else
                                 'block - 1' if 'block - 1' in phase else 'block')
                        before = [marker('compute_pv_wait', owner, 0)]
                        after = [marker('compute_pv_ready', owner, 0)]
                elif self.role == 'issuer' and target == 'p_ready':
                    leader = None if self.issuer_elected else self.issuer_tid
                    before = [marker('issuer_p_wait', leader=leader)]
                    after = [marker('issuer_p_ready', leader=leader)]
                elif self.role == 'producer' and target.startswith('empty'):
                    before = [marker('producer_empty_wait', leader=256)]
                    after = [marker('producer_empty_ready', leader=256)]
            if self.role == 'compute':
                if call(original, 'fence_view_async_tmem_load') and index:
                    previous = nodes[index - 1]
                    if call(previous, 'cute.copy') and ast.unparse(previous.value.args[0]) == 'scopy':
                        after.append(marker('scores_loaded', leader=0))
                if call(original, 'packed_p.store'):
                    value = original.value.args[0]
                    if isinstance(value, ast.Call) and ast.unparse(value.func) == 'probs.to':
                        after.append(marker('probability_high_ready', leader=0))
                if call(original, 'mbarrier_arrive') and ast.unparse(original.value.args[0]) == 'p_ready':
                    after.append(marker('probability_published', leader=0))
            if self.role == 'issuer':
                if isinstance(original, ast.For) and ast.unparse(original.target) == 'ntile':
                    before.append(marker('pv_issue'))
                if isinstance(original, ast.For) and ast.unparse(original.target) == 'k':
                    loop = ast.unparse(original.iter)
                    if loop.startswith('cutlass.range(16,'):
                        body = ast.unparse(original)
                        owner = ('0' if 'boot_k' in body else 'next_block'
                                 if 'early_k' in body or 'fallback_k' in body else 'block')
                        before.append(marker('qk_issue', owner))
                if call(original, 'tcgen05.commit'):
                    target = ast.unparse(original.value.args[0])
                    if target.startswith('qk_done'):
                        owner = '0' if 'boot_stage' in target else 'next_block' if 'stage' in target else 'block'
                        after.append(marker('qk_commit', owner))
                    elif target == 'pv_done':
                        after.append(marker('pv_commit'))
            for item in before + after:
                key = ast.unparse(item)
                self.counts[key] = self.counts.get(key, 0) + 1
            result.extend(before)
            result.append(self.visit(original))
            result.extend(after)
        return result


def instrument_source(source, stride, issuer_tid=384):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseMLA')
    kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    transform = Instrument(issuer_tid)
    kernel.body = transform.statements(kernel.body)
    # Allocate after existing objects so original shared addresses remain stable.
    anchor = next(i for i, n in enumerate(kernel.body)
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'empty' for t in n.targets))
    allocation = ast.parse(f'''trace_smem = alloc.allocate_tensor(cutlass.Int64, cute.make_layout({NKEY_TILES * NEVENT}))
if tid < {NKEY_TILES * NEVENT}:
    trace_smem[tid] = cutlass.Int64(0)
''').body
    kernel.body[anchor + 1:anchor + 1] = allocation
    kernel.body.extend(ast.parse(f'''cute.arch.barrier()
trace_tid = trace_thread_id()
if (qi % {stride} == 0) & (trace_tid < {NKEY_TILES * NEVENT}):
    timeline[qi // {stride}, trace_tid] = trace_smem[trace_tid]
''').body)
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + '\n'
    def once(old, new):
        nonlocal code
        if code.count(old) != 1:
            raise ValueError(f'Expected one source anchor: {old}')
        code = code.replace(old, new)
    once('tensor_map: cute.Tensor, stream: cuda.CUstream',
         'tensor_map: cute.Tensor, timeline: cute.Tensor, stream: cuda.CUstream')
    once('tensor_map: cute.Tensor, qk: cute.TiledMma',
         'tensor_map: cute.Tensor, timeline: cute.Tensor, qk: cute.TiledMma')
    once('self.kernel(q, kv, idx, lens, out, tensor_map, qk, pv,',
         'self.kernel(q, kv, idx, lens, out, tensor_map, timeline, qk, pv,')
    once('args = [from_dlpack',
         f'timeline = torch.empty(((q.shape[0] + {stride - 1}) // {stride}, {NKEY_TILES * NEVENT}), dtype=torch.int64, device=q.device)\n    args = [from_dlpack')
    once('(q, kv, idx, lens, out, tensor_map)]', '(q, kv, idx, lens, out, tensor_map, timeline)]')
    once('    return run', '    run.timeline = timeline\n    return run')
    helper = f'''
@dsl_user_op
def trace_thread_id(*, loc=None, ip=None):
    # Re-read at export so the per-thread shared address need not survive all roles.
    return cutlass.Int32(llvm.inline_asm(cutlass.Int32.mlir_type, [],
        "mov.u32 $0, %tid.x;", "=r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

@dsl_user_op
def trace_timestamp(pointer, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int32(pointer.toint()).ir_value(loc=loc, ip=ip)],
        "{{ .reg .u64 ticks; mov.u64 ticks, %globaltimer; st.shared.u64 [$0], ticks; }}",
        "r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@cute.jit
def trace_mark(trace_smem: cute.Tensor, qi: cutlass.Int32, block: cutlass.Int32, event: cutlass.Constexpr):
    if qi % {stride} == 0:
        trace_timestamp(trace_smem.iterator + block * {NEVENT} + event)

'''
    once('class SparseMLA:', helper + 'class SparseMLA:')
    ast.parse(code)
    return code, transform.counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--versions', nargs='+', default=['v190', 'v252', 'v197', 'v257'])
    p.add_argument('--local-tokens', type=int, default=8192)
    p.add_argument('--stride', type=int, default=1024)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--chunk', type=int, default=3)
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert set(args.versions) <= {'v190', 'v197', 'v252', 'v257', 'v265', 'v266'}
    assert args.local_tokens > 0 and args.stride > 0 and args.repeat > 0
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    generated = []
    for version in args.versions:
        original = ROOT / f'kernel_{version}.py'
        issuer_tid = 512 if version in {'v265', 'v266'} else 384
        code, counts = instrument_source(original.read_text(), args.stride, issuer_tid)
        path = outdir / f'kernel_{version}_timeline.py'
        path.write_text(code)
        generated.append((version, path, counts, hashlib.sha256(original.read_bytes()).hexdigest()))
    if args.generate_only:
        print(json.dumps({'generated': [str(x[1]) for x in generated]}))
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2'
    os.chdir(outdir)
    import torch
    import benchmark_source as source
    source.LOCAL_TOKENS = args.local_tokens
    source.make_sparse_indices.__defaults__ = (args.local_tokens, source.TOPK)
    result = {'events': EVENTS, 'stride': args.stride, 'local_tokens': args.local_tokens,
              'chunk': args.chunk, 'notice': 'Instrumented shared-memory timestamps; not latency rankings.', 'versions': {}}
    with torch.inference_mode():
        inputs = None if args.compile_only else source.make_inputs(args.chunk, 0, 1234)
        for version, path, counts, original_sha in generated:
            name = f'kernel_{version}_timeline'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if args.compile_only:
                import cutlass
                import cutlass.cute as cute
                from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
                torch.cuda.init()
                batch = args.local_tokens
                context = (args.chunk + 1) * source.CHUNK_TOKENS
                descriptors = [
                    (cutlass.Float8E4M3FN, (batch, 64, 576), (36864, 576, 1)),
                    (cutlass.Float8E4M3FN, (context, 576), (576, 1)),
                    (cutlass.Int32, (batch, 2048), (2048, 1)),
                    (cutlass.Int32, (batch,), (1,)),
                    (cutlass.BFloat16, (batch, 64, 512), (32768, 512, 1)),
                    (cutlass.Uint8, (module.TENSOR_MAP_BYTES,), (1,)),
                    (cutlass.Int64, ((batch + args.stride - 1) // args.stride, NKEY_TILES * NEVENT), (NKEY_TILES * NEVENT, 1)),
                ]
                tensors = [make_fake_tensor(dtype, shape, stride, assumed_align=16)
                           for dtype, shape, stride in descriptors]
                cute.compile(module.SparseMLA(128), *tensors, make_fake_stream(),
                    options='--gpu-arch sm_103a --keep-cubin --keep-ptx --ptxas-options -v')
                print(json.dumps({'version': version, 'offline_compile': 'PASS', 'candidate_launched': False}), flush=True)
                continue
            parent = __import__(f'kernel_{version}')
            reference_run = parent.make_runner(inputs, 128)
            traced_run = module.make_runner(inputs, 128)
            reference = reference_run().clone()
            runs = []
            for repeat in range(args.repeat):
                candidate = traced_run()
                torch.cuda.synchronize()
                mismatches = int((reference.view(torch.int16) != candidate.view(torch.int16)).sum().item())
                finite = bool(torch.isfinite(candidate).all().item())
                record = {'repeat': repeat, 'finite': finite, 'bitwise_mismatches': mismatches,
                          'timestamps_ns': traced_run.timeline.cpu().reshape(-1, NKEY_TILES, NEVENT).tolist()}
                runs.append(record)
                print(json.dumps({'version': version, 'repeat': repeat, 'finite': finite, 'bitwise_mismatches': mismatches}), flush=True)
                if mismatches or not finite:
                    result['versions'][version] = {'runs': runs, 'original_sha256': original_sha, 'instrumentation': counts}
                    (outdir / 'timeline.json').write_text(json.dumps(result, indent=2) + '\n')
                    raise RuntimeError('Instrumented output differs from qualified parent')
            result['versions'][version] = {'runs': runs, 'original_sha256': original_sha,
                'instrumented_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'instrumentation': counts}
            (outdir / 'timeline.json').write_text(json.dumps(result, indent=2) + '\n')
            del traced_run, reference_run, reference, candidate
    print(f'[RESULT] {outdir / "timeline.json"}')


if __name__ == '__main__':
    main()
