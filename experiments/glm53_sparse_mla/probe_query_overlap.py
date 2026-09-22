"""Diagnostic query-boundary timestamps, never a production latency benchmark.

Record first-QK submission-entry and compute-epilogue completion in instrumented
copies. The historical first_qk_issue field precedes descriptor preparation and
the actual MMA instruction; it is not Tensor Core execution time.
No added CTA barrier or shared-memory allocation. Global timestamp
stores and their register/control overhead can still perturb scheduling.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
VERSIONS = {"v278", "v281", "v284", "v285"}
EVENTS = {"first_qk_issue": 0, "compute_epilogue_done": 1}


def instrument_source(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseMLA")
    kernel = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    issuer = next(n for n in ast.walk(kernel) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "warp == 12")
    compute = next(n for n in ast.walk(kernel) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "warp < 8")
    qk_groups = []
    for node in ast.walk(issuer):
        if isinstance(node, ast.With):
            calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "mma_ws"]
            if calls and ast.unparse(calls[0].args[0]) == "tp + 256":
                qk_groups.append(node)
    assert len(qk_groups) == 1
    qk_groups[0].body[:0] = ast.parse(
        "if block == 0:\n    query_timestamp(timeline.iterator + qi * 2)"
    ).body
    query_loop = next(n for n in compute.body if isinstance(n, ast.For)
                      and ast.unparse(n.target) == "qi")
    assert isinstance(query_loop.body[-2], ast.AugAssign)
    assert ast.unparse(query_loop.body[-2].target) == "tile_base"
    assert ast.unparse(query_loop.body[-3]).startswith("tmem_after_sync()")
    query_loop.body[-2:-2] = ast.parse(
        "if tid == 0:\n    query_timestamp(timeline.iterator + qi * 2 + 1)"
    ).body
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree) + "\n"

    def once(old, new):
        nonlocal code
        assert code.count(old) == 1, old
        code = code.replace(old, new)

    once("tensor_map: cute.Tensor, stream: cuda.CUstream",
         "tensor_map: cute.Tensor, timeline: cute.Tensor, stream: cuda.CUstream")
    once("tensor_map: cute.Tensor, qk: cute.TiledMma",
         "tensor_map: cute.Tensor, timeline: cute.Tensor, qk: cute.TiledMma")
    once("self.kernel(q, kv, idx, lens, out, tensor_map, qk, pv,",
         "self.kernel(q, kv, idx, lens, out, tensor_map, timeline, qk, pv,")
    once("args = [from_dlpack",
         "timeline = torch.zeros((q.shape[0], 2), dtype=torch.int64, device=q.device)\n"
         "    args = [from_dlpack")
    once("(q, kv, idx, lens, out, tensor_map)]", "(q, kv, idx, lens, out, tensor_map, timeline)]")
    once("    return run", "    run.timeline = timeline\n    return run")
    helper = '''
@dsl_user_op
def query_timestamp(pointer, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int64(pointer.toint()).ir_value(loc=loc, ip=ip)],
        "{ .reg .u64 ticks; mov.u64 ticks, %globaltimer; st.global.u64 [$0], ticks; }",
        "l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

'''
    once("class SparseMLA:", helper + "class SparseMLA:")
    ast.parse(code)
    return code


def summarize(timestamps):
    batch = len(timestamps)
    grid = min(batch, 148)
    # Both timestamps in a pair belong to consecutive queries of the same CTA.
    deltas = [timestamps[qi - grid][1] - timestamps[qi][0]
              for qi in range(grid, batch)
              if timestamps[qi - grid][1] > 0 and timestamps[qi][0] > 0]
    return {
        "query_pairs": len(deltas),
        "next_qk_before_prior_epilogue_done": sum(delta > 0 for delta in deltas),
        "equal_timestamps": sum(delta == 0 for delta in deltas),
        "prior_epilogue_done_minus_next_qk_ns": None if not deltas else {
            "min": min(deltas), "median": statistics.median(deltas), "max": max(deltas)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["v278", "v284", "v281", "v285"])
    parser.add_argument("--local-tokens", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert set(args.versions) <= VERSIONS
    assert args.local_tokens > 0 and args.repeat > 0
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    generated = []
    for version in args.versions:
        original = ROOT / f"kernel_{version}.py"
        target = outdir / f"kernel_{version}_query_trace.py"
        target.write_text(instrument_source(original.read_text()))
        generated.append((version, original, target))
    if args.generate_only:
        print(json.dumps({"generated": [str(item[2]) for item in generated]}))
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2"
    os.chdir(outdir)
    import torch
    import benchmark_source as source
    source.LOCAL_TOKENS = args.local_tokens
    source.make_sparse_indices.__defaults__ = (args.local_tokens, source.TOPK)
    result = {"events": EVENTS, "local_tokens": args.local_tokens, "chunk": args.chunk,
              "notice": "Instrumented global timestamps; no production latency or saved-time claims.",
              "versions": {}}
    with torch.inference_mode():
        inputs = None if args.compile_only else source.make_inputs(args.chunk, 0, 1234)
        for version, original, target in generated:
            name = target.stem
            spec = importlib.util.spec_from_file_location(name, target)
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
                descriptions = [
                    (cutlass.Float8E4M3FN, (batch, 64, 576), (36864, 576, 1)),
                    (cutlass.Float8E4M3FN, (context, 576), (576, 1)),
                    (cutlass.Int32, (batch, 2048), (2048, 1)),
                    (cutlass.Int32, (batch,), (1,)),
                    (cutlass.BFloat16, (batch, 64, 512), (32768, 512, 1)),
                    (cutlass.Uint8, (module.TENSOR_MAP_BYTES,), (1,)),
                    (cutlass.Int64, (batch, 2), (2, 1)),
                ]
                tensors = [make_fake_tensor(dtype, shape, stride, assumed_align=16)
                           for dtype, shape, stride in descriptions]
                cute.compile(module.SparseMLA(128), *tensors, make_fake_stream(),
                             options="--gpu-arch sm_103a --keep-cubin --keep-ptx --ptxas-options -v")
                print(json.dumps({"version": version, "offline_compile": "PASS",
                                  "candidate_launched": False}), flush=True)
                continue
            parent = __import__(f"kernel_{version}")
            reference_run = parent.make_runner(inputs, 128)
            traced_run = module.make_runner(inputs, 128)
            reference = reference_run().clone()
            record = {"original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                      "instrumented_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "runs": []}
            result["versions"][version] = record
            for repeat in range(args.repeat):
                traced_run.timeline.zero_()
                candidate = traced_run()
                torch.cuda.synchronize()
                mismatches = int((reference.view(torch.int16) != candidate.view(torch.int16)).sum().item())
                finite = bool(torch.isfinite(candidate).all().item())
                timestamps = traced_run.timeline.cpu().tolist()
                run = {"repeat": repeat, "finite": finite, "bitwise_mismatches": mismatches,
                       "summary": summarize(timestamps), "timestamps_ns": timestamps}
                record["runs"].append(run)
                (outdir / "query_trace.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"version": version, **{k: v for k, v in run.items()
                                                        if k != "timestamps_ns"}}), flush=True)
                if mismatches or not finite:
                    raise RuntimeError("Instrumented output differs from its qualified parent")
            del traced_run, reference_run, reference, candidate
    print(f"[RESULT] {outdir / 'query_trace.json'}")


if __name__ == "__main__":
    main()
