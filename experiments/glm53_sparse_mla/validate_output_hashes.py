"""Record complete input/output byte hashes for isolated compiler comparisons.

Run separately under each compiler, then compare records by scenario/version.
Hash equality inherits every baseline numerical limitation; it is not an
independent FP32-reference check. Copies and hashing are never timed as kernels.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path

import torch
import cutlass
import benchmark_source as source


parser = argparse.ArgumentParser()
parser.add_argument("--kernel-versions", nargs="+", required=True)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output-json", required=True)
args = parser.parse_args()


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


report = {
    "purpose": "complete byte-hash equivalence across isolated compiler processes; not accuracy or timing",
    "args": vars(args),
    "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
    "cutlass_module": cutlass.__file__,
    "torch": torch.__version__,
    "benchmark_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
    "kernel_sha256": {},
    "records": [],
}

with torch.inference_mode():
    for tokens, chunk, seed, masked in ((8192, 3, 1234, False),
                                        (8192, 3, 5678, False),
                                        (1024, 0, 5678, False),
                                        (2, 3, 1234, True)):
        source.LOCAL_TOKENS = tokens
        source.make_sparse_indices.__defaults__ = (tokens, source.TOPK)
        inputs = source.make_inputs(chunk, 0, seed)
        if masked:
            # Exact fixture from audit_mask_precision.py, including partial tile.
            indices = inputs["block_tables"].view(2, -1)
            indices[0, [0, 31, 32, 63, 64, 127, 128, 511, 1023, 2047]] = -1
            indices[1, 129:] = -1
            indices[1, [31, 32, 127]] = -1
            inputs["seq_lens"][1] = 129
        input_hashes = {key: tensor_hash(inputs[key]) for key in
                        ("query", "kv_cache", "block_tables", "seq_lens")}
        for version in args.kernel_versions:
            module = importlib.import_module(f"kernel_{version}")
            report["kernel_sha256"][version] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            run = module.make_runner(inputs, 128)
            for repeat in range(args.repeats):
                output = run()
                record = {
                    "local_tokens": tokens, "chunk": chunk, "seed": seed,
                    "masked": masked,
                    "version": version, "repeat": repeat,
                    "input_sha256": input_hashes,
                    "elements": output.numel(),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "output_sha256": tensor_hash(output),
                }
                report["records"].append(record)
                print(json.dumps(record), flush=True)
            del run, output
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(0 if all(row["finite"] for row in report["records"]) else 1)
