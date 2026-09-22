"""Audit checked fixed-TopK dispatch and masked persistent equivalence.

Native timing excludes runner construction. The optional setup-check measurement
isolates the extra torch.all(...).item() after prior CUDA work is synchronized.
Rebuild a specialized runner when its length metadata changes.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import time

PARENTS = {"v294": "v284", "v295": "v287"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", choices=tuple(PARENTS), required=True)
    parser.add_argument("--fixture", choices=("full", "dynamic", "masked-full"), required=True)
    parser.add_argument("--local-tokens", type=int, default=513)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-setup-check", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    assert args.local_tokens >= 2 and args.repeats > 0
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2"
    import torch
    import benchmark_source as source

    source.LOCAL_TOKENS = args.local_tokens
    source.make_sparse_indices.__defaults__ = (args.local_tokens, source.TOPK)
    expected_full = args.fixture != "dynamic"
    chunk = 3 if expected_full else 0
    report = {
        "purpose": "dispatch and exact equivalence; not a FP32-reference or native timing audit",
        "args": {**vars(args), "output_json": str(args.output_json)},
        "expected_fixed_topk_2048": expected_full,
        "benchmark_sha256": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        "versions": {},
    }

    def save():
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    with torch.inference_mode():
        inputs = source.make_inputs(chunk, 0, args.seed)
        if args.fixture == "masked-full":
            indices = inputs["block_tables"].view(args.local_tokens, -1)
            indices[0, [0, 31, 32, 63, 64, 127, 128, 511, 1023, 2047]] = -1
            indices[1, 129:] = -1
            indices[1, [31, 32, 127]] = -1
            indices[-1, ::17] = -1
            indices[::37, :32] = -1
        assert bool(torch.all(inputs["seq_lens"] == 2048).item()) == expected_full
        report["length_range"] = [
            int(inputs["seq_lens"].min().item()), int(inputs["seq_lens"].max().item())
        ]
        if args.measure_setup_check:
            torch.cuda.synchronize()
            values = []
            for _ in range(5):
                begin = time.perf_counter_ns()
                selected = bool(torch.all(inputs["seq_lens"] == 2048).item())
                values.append((time.perf_counter_ns() - begin) / 1000.0)
                assert selected == expected_full
            report["setup_check"] = {
                "scope": "one GPU length reduction plus host read, synchronized prior work; excludes compilation",
                "samples_us": values,
                "first_us": values[0],
                "subsequent_median_us": statistics.median(values[1:]),
            }
        for version in args.versions:
            parent = PARENTS[version]
            baseline_module = importlib.import_module("kernel_" + parent)
            candidate_module = importlib.import_module("kernel_" + version)
            baseline_run = baseline_module.make_runner(inputs, 128)
            candidate_run = candidate_module.make_runner(inputs, 128)
            assert candidate_run.fixed_topk_2048 == expected_full
            reference = baseline_run().clone()
            record = {
                "parent": parent,
                "fixed_topk_2048": candidate_run.fixed_topk_2048,
                "kernel_sha256": {
                    name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    for name, module in [(parent, baseline_module), (version, candidate_module)]
                },
                "repeats": [],
            }
            report["versions"][version] = record
            for repeat in range(args.repeats):
                actual = candidate_run()
                mismatches = int((actual.view(torch.int16) != reference.view(torch.int16)).sum().item())
                finite = bool(torch.isfinite(actual).all().item())
                result = {
                    "repeat": repeat, "elements": actual.numel(), "finite": finite,
                    "bitwise_mismatches": mismatches, "pass": finite and mismatches == 0,
                }
                record["repeats"].append(result)
                save()
                print(json.dumps({"version": version, "fixture": args.fixture, **result}), flush=True)
                if not result["pass"]:
                    raise RuntimeError("Specialized candidate differs from its qualified parent")
            del candidate_run, baseline_run, reference, actual
    save()
    print(f"[RESULT] {args.output_json}")


if __name__ == "__main__":
    main()
