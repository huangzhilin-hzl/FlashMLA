"""Print the measured full-target iteration table from committed raw evidence."""
import csv
import json
from pathlib import Path

root = Path(__file__).parent / "artifacts"
print("# Measured full-target results\n")
print("B300 physical GPU1; b8192, H64, D576/512, TopK2048, chunk3. Each row is a paired warm CUDA-event run with 3 warmups and 5 repeats. All listed runs passed the original 8-row FP32-reference check; this is sampled numerical validation. v002/v003 were later found memory-unsafe and are rejected. NCU metrics are from separate profiled invocations.\n")
print("| Version | Candidate µs | Paired TRT µs | TRT/candidate | Registers | Local read/write sectors (M) | Tensor active | Status |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for directory in sorted(root.glob("v[0-9][0-9][0-9]")):
    version = directory.name
    path = directory / f"{version}_full_event.json"
    if not path.exists():
        continue
    result = json.loads(path.read_text())
    measures = result["benchmarks"]
    candidate = next(x for x in measures if x["case"].startswith("cute-") and x["cache"] == "warm")
    baseline = next(x for x in measures if x["case"] == "trtllm/native" and x["cache"] == "warm")
    with (directory / f"{version}_ncu_raw.csv").open() as f:
        metrics = list(csv.DictReader(f))[1]
    def number(key):
        value = metrics.get(key, "").replace(",", "")
        return float(value) if value else None
    loads = number("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")
    stores = number("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
    traffic = "unmeasured" if loads is None or stores is None else f"{loads/1e6:.2f} / {stores/1e6:.2f}"
    tensor = number("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    status = "unsafe; rejected" if version in ("v002", "v003") else "prototype"
    if version == "v053":
        status = "all 8192 rows PASS, seed1234"
    if version == "v054":
        status = "same 9 all-row failures as TRT"
    if version in ("v049", "v051"):
        status = "fails expanded 512-row check"
    if version == "v033":
        status = "non-isolated timing; do not rank"
    print(f"| {version} | {candidate['median_us']:.2f} | {baseline['median_us']:.2f} | {candidate['speedup_vs_trtllm']:.4f}x | {metrics['launch__registers_per_thread']} | {traffic} | {tensor:.2f}% | {status} |")
print("\nRaw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. No fully validated implementation has yet established a speedup over TRTLLM.\n")
graph_paths = sorted(root.glob("v[0-9][0-9][0-9]_validation/*_graph.json"))
if graph_paths:
    print("## CUDA Graph validation runs\n")
    print("These are separate warm/cold runs with 20 warmups and 100 repeats; sampled row counts are shown explicitly.\n")
    print("| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |")
    print("|---|---:|---|---:|---:|---:|")
    for path in graph_paths:
        result = json.loads(path.read_text())
        version = path.name.split("_", 1)[0]
        for candidate in result["benchmarks"]:
            if not candidate["case"].startswith("cute-"):
                continue
            baseline = next(x for x in result["benchmarks"] if x["case"] == "trtllm/native" and x["cache"] == candidate["cache"])
            checked_rows = len(result["correctness"][candidate["case"]]["rows"])
            print(f"| {version} | {checked_rows} | {candidate['cache']} | {candidate['median_us']:.2f} | {baseline['median_us']:.2f} | {candidate['speedup_vs_trtllm']:.4f}x |")
    print()
print("See [ITERATIONS.md](ITERATIONS.md) for changes, failed hypotheses, correctness limits and source references.")
