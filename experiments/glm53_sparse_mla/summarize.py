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
    if version in ("v053", "v077", "v084"):
        status = "all 8192 rows PASS, seed1234"
    if version in ("v066", "v067", "v069", "v075", "v081"):
        status = "all8192 rows PASS, seeds1234/5678"
    if version in ("v054", "v063"):
        status = "same 9 all-row failures as TRT"
    if version in ("v085", "v086"):
        status = "same 9/6 all-row failures as TRT, seeds1234/5678"
    if version == "v088":
        status = "bitwise v085 on all8192 rows, seed1234"
    if version == "v093":
        status = "bitwise v090 on full seed1234/mask; slower"
    if version == "v095":
        status = "full seed1234 bitwise equivalence; see iteration log"
    if version in ("v094", "v096", "v097", "v105", "v108"):
        status = "bitwise validated predecessor, full2seeds/short/masks; FP8 limits retained"
    if version in ("v101", "v102", "v103", "v104"):
        status = "bitwise v097 full seed1234/mask; no gain"
    if version == "v099":
        status = "bitwise v097 full seed1234/mask; spills; slower"
    if version == "v100":
        status = "bitwise v098 full seed1234/mask; no gain"
    if version == "v106":
        status = "bitwise v098, full2seeds/short/masks; higher precision"
    if version == "v107":
        status = "bitwise v105 full seed1234/mask; not promoted"
    if version in ("v110", "v111"):
        status = "bitwise v106 full seed1234/mask; no consistent gain"
    if version == "v112":
        status = "same9/6 full-seed failures asTRT; earlier fast path"
    if version == "v113":
        status = "smoke/eight-row and qualified sanitizers pass; slower; no full audit"
    if version == "v114":
        status = "full2seeds/short/masks PASS; earlier higher precision"
    if version in ("v115", "v116"):
        status = "bitwise predecessor full seed1234/mask; no short-run gain"
    if version in ("v117", "v118"):
        status = "smoke/eight-row and qualified sanitizers pass; slower; no full audit"
    if version == "v119":
        status = "bitwise v112 full seed1234/short; slower"
    if version == "v120":
        status = "bitwise v114 full seed1234/mask; no short-run gain"
    if version == "v121":
        status = "bitwise v112 full seed1234/mask; spills; slower"
    if version == "v123":
        status = "bitwise v112 full seed1234/mask; slower"
    if version == "v125":
        status = "bitwise v112 full2seeds/short/masks; earlier fast path"
    if version == "v126":
        status = "bitwise v112 full seed1234/mask; more spills; slower"
    if version == "v127":
        status = "bitwise v112 full seed1234/mask; spills unchanged; slower"
    if version == "v128":
        status = "bitwise v114 full2seeds/short/masks; current higher precision"
    if version == "v129":
        status = "bitwise v125 full2seeds/short/masks; cold gain, mixed warm; not promoted"
    if version == "v130":
        status = "bitwise v125 full seed1234/mask; mixed cache-policy gain"
    if version == "v131":
        status = "bitwise v125 full seed1234/mask; mandatory lookahead slower"
    if version == "v132":
        status = "bitwise v125 full seed1234/mask; conditional lookahead, no net gain"
    if version == "v133":
        status = "bitwise v128 full2seeds/short/masks; small cache-policy alternative"
    if version == "v134":
        status = "bitwise v125 full seed1234/mask; no net scheduling gain"
    if version == "v140":
        status = "bitwise v128 full seed1234/mask; paired correction slower"
    if version == "v141":
        status = "bitwise v138 full seed1234/mask; expanded audit pending"
    if version == "v138":
        status = "bitwise v125 full2seeds/short/masks; current fast path"
    if version == "v136":
        status = "guarded smoke/b512 and eight rows pass; slower; no full audit"
    if version == "v109":
        status = "bitwise v105 full seed1234; slower overlap control"
    if version == "v098":
        status = "bitwise v075, full2seeds/short/masks; higher precision"
    if version == "v092":
        status = "bitwise v075 on all8192 rows, seed1234"
    if version == "v091":
        status = "bitwise v090, full2seeds/short/masks; FP8 limits retained"
    if version == "v090":
        status = "bitwise v086, full2seeds/short/masks; FP8 limits retained"
    if version == "v089":
        status = "bitwise v086 on full/short audited inputs"
    if version == "v087":
        status = "bitwise v085 on all8192 rows, seed1234"
    if version == "v065":
        status = "bitwise v053 on all8192 rows, seed1234"
    if version in ("v049", "v051"):
        status = "fails expanded 512-row check"
    if version == "v033":
        status = "non-isolated timing; do not rank"
    print(f"| {version} | {candidate['median_us']:.2f} | {baseline['median_us']:.2f} | {candidate['speedup_vs_trtllm']:.4f}x | {metrics['launch__registers_per_thread']} | {traffic} | {tensor:.2f}% | {status} |")
print("\nRaw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. v090/v091/v094/v096/v097/v105/v108/v112/v125 have an observed sustained warm advantage in the extended eager-event, Graph and rotating-order runs, at baseline-level FP8 precision; v125 is the current fast path. Short five-event v125 tuning has a small observed advantage; v105 is near parity. Use the matching execution regime and retained distributions. v086 is near parity warm and its initial cold advantage does not reproduce in its rotating-order audit.\n")
graph_paths = sorted(root.glob("v[0-9][0-9][0-9]_validation*/*graph*.json"))
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
event_paths = sorted(p for p in root.glob("v[0-9][0-9][0-9]_validation*/*_event100.json") if "graph" not in p.name)
if event_paths:
    print("## Extended eager CUDA-event runs\n")
    print("20 warmups and 100 repeats; the unchanged original timing function.\n")
    print("| Version | Checked rows | Cache | Candidate µs | Paired TRT µs | TRT/candidate |")
    print("|---|---:|---|---:|---:|---:|")
    for path in event_paths:
        result = json.loads(path.read_text())
        for candidate in result["benchmarks"]:
            if not candidate["case"].startswith("cute-"):
                continue
            baseline = next(x for x in result["benchmarks"] if x["case"] == "trtllm/native" and x["cache"] == candidate["cache"])
            rows = len(result["correctness"][candidate["case"]]["rows"])
            print(f"| {path.name.split('_',1)[0]} | {rows} | {candidate['cache']} | {candidate['median_us']:.2f} | {baseline['median_us']:.2f} | {candidate['speedup_vs_trtllm']:.4f}x |")
    print()
round_paths = sorted(root.glob("*_round_robin/*_round_robin.json"))
if round_paths:
    print("## Same-process rotating-order audits\n")
    print("Ranges below are the minimum and maximum per-round medians or paired ratios, not confidence intervals.\n")
    print("| Audit | Candidate | Cache | Rounds | Candidate median range µs | Paired TRT/candidate range |")
    print("|---|---|---|---:|---:|---:|")
    for path in round_paths:
        records = json.loads(path.read_text())["benchmarks"]
        for name in sorted({r["case"] for r in records if r["case"].startswith("cute-")}):
            for cache in ("warm", "cold"):
                selected = [r for r in records if r["case"] == name and r["cache"] == cache]
                values = [r["median_us"] for r in selected]
                ratios = [next(b["median_us"] for b in records if b["case"] == "trtllm/native" and b["cache"] == cache and b["round"] == r["round"])/r["median_us"] for r in selected]
                print(f"| {path.stem} | {name} | {cache} | {len(values)} | {min(values):.2f}–{max(values):.2f} | {min(ratios):.4f}–{max(ratios):.4f}x |")
    print()
print("See [ITERATIONS.md](ITERATIONS.md) for changes, failed hypotheses, correctness limits and source references.")
