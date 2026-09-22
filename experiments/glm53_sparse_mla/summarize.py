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
        status = "bitwise v114 full2seeds/short/masks; earlier higher precision"
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
        status = "bitwise v138 full2seeds/short/masks; small cache-policy alternative"
    if version in ("v143", "v144"):
        status = "bitwise v138 full2seeds/short/masks; small scheduling alternative"
    if version == "v145":
        status = "bitwise v128 full2seeds/short/masks; earlier higher precision"
    if version == "v146":
        status = "bitwise v138 full2seeds/short/masks; earlier fast path"
    if version == "v147":
        status = "bitwise v138 full seed1234/mask; correction prefetch slower"
    if version == "v148":
        status = "full2seeds/short/masks FP32 PASS; earlier higher precision"
    if version == "v149":
        status = "bitwise v145 full2seeds/short/masks; superseded by v148"
    if version == "v150":
        status = "bitwise v146 full2seeds/short/masks; mixed scheduling gain"
    if version == "v151":
        status = "bitwise v146 full seed1234/masks; no short-run gain"
    if version == "v152":
        status = "bitwise v148 full2seeds/short/masks; warm gain, mixed cold"
    if version == "v153":
        status = "bitwise v152 full2seeds/short/masks; earlier higher precision"
    if version == "v154":
        status = "bitwise v152 full seed1234/masks; no short-run gain"
    if version in ("v155", "v156"):
        status = "bitwise v152 full seed1234/masks; alternate PV tiles slower"
    if version == "v157":
        status = "bitwise v152 full2seeds/short/masks; packed scaling alternative"
    if version == "v158":
        status = "bitwise v146 full seed1234/masks; Q cache policy, no short gain"
    if version == "v159":
        status = "bitwise v153 full seed1234/masks; Q cache policy slower"
    if version == "v160":
        status = "bitwise v153 full2seeds/short/masks; earlier higher precision"
    if version in ("v161", "v162"):
        status = "guarded memory/sync and full seed1234/mask bits pass; load/max slower"
    if version in ("v163", "v164"):
        status = "guarded checks and full seed1234/mask bits pass; mask fallback slower"
    if version in ("v165", "v166"):
        status = "bitwise full seed1234/masks and qualified sanitizers pass; mask branch slower"
    if version in ("v177", "v178"):
        status = "guarded checks/full seed1234/mask bits pass; recovers regression, slower than defaults"
    if version in ("v179", "v180"):
        status = "full seed1234/mask bits and sanitizers pass; role-only control has no short gain"
    if version == "v183":
        status = "bitwise v146 full2seeds/short/masks; earlier fast path"
    if version == "v184":
        status = "bitwise v160 full2seeds/short/masks; earlier higher precision"
    if version == "v190":
        status = "bitwise v183 full2seeds/short/masks; current fast path"
    if version == "v191":
        status = "bitwise v184 full2seeds/short/masks; earlier higher precision"
    if version in ("v185", "v186"):
        status = "full2seeds/short/masks bitwise; mixed small gain; validated alternative"
    if version == "v194":
        status = "guarded checks/full seed1234/mask bits pass; register redistribution slower"
    if version == "v195":
        status = "full2seeds/short/masks bitwise; register redistribution gain, superseded"
    if version == "v197":
        status = "bitwise v184 full2seeds/short/masks; current higher precision"
    if version in ("v198", "v199"):
        status = "guarded checks/full seed1234/mask bits pass; native x64 no gain"
    if version in ("v200", "v201"):
        status = "guarded/full seed1234/mask checks pass; maximal role budget no default gain"
    if version == "v202":
        status = "full2seeds/short/masks bitwise; cache hint mixed warm gain; validated alternative"
    if version in ("v205", "v206"):
        status = "full2seeds/short/masks bitwise; reciprocal handoff alternative; timing regime matters"
    if version in ("v207", "v208"):
        status = "guarded/full seed1234/mask checks pass; packed adjacent nodes slower"
    if version in ("v209", "v210"):
        status = "guarded/full seed1234/mask checks pass; packed half-trees improve previous packing, still slower"
    if version in ("v213", "v214"):
        status = "guarded/full seed1234/mask checks pass; looped TMA coordinate reuse slower"
    if version in ("v215", "v216"):
        status = "full seed1234/masked FP32 and guarded checks pass; half residual conversion slower"
    if version in ("v219", "v220", "v221", "v222", "v223", "v224", "v225", "v226"):
        status = "guarded checks/full seed1234/mask parent equivalence; slower than defaults"
    if version == "v227":
        status = "guarded memory/sync pass;9 full seed1234 failures,66 mask failures; slower"
    if version == "v228":
        status = "guarded checks/full seed1234/mask exact v227; no performance gain"
    if version == "v229":
        status = "guarded checks/full seed1234 and masks FP32 PASS; slower; no other full audits"
    if version in ("v230", "v231"):
        status = "guarded checks/full seed1234/mask parent equivalence; KV-tail P reuse slower"
    if version == "v232":
        status = "guarded checks pass;9 full seed1234 failures,86 mask failures; single-buffer slower"
    if version in ("v240", "v241"):
        status = "guarded/full seed1234/short/mask exact v197; zero local traffic; unroll and role budgets slower"
    if version == "v235":
        status = "guarded/full seed1234/short/mask exact v190; fast precision limits; producer unroll no gain"
    if version == "v234":
        status = "guarded/full seed1234 exact v228;9 full and66 mask failures; faster than normal control,slower than default"
    if version == "v217":
        status = "guarded/full seed1234/mask equivalence pass; intermediate register budget no short gain"
    if version == "v142":
        status = "bitwise v138 full seed1234/mask; paired epilogue slower"
    if version == "v138":
        status = "bitwise v125 full2seeds/short/masks; earlier fast path"
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
print("\nRaw JSON records exact tensor shapes, seed, software versions, candidate SHA256 and unchanged benchmark SHA256. The baseline B0 used 20 warmups/100 repeats and measured 1860.70 µs warm / 1854.66 µs cold; use the paired baseline for each ratio because clocks vary. v090/v091/v094/v096/v097/v105/v108/v112/v125 have an observed sustained warm advantage in the extended eager-event, Graph and rotating-order runs, at baseline-level FP8 precision; v190 is the current fast path. Short five-event v125 tuning has a small observed advantage; v105 is near parity. Use the matching execution regime and retained distributions. v086 is near parity warm and its initial cold advantage does not reproduce in its rotating-order audit.\n")
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
    print("Ranges below are the minimum and maximum per-round medians or paired ratios, not confidence intervals. Historical audits here include nvidia-smi queries between cases; the endpoint control found roughly209ms idle gaps that change the operating regime. Future helper runs default to queries off. See ITERATIONS.md for the controls and original standalone timing.\n")
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
