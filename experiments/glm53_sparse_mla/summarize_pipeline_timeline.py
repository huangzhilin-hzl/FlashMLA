"""Summarize diagnostic CTA timestamps without inferring production latency."""
import argparse
import json
from pathlib import Path
import statistics


def describe(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "min_us": ordered[0], "median_us": statistics.median(ordered),
            "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * .95))],
            "max_us": ordered[-1], "mean_us": statistics.mean(ordered)}


def summarize(data):
    events = data["events"]
    intervals = {
        "compute_full_wait": ("compute_full_wait", "compute_full_ready"),
        "compute_qk_wait": ("compute_qk_wait", "compute_qk_ready"),
        "compute_pv_wait": ("compute_pv_wait", "compute_pv_ready"),
        "score_load_observation": ("compute_qk_ready", "scores_loaded"),
        "softmax_high": ("scores_loaded", "probability_high_ready"),
        "probability_tail": ("probability_high_ready", "probability_published"),
        "issuer_p_wait": ("issuer_p_wait", "issuer_p_ready"),
        "qk_submission": ("qk_issue", "qk_commit"),
        "pv_submission": ("pv_issue", "pv_commit"),
        "producer_empty_wait": ("producer_empty_wait", "producer_empty_ready"),
        "producer_after_empty": ("producer_empty_ready", "producer_gathers_issued"),
    }
    mandatory = set(events) - {"producer_empty_wait", "producer_empty_ready"}
    summary = {"notice": "Instrumented observations, not production timings or instruction latencies.",
               "versions": {}}
    for version, entry in data["versions"].items():
        measures = {key: [] for key in intervals}
        measures.update(cta_observed_span=[], probability_tail_without_prior_pv_wait=[],
                        qk_issue_relative_previous_pv=[])
        early = total = 0
        per_cta = []
        for run in entry["runs"]:
            assert run["finite"] and run["bitwise_mismatches"] == 0
            for index, trace in enumerate(run["timestamps_ns"]):
                active = [i for i, row in enumerate(trace) if row[events["qk_issue"]]]
                assert active == list(range(len(active))), (version, index, "noncontiguous tiles")
                if not active:
                    continue
                positive = [x for tile in active for x in trace[tile] if x]
                span = (max(positive) - min(positive)) / 1000
                measures["cta_observed_span"].append(span)
                sums = {key: 0.0 for key in intervals}
                for tile in active:
                    row = trace[tile]
                    assert all(row[events[key]] for key in mandatory), (version, index, tile, "missing event")
                    assert tile < 2 or (row[events["producer_empty_wait"]] and row[events["producer_empty_ready"]])
                    for key, (begin, end) in intervals.items():
                        a, b = row[events[begin]], row[events[end]]
                        if a and b:
                            assert b >= a, (version, index, tile, key, a, b)
                            value = (b - a) / 1000
                            measures[key].append(value)
                            sums[key] += value
                    tail = row[events["probability_published"]] - row[events["probability_high_ready"]]
                    if version in {"v252", "v257"} and tile:
                        previous = trace[tile - 1]
                        tail -= previous[events["compute_pv_ready"]] - previous[events["compute_pv_wait"]]
                    assert tail >= 0, (version, index, tile, "negative adjusted tail")
                    measures["probability_tail_without_prior_pv_wait"].append(tail / 1000)
                    if tile:
                        difference = (row[events["qk_issue"]] - trace[tile - 1][events["pv_issue"]]) / 1000
                        measures["qk_issue_relative_previous_pv"].append(difference)
                        early += difference < 0
                        total += 1
                per_cta.append({"repeat": run["repeat"], "query_row": index * data["stride"],
                                "tiles": len(active), "observed_span_us": span,
                                "summed_intervals_us": sums})
        summary["versions"][version] = {
            "intervals": {key: describe(values) for key, values in measures.items()},
            "nonbootstrap_qk_early_count": early, "nonbootstrap_qk_count": total,
            "sampled_early_fraction": early / total if total else None,
            "per_cta": per_cta,
        }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for version, entry in result["versions"].items():
        print(version, "early_fraction", entry["sampled_early_fraction"])
        for name, values in entry["intervals"].items():
            print(" ", name, values)
