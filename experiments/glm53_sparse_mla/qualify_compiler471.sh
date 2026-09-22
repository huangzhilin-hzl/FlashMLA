#!/usr/bin/env bash
# Qualify unchanged kernels with a process-local compiler; GPU1 only.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
python_bin=/opt/sglang/bin/python
artifact_dir=artifacts/compiler471_runtime
[[ ! -e "$artifact_dir" ]] || { echo "Use a fresh artifact directory" >&2; exit 2; }
mkdir -p "$artifact_dir"
trap 'mkdir -p artifacts/snapshots/compiler471_runtime; cp -a "$artifact_dir"/. artifacts/snapshots/compiler471_runtime/' EXIT
"$python_bin" check_gpu_idle.py > "$artifact_dir/preflight.log" 2>&1
for version in v190 v197; do
  echo "$version: guarded smoke"
  MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 120s bash with_cutlass471.sh \
    "$python_bin" -u bench.py --backends cute --kernel-version "$version" \
    --block-k 128 --local-tokens 2 --scope native --check-rows 2 \
    --warmup-iters 1 --repeat-iters 1 --cache warm \
    --output-json "$artifact_dir/${version}_471_guarded_smoke.json" \
    > "$artifact_dir/${version}_471_guarded_smoke.log" 2>&1
  echo "$version: guarded memcheck"
  MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s bash with_cutlass471.sh \
    compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 \
    "$python_bin" -u bench.py --backends cute --kernel-version "$version" \
    --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 \
    --warmup-iters 1 --repeat-iters 1 --cache warm \
    --output-json "$artifact_dir/${version}_471_guarded_memcheck.json" \
    > "$artifact_dir/${version}_471_guarded_memcheck.log" 2>&1
  echo "$version: synccheck"
  timeout --kill-after=10s 180s bash with_cutlass471.sh \
    compute-sanitizer --tool synccheck --error-exitcode 86 \
    "$python_bin" -u bench.py --backends cute --kernel-version "$version" \
    --block-k 128 --local-tokens 2 --scope native --check-rows 2 \
    --warmup-iters 1 --repeat-iters 3 --cache warm \
    --output-json "$artifact_dir/${version}_471_synccheck.json" \
    > "$artifact_dir/${version}_471_synccheck.log" 2>&1
done
echo "4.6.2: full input/output hashes"
"$python_bin" -u validate_output_hashes.py --kernel-versions v190 v197 \
  --output-json "$artifact_dir/kernels_462_hashes.json" > "$artifact_dir/kernels_462_hashes.log" 2>&1
echo "4.7.1: full input/output hashes"
bash with_cutlass471.sh "$python_bin" -u validate_output_hashes.py --kernel-versions v190 v197 \
  --output-json "$artifact_dir/kernels_471_hashes.json" > "$artifact_dir/kernels_471_hashes.log" 2>&1
"$python_bin" - "$artifact_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
a, b = [json.loads((root / f"kernels_{v}_hashes.json").read_text()) for v in (462, 471)]
assert a['cutlass_dsl'] == '4.6.2' and b['cutlass_dsl'] == '4.7.1'
assert a['benchmark_sha256'] == b['benchmark_sha256']
assert a['kernel_sha256'] == b['kernel_sha256']
assert len(a['records']) == len(b['records']) == 24
report = {'records': len(a['records']), 'records_equal': a['records'] == b['records'],
          'purpose': 'All input/output bytes across compiler processes; inherits baseline numerical limitations.'}
(root / 'equivalence_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
assert report['records_equal']
PY
