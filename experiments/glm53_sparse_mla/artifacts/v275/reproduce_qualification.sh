#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v275_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v275 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v275_guarded_smoke.json > artifacts/v275_guarded_smoke.log 2>&1
tail -1 artifacts/v275_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v275 --block-k 128 --local-tokens 1024 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v275_guarded_memcheck.json > artifacts/v275_guarded_memcheck.log 2>&1
tail -1 artifacts/v275_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v275 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v275_synccheck.json > artifacts/v275_synccheck.log 2>&1
tail -1 artifacts/v275_synccheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v275 --block-k 128 --local-tokens 1024 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v275_synccheck_varlen.json > artifacts/v275_synccheck_varlen.log 2>&1
tail -1 artifacts/v275_synccheck_varlen.log

MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u validate_equivalence.py --baseline v272 --candidate v275 --block-k 128 --local-tokens 1025 --chunk 0 --seed 5678 --repeats 3 --output-json artifacts/v275_odd_guarded_equivalence.json > artifacts/v275_odd_guarded_equivalence.log 2>&1
tail -4 artifacts/v275_odd_guarded_equivalence.log

/opt/sglang/bin/python -u validate_equivalence.py --baseline v272 --candidate v275 --block-k 128 --output-json artifacts/v275_equivalence.json > artifacts/v275_equivalence.log 2>&1
tail -3 artifacts/v275_equivalence.log
/opt/sglang/bin/python -u validate_equivalence.py --baseline v272 --candidate v275 --block-k 128 --local-tokens 1024 --chunk 0 --seed 5678 --output-json artifacts/v275_short_equivalence.json > artifacts/v275_short_equivalence.log 2>&1
tail -3 artifacts/v275_short_equivalence.log

set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v272 v275 --block-k 128 --output-json artifacts/v275_masks.json > artifacts/v275_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v275_masks.json'))
assert d['pairwise_bitwise_mismatches']['v272/v275']==0
print('v272/v275 masked exact match; inherited86 tolerance failures')
PY

