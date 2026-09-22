#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v257_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v257 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v257_guarded_smoke.json > artifacts/v257_guarded_smoke.log 2>&1
tail -1 artifacts/v257_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v257 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v257_guarded_memcheck.json > artifacts/v257_guarded_memcheck.log 2>&1
tail -1 artifacts/v257_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v257 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v257_synccheck.json > artifacts/v257_synccheck.log 2>&1
tail -1 artifacts/v257_synccheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v257 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v257_synccheck_varlen.json > artifacts/v257_synccheck_varlen.log 2>&1
tail -1 artifacts/v257_synccheck_varlen.log

/opt/sglang/bin/python -u validate_equivalence.py --baseline v243 --candidate v257 --block-k 128 --output-json artifacts/v257_equivalence.json > artifacts/v257_equivalence.log 2>&1
tail -3 artifacts/v257_equivalence.log
/opt/sglang/bin/python -u validate_equivalence.py --baseline v243 --candidate v257 --block-k 128 --local-tokens 1024 --chunk 0 --seed 5678 --output-json artifacts/v257_short_equivalence.json > artifacts/v257_short_equivalence.log 2>&1
tail -3 artifacts/v257_short_equivalence.log

set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v243 v257 --block-k 128 --output-json artifacts/v257_masks.json > artifacts/v257_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v257_masks.json'))
assert d['pairwise_bitwise_mismatches']['v243/v257']==0
assert d['cases']['v243']['pass'] and d['cases']['v257']['pass']
print('v243/v257 masked exact match and FP32 tolerance PASS')
PY
