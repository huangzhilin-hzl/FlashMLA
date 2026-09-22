set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v298_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v298 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v298_guarded_smoke.json > artifacts/v298_guarded_smoke.log 2>&1
tail -1 artifacts/v298_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v298 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v298_guarded_memcheck.json > artifacts/v298_guarded_memcheck.log 2>&1
tail -1 artifacts/v298_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v298 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v298_synccheck.json > artifacts/v298_synccheck.log 2>&1
tail -1 artifacts/v298_synccheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v298 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v298_synccheck_varlen.json > artifacts/v298_synccheck_varlen.log 2>&1
tail -1 artifacts/v298_synccheck_varlen.log

MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u validate_equivalence.py --baseline v284 --candidate v298 --block-k 128 --local-tokens 513 --chunk 0 --seed 5678 --repeats 3 --output-json artifacts/v298_odd_guarded_equivalence.json > artifacts/v298_odd_guarded_equivalence.log 2>&1
tail -4 artifacts/v298_odd_guarded_equivalence.log

/opt/sglang/bin/python -u validate_equivalence.py --baseline v284 --candidate v298 --block-k 128 --output-json artifacts/v298_equivalence.json > artifacts/v298_equivalence.log 2>&1
tail -3 artifacts/v298_equivalence.log
/opt/sglang/bin/python -u validate_equivalence.py --baseline v284 --candidate v298 --block-k 128 --local-tokens 1024 --chunk 0 --seed 5678 --output-json artifacts/v298_short_equivalence.json > artifacts/v298_short_equivalence.log 2>&1
tail -3 artifacts/v298_short_equivalence.log

set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v284 v298 --block-k 128 --output-json artifacts/v298_masks.json > artifacts/v298_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v298_masks.json'))
assert d['pairwise_bitwise_mismatches']['v284/v298']==0
print('v284/v298 masked exact match; inherited86 tolerance failures')
PY

