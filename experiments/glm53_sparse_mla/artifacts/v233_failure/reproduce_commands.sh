#!/usr/bin/env bash
# Historical diagnostic commands; failed numerical runs are retained,not qualified.
# v233_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v233_compile_preflight.log 2>&1
mkdir -p artifacts/normal128_collector_compile/v233
cd artifacts/normal128_collector_compile/v233
PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python /tmp/glm53_sparse_mla_dev/compile_offline.py --initialize-cuda --kernel-version v233 --block-k 128 > compile.log 2>&1
cubin=( *.cubin )
cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
cuobjdump --dump-sass "${cubin[0]}" > sass.txt
cat resources.txt


# v233_qualification_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v233_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v233 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v233_guarded_smoke.json > artifacts/v233_guarded_smoke.log 2>&1
tail -1 artifacts/v233_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v233 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v233_guarded_memcheck.json > artifacts/v233_guarded_memcheck.log 2>&1
tail -1 artifacts/v233_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v233 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v233_synccheck.json > artifacts/v233_synccheck.log 2>&1
tail -1 artifacts/v233_synccheck.log

/opt/sglang/bin/python -u validate_equivalence.py --baseline v232 --candidate v233 --block-k 128 --output-json artifacts/v233_equivalence.json > artifacts/v233_equivalence.log 2>&1
tail -3 artifacts/v233_equivalence.log
set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v232 v233 --block-k 128 --output-json artifacts/v233_masks.json > artifacts/v233_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v233_masks.json'))
assert d['pairwise_bitwise_mismatches']['v232/v233']==0
print('v232/v233 masked exact match; inherited86 tolerance failures')
PY


# v233_diagnosis_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v233_diagnosis_preflight.log 2>&1
timeout --kill-after=10s 180s /opt/sglang/bin/python -u diagnose_normal_collector.py --sizes 2 148 296 --output-json artifacts/v233_grid_diagnosis.json > artifacts/v233_grid_diagnosis.log 2>&1
cat artifacts/v233_grid_diagnosis.log

