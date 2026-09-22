set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v227_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v227 --block-k 64 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v227_guarded_memcheck.json > artifacts/v227_guarded_memcheck.log 2>&1
tail -1 artifacts/v227_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v227 --block-k 64 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v227_synccheck.json > artifacts/v227_synccheck.log 2>&1
tail -1 artifacts/v227_synccheck.log
set +e
/opt/sglang/bin/python -u validate_full_accuracy.py --kernel-versions v227 --block-k 64 --include-trtllm --seed 1234 --output-json artifacts/v227_full_accuracy_seed1234.json > artifacts/v227_full_accuracy_seed1234.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v227_full_accuracy_seed1234.json'))
for k,v in d['cases'].items():
 assert v['finite']
 print(k,{x:v[x] for x in ['mismatches','max_abs','relative_rmse','checked_elements','pass']})
PY
set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v227 --block-k 64 --output-json artifacts/v227_masks.json > artifacts/v227_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v227_masks.json'))
for k,v in d['cases'].items():print(k,{x:v[x] for x in ['mismatches','max_abs','relative_rmse','pass']})
PY
