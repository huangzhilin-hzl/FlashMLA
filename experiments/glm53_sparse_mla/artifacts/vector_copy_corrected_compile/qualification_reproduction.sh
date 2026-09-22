set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/vector_copy_qualification_preflight.log 2>&1
for version in v219 v220; do
  echo "$version guarded smoke"
  MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version "$version" --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json "artifacts/${version}_guarded_smoke.json" > "artifacts/${version}_guarded_smoke.log" 2>&1
  echo "$version guarded memcheck"
  MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version "$version" --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json "artifacts/${version}_guarded_memcheck.json" > "artifacts/${version}_guarded_memcheck.log" 2>&1
  echo "$version synccheck"
  timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version "$version" --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json "artifacts/${version}_synccheck.json" > "artifacts/${version}_synccheck.log" 2>&1
  parent=v190
  if [[ "$version" == v220 ]]; then parent=v197; fi
  echo "$version full1234 exact comparison"
  /opt/sglang/bin/python -u validate_equivalence.py --baseline "$parent" --candidate "$version" --output-json "artifacts/${version}_equivalence.json" > "artifacts/${version}_equivalence.log" 2>&1
done
echo "mask fixture"
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v190 v197 v219 v220 --output-json artifacts/vector_copy_masks.json > artifacts/vector_copy_masks.log 2>&1 || [[ $? == 1 ]]
