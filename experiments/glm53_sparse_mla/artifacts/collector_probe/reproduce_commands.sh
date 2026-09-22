#!/usr/bin/env bash
# Historical diagnostic commands; failed numerical runs are retained,not qualified.
# collector_probe_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/collector_probe_compile_preflight.log 2>&1
for mode in discard same cross; do
 for guard in yes no; do
  tag="collector_probe_compile_${mode}_${guard}"
  options=()
  if [[ "$guard" == yes ]]; then options+=(--guardrails); fi
  /opt/sglang/bin/python -u probe_normal_collector.py --blocks 1 --mode "$mode" "${options[@]}" --compile-only --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
  cubin=( artifacts/"$tag"/*.cubin )
  cuobjdump --dump-resource-usage "${cubin[0]}" > "artifacts/$tag/resources.txt"
  cuobjdump --dump-sass "${cubin[0]}" > "artifacts/$tag/sass.txt"
  printf '%s\n' "$tag"
  cat "artifacts/$tag/resources.txt"
 done
done


# collector_probe_fixed_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/collector_probe_compile_fixed_preflight.log 2>&1
for mode in discard same cross; do
 for guard in yes no; do
  tag="collector_probe_compile_fixed_${mode}_${guard}"
  options=()
  if [[ "$guard" == yes ]]; then options+=(--guardrails); fi
  /opt/sglang/bin/python -u probe_normal_collector.py --blocks 1 --mode "$mode" "${options[@]}" --compile-only --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
  cubin=( artifacts/"$tag"/*.cubin )
  cuobjdump --dump-resource-usage "${cubin[0]}" > "artifacts/$tag/resources.txt"
  cuobjdump --dump-sass "${cubin[0]}" > "artifacts/$tag/sass.txt"
  printf '%s\n' "$tag"
  cat "artifacts/$tag/resources.txt"
 done
done


# collector_probe_runtime_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib
print('probe sha',hashlib.sha256(Path('probe_normal_collector.py').read_bytes()).hexdigest())
for mode in ['discard','same','cross']:
 for guard in ['yes','no']:
  p=Path(f'artifacts/collector_probe_compile_fixed_{mode}_{guard}')
  s=(p/'sass.txt').read_text()
  print(mode,guard,'A_KEEP',s.count('.A_KEEP'),'A_REUSE',s.count('.A_REUSE'),'UTCCHECK',s.count('UTCQMMA'))
PY
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/collector_probe_runtime_preflight.log 2>&1
for mode in discard same cross; do
 tag="collector_probe_memcheck_${mode}"
 timeout --kill-after=10s 120s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_normal_collector.py --blocks 1 --mode "$mode" --guardrails --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
 tail -5 "artifacts/$tag.log"
done
for blocks in 1 296; do
 for mode in discard same cross; do
  tag="collector_probe_plain_${mode}_${blocks}"
  timeout --kill-after=10s 90s /opt/sglang/bin/python -u probe_normal_collector.py --blocks "$blocks" --mode "$mode" --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
  cat "artifacts/$tag.log"
 done
done

