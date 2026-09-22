#!/usr/bin/env bash
# Guarded and uninstrumented diagnostic commands; physical GPU1 only.
# mixed_mma_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/mixed_mma_compile_preflight.log 2>&1
/opt/sglang/bin/python -u probe_tmem_mixed_mma.py --blocks 1 --compile-only --output-dir artifacts/mixed_mma_compile > artifacts/mixed_mma_compile.log 2>&1
cat artifacts/mixed_mma_compile/resources.txt


# mixed_mma_guard_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/mixed_mma_runtime_preflight.log 2>&1
for spec in memcheck:1 synccheck:1 memcheck:296 synccheck:296; do
 checktool="${spec%:*}";blocks="${spec#*:}"
 tag="mixed_mma_${checktool}_${blocks}"
 timeout --kill-after=10s 180s compute-sanitizer --tool "$checktool" --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_mixed_mma.py --blocks "$blocks" --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
 tail -2 "artifacts/$tag.log"
done


# mixed_mma_plain_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/mixed_mma_plain_compile_preflight.log 2>&1
/opt/sglang/bin/python -u validate_mixed_mma_uninstrumented.py --blocks 1 --compile-only --output-dir artifacts/mixed_mma_plain_compile > artifacts/mixed_mma_plain_compile.log 2>&1
cat artifacts/mixed_mma_plain_compile/resources.txt


# mixed_mma_plain_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/mixed_mma_plain_preflight.log 2>&1
for blocks in 1 296; do
 /opt/sglang/bin/python -u validate_mixed_mma_uninstrumented.py --blocks "$blocks" --output-dir "artifacts/mixed_mma_plain_$blocks" > "artifacts/mixed_mma_plain_$blocks.log" 2>&1
 tail -1 "artifacts/mixed_mma_plain_$blocks.log"
done

