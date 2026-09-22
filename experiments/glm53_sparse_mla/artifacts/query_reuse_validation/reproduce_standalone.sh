#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_reuse_validation
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_reuse_validation/standalone_preflight.log 2>&1
for timing in cuda-event cuda-graph; do
 for version in v190 v272; do
  timeout --kill-after=10s 240s /opt/sglang/bin/python -u bench.py --backends trtllm cute --kernel-version "$version" --block-k 128 --scope native --check-rows 512 --warmup-iters 20 --repeat-iters 100 --timing "$timing" --cache both --output-json "artifacts/query_reuse_validation/${version}_${timing}_event100.json" > "artifacts/query_reuse_validation/${version}_${timing}_event100.log" 2>&1
  tail -7 "artifacts/query_reuse_validation/${version}_${timing}_event100.log"
 done
done

