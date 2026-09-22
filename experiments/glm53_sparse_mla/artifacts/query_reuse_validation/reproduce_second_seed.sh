#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_reuse_validation
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_reuse_validation/second_seed_preflight.log 2>&1
for version in v271 v272; do
 timeout --kill-after=10s 180s /opt/sglang/bin/python -u validate_equivalence.py --baseline v190 --candidate "$version" --block-k 128 --local-tokens 8192 --chunk 3 --seed 5678 --repeats 3 --output-json "artifacts/query_reuse_validation/${version}_second_seed.json" > "artifacts/query_reuse_validation/${version}_second_seed.log" 2>&1
 tail -3 "artifacts/query_reuse_validation/${version}_second_seed.log"
done

