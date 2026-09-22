#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v257_compile_preflight.log 2>&1
mkdir -p artifacts/strict_dual_score_compile/v257
cd artifacts/strict_dual_score_compile/v257
PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python /tmp/glm53_sparse_mla_dev/compile_offline.py --initialize-cuda --kernel-version v257 --block-k 128 > compile.log 2>&1
cubin=( *.cubin )
cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
cuobjdump --dump-sass "${cubin[0]}" > sass.txt
cat resources.txt
