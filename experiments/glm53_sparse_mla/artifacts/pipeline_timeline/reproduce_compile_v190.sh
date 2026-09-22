#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/pipeline_timeline_compile/v190
/opt/sglang/bin/python check_gpu_idle.py > artifacts/pipeline_timeline_compile/preflight.log 2>&1
PYTHONPATH=/tmp/glm53_sparse_mla_dev timeout --kill-after=10s 180s /opt/sglang/bin/python -u probe_pipeline_timeline.py --versions v190 --compile-only --output-dir artifacts/pipeline_timeline_compile/v190 > artifacts/pipeline_timeline_compile/v190/compile.log 2>&1
tail -4 artifacts/pipeline_timeline_compile/v190/compile.log
cd artifacts/pipeline_timeline_compile/v190
cubin=( *.cubin )
cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
cuobjdump --dump-sass "${cubin[0]}" > sass.txt
cat resources.txt
