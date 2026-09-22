#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/pipeline_timeline_guarded
/opt/sglang/bin/python check_gpu_idle.py > artifacts/pipeline_timeline_guarded/preflight.log 2>&1
PYTHONPATH=/tmp/glm53_sparse_mla_dev MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s /opt/sglang/bin/python -u probe_pipeline_timeline.py --local-tokens 2 --repeat 1 --stride 1 --output-dir artifacts/pipeline_timeline_guarded/smoke > artifacts/pipeline_timeline_guarded/smoke.log 2>&1
tail -6 artifacts/pipeline_timeline_guarded/smoke.log
