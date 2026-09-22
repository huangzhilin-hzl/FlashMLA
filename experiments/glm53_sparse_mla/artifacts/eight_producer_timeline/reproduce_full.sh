#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
export PYTHONPATH=/tmp/glm53_sparse_mla_dev
unset MLA_TMEM_GUARDRAILS
mkdir -p artifacts/eight_producer_timeline_full
/opt/sglang/bin/python check_gpu_idle.py > artifacts/eight_producer_timeline_full/preflight.log 2>&1
timeout --kill-after=10s 300s /opt/sglang/bin/python -u probe_pipeline_timeline.py --versions v190 v266 v197 v265 --local-tokens 8192 --stride 1024 --repeat 3 --output-dir artifacts/eight_producer_timeline_full > artifacts/eight_producer_timeline_full/run.log 2>&1
tail -14 artifacts/eight_producer_timeline_full/run.log

/opt/sglang/bin/python summarize_pipeline_timeline.py artifacts/eight_producer_timeline_full/timeline.json --output artifacts/eight_producer_timeline_full/summary.json > artifacts/eight_producer_timeline_full/summary.txt
cat artifacts/eight_producer_timeline_full/summary.txt

