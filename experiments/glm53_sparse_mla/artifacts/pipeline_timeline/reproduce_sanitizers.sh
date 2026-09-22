#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
export PYTHONPATH=/tmp/glm53_sparse_mla_dev
export MLA_TMEM_GUARDRAILS=1
timeout --kill-after=10s 240s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_pipeline_timeline.py --local-tokens 512 --chunk 0 --repeat 1 --stride 128 --output-dir artifacts/pipeline_timeline_guarded/memcheck > artifacts/pipeline_timeline_guarded/memcheck.log 2>&1
tail -2 artifacts/pipeline_timeline_guarded/memcheck.log
timeout --kill-after=10s 240s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_pipeline_timeline.py --local-tokens 2 --repeat 1 --stride 1 --output-dir artifacts/pipeline_timeline_guarded/synccheck > artifacts/pipeline_timeline_guarded/synccheck.log 2>&1
tail -2 artifacts/pipeline_timeline_guarded/synccheck.log
timeout --kill-after=10s 240s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_pipeline_timeline.py --local-tokens 512 --chunk 0 --repeat 1 --stride 128 --output-dir artifacts/pipeline_timeline_guarded/synccheck_varlen > artifacts/pipeline_timeline_guarded/synccheck_varlen.log 2>&1
tail -2 artifacts/pipeline_timeline_guarded/synccheck_varlen.log
