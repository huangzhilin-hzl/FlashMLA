#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v249_source_preflight.log 2>&1
ncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --section SourceCounters --section WarpStateStats --force-overwrite -o artifacts/v249_source /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version v249 --block-k 128 > artifacts/v249_source.log 2>&1
ncu --import artifacts/v249_source.ncu-rep --page source --print-source sass --csv > artifacts/v249_source.csv
/opt/sglang/bin/python summarize_source_sass.py artifacts/v249_source.csv artifacts/v249_source_summary.json

