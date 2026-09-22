#!/usr/bin/env bash
# Historical commands run inside the authorized pod,physical GPU1 only.
# v232_source_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v232_source_preflight.log 2>&1
ncu --clock-control none --pipeline-boost-state dynamic --target-processes all --profile-from-start off --section SourceCounters --section WarpStateStats --force-overwrite -o artifacts/v232_source /opt/sglang/bin/python profile_ncu.py --backend cute --kernel-version v232 --block-k 128 > artifacts/v232_source.log 2>&1
ncu --import artifacts/v232_source.ncu-rep --page source --print-source sass --csv > artifacts/v232_source.csv
/opt/sglang/bin/python summarize_source_sass.py artifacts/v232_source.csv artifacts/v232_source_summary.json
cat artifacts/v232_source_summary.json

