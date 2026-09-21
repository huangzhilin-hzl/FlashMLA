#!/usr/bin/env bash
# Run inside the authorized pod. GPU1 is selected by its physical UUID.
set -euo pipefail
cd "$(dirname "$0")"
version="${1:?usage: bash run_iteration.sh vNNN}"
[[ "$version" =~ ^v[0-9]{3}$ ]] || exit 2
block_k="${2:-64}"
[[ "$block_k" =~ ^(64|128|256)$ ]] || exit 2
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
python_bin=/opt/sglang/bin/python
mkdir -p artifacts
"$python_bin" check_gpu_idle.py > "artifacts/${version}_gpu_preflight.log" 2>&1 || {
  cat "artifacts/${version}_gpu_preflight.log"
  exit 3
}
"$python_bin" -u bench.py --backends trtllm cute --kernel-version "$version" --block-k "$block_k" \
  --scope native --check-rows 8 --warmup-iters 3 --repeat-iters 5 --cache warm \
  --output-json "artifacts/${version}_full_event.json" \
  > "artifacts/${version}_full_event.log" 2>&1
tail -14 "artifacts/${version}_full_event.log"
ncu --profile-from-start off --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section Occupancy --section SchedulerStats \
  --section WarpStateStats \
  --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --force-overwrite -o "artifacts/${version}_ncu" \
  "$python_bin" profile_ncu.py --backend cute --kernel-version "$version" --block-k "$block_k" \
  > "artifacts/${version}_ncu.log" 2>&1
ncu --import "artifacts/${version}_ncu.ncu-rep" --page details \
  > "artifacts/${version}_ncu_details.txt"
ncu --import "artifacts/${version}_ncu.ncu-rep" --page raw --csv \
  > "artifacts/${version}_ncu_raw.csv"
"$python_bin" extract_ncu.py "artifacts/${version}_ncu_raw.csv" \
  > "artifacts/${version}_ncu_summary.json"
mkdir -p "artifacts/snapshots/${version}"
cp artifacts/"${version}"_* "artifacts/snapshots/${version}/"
cat "artifacts/${version}_ncu_summary.json"
