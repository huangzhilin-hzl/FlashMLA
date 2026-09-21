#!/usr/bin/env bash
# GPU1 only. Compare profiler controls without overwriting default NCU evidence.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
for regime in none_stable base_dynamic none_dynamic; do
  clock_mode=${regime%_*}
  boost_mode=${regime#*_}
  for version in v086 v088; do
    stem="artifacts/${version}_${regime}"
    /opt/sglang/bin/python check_gpu_idle.py > "${stem}_preflight.log"
    ncu --clock-control "$clock_mode" --pipeline-boost-state "$boost_mode" \
      --profile-from-start off --section SpeedOfLight --section SchedulerStats \
      --section WarpStateStats --force-overwrite -o "$stem" \
      /opt/sglang/bin/python profile_ncu.py --backend cute \
      --kernel-version "$version" --block-k 128 > "${stem}.log" 2>&1
    ncu --import "${stem}.ncu-rep" --page raw --csv > "${stem}.csv"
    ncu --import "${stem}.ncu-rep" --page details > "${stem}_details.txt"
    echo "$version $regime profile saved"
  done
done
mkdir -p artifacts/snapshots/profiling_regimes_086_088
cp artifacts/v08[68]_none_* artifacts/v08[68]_base_dynamic* \
  artifacts/snapshots/profiling_regimes_086_088/
