set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
artifact_dir=artifacts/compiler471_timing
test ! -e "$artifact_dir"
mkdir -p "$artifact_dir"
/opt/sglang/bin/python check_gpu_idle.py > "$artifact_dir/preflight.log" 2>&1
for round in 0 1; do
  if [[ "$round" == 0 ]]; then compilers="462 471"; else compilers="471 462"; fi
  for version in v190 v197; do
    for compiler in $compilers; do
      echo "round=$round version=$version compiler=$compiler"
      runner=()
      if [[ "$compiler" == 471 ]]; then runner=(bash with_cutlass471.sh); fi
      "${runner[@]}" /opt/sglang/bin/python -u bench.py --backends trtllm cute --kernel-version "$version" --block-k 128 --scope native --check-rows 8 --warmup-iters 20 --repeat-iters 100 --cache both --timing cuda-graph --output-json "$artifact_dir/${version}_${compiler}_round${round}.json" > "$artifact_dir/${version}_${compiler}_round${round}.log" 2>&1
      tail -6 "$artifact_dir/${version}_${compiler}_round${round}.log"
    done
  done
done
mkdir -p artifacts/snapshots/compiler471_timing
cp -a "$artifact_dir"/. artifacts/snapshots/compiler471_timing/
