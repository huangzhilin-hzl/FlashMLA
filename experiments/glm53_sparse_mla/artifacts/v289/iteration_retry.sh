set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
cp -n artifacts/v289_gpu_preflight.log artifacts/v289_gpu_preflight_busy.log
bash run_iteration.sh v289 128
