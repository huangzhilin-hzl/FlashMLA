set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v229_replay_preflight.log 2>&1
mkdir -p artifacts/normal_pipeline_compile/v229_initial_replay
cp artifacts/normal_pipeline_compile/v229/initial_source.py artifacts/normal_pipeline_compile/v229_initial_replay/kernel_v229.py
cd artifacts/normal_pipeline_compile/v229_initial_replay
set +e
PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python -c 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); exec(compile(Path("/tmp/glm53_sparse_mla_dev/compile_offline.py").read_text(),"/tmp/glm53_sparse_mla_dev/compile_offline.py","exec"))' --initialize-cuda --kernel-version v229 --block-k 64 > replay_compile.log 2>&1
compile_status=$?
set -e
test "$compile_status" -ne 0
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
s=Path('replay_compile.log').read_text()
assert "profile of input tuples doesn't match: ((32, 1), 32)" in s
print('Initial shape failure reproduced from preserved source; replay log labelled explicitly.')
PY
