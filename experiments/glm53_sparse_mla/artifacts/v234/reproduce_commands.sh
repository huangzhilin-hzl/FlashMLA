#!/usr/bin/env bash
# Historical commands onthe authorized physical GPU1.
# v234_guard_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,re
root=Path('artifacts/mixed_pipeline_compile/v234')
ptx=next(root.glob('*.ptx')).read_text()
assert ptx.count('tcgen05.mma.ws.cta_group')==18
assert ptx.count('tcgen05.mma.cta_group')==4
assert '.collector::' not in ptx
assert '.minnctapersm 2' in ptx
assert re.search(r'setmaxnreg.dec.sync.aligned.u32\s+32;',ptx)
assert re.search(r'setmaxnreg.inc.sync.aligned.u32\s+224;',ptx)
sha=hashlib.sha256(Path('kernel_v234.py').read_bytes()).hexdigest()
assert sha in (root/'compile.log').read_text()
sass=(root/'sass.txt').read_text()
print('v234 sha',sha,'static instructions',len(re.findall(r'0x[0-9a-fA-F]{16}',sass))//2)
print('MMA mode andresource assertions PASS')
PY
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v234_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v234 --block-k 64 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v234_guarded_smoke.json > artifacts/v234_guarded_smoke.log 2>&1
tail -1 artifacts/v234_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v234 --block-k 64 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v234_guarded_memcheck.json > artifacts/v234_guarded_memcheck.log 2>&1
tail -1 artifacts/v234_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v234 --block-k 64 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v234_synccheck.json > artifacts/v234_synccheck.log 2>&1
tail -1 artifacts/v234_synccheck.log


# v234_accuracy_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v234_accuracy_preflight.log 2>&1
set +e
/opt/sglang/bin/python -u validate_full_accuracy.py --kernel-versions v234 --block-k 64 --include-trtllm --seed 1234 --output-json artifacts/v234_full_accuracy.json > artifacts/v234_full_accuracy.log 2>&1
full_status=$?
set -e
test "$full_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v234_full_accuracy.json'))
assert all(c['checked_elements']==268435456 and c['finite'] for c in d['cases'].values())
for n,c in d['cases'].items(): print(n,{k:c[k] for k in ['mismatches','max_abs','relative_rmse']})
PY
set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v234 --block-k 64 --output-json artifacts/v234_masks.json > artifacts/v234_masks.log 2>&1
mask_status=$?
set -e
test "$mask_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v234_masks.json'))
assert d['cases']['v234']['checked_elements']==65536
for n,c in d['cases'].items(): print(n,{k:c[k] for k in ['mismatches','max_abs','relative_rmse']})
PY


# v234_equivalence_iteration_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v234_equivalence_preflight.log 2>&1
/opt/sglang/bin/python -u validate_equivalence.py --baseline v228 --candidate v234 --block-k 64 --output-json artifacts/v234_equivalence.json > artifacts/v234_equivalence.log 2>&1
tail -3 artifacts/v234_equivalence.log
bash run_iteration.sh v234 64

