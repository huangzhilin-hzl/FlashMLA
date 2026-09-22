#!/usr/bin/env bash
# Historical commands run inside the authorized pod,physical GPU1 only.
# v232_guard_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import hashlib,re
root=Path('artifacts/normal128_compile/v232')
ptx=next(root.glob('*.ptx')).read_text()
assert re.search(r'tcgen05.ld.red[^;]+, 64;',ptx), 'missing explicit x64 reduction half offset'
assert '.minnctapersm 2' in ptx
assert re.search(r'setmaxnreg.dec.sync.aligned.u32\s+32;', ptx)
assert re.search(r'setmaxnreg.inc.sync.aligned.u32\s+224;', ptx)
log=(root/'compile.log').read_text()
sha=hashlib.sha256(Path('kernel_v232.py').read_bytes()).hexdigest()
assert sha in log
sass=(root/'sass.txt').read_text()
print('v232 sha',sha,'static instructions',len(re.findall(r'0x[0-9a-fA-F]{16}',sass))//2)
print('compile ownership/offset assertions PASS')
PY
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v232_memory_preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 90s /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v232 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v232_guarded_smoke.json > artifacts/v232_guarded_smoke.log 2>&1
tail -1 artifacts/v232_guarded_smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v232 --block-k 128 --local-tokens 512 --chunk 0 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 1 --cache warm --output-json artifacts/v232_guarded_memcheck.json > artifacts/v232_guarded_memcheck.log 2>&1
tail -1 artifacts/v232_guarded_memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u bench.py --backends cute --kernel-version v232 --block-k 128 --local-tokens 2 --scope native --check-rows 2 --warmup-iters 1 --repeat-iters 3 --cache warm --output-json artifacts/v232_synccheck.json > artifacts/v232_synccheck.log 2>&1
tail -1 artifacts/v232_synccheck.log


# v232_accuracy_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v232_accuracy_preflight.log 2>&1
set +e
/opt/sglang/bin/python -u validate_full_accuracy.py --kernel-versions v232 --block-k 128 --include-trtllm --seed 1234 --output-json artifacts/v232_full_accuracy.json > artifacts/v232_full_accuracy.log 2>&1
full_status=$?
set -e
test "$full_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v232_full_accuracy.json'))
assert all(c['checked_elements']==268435456 and c['finite'] for c in d['cases'].values())
for n,c in d['cases'].items(): print(n,{k:c[k] for k in ['mismatches','max_abs','relative_rmse']})
PY
set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v232 --block-k 128 --output-json artifacts/v232_masks.json > artifacts/v232_masks.log 2>&1
mask_status=$?
set -e
test "$mask_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v232_masks.json'))
assert d['cases']['v232']['checked_elements']==65536
for n,c in d['cases'].items(): print(n,{k:c[k] for k in ['mismatches','max_abs','relative_rmse']})
PY


# v232_iteration_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
bash run_iteration.sh v232 128

