set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/v289_continuation_preflight.log 2>&1
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import json
for suffix in ['guarded_memcheck','synccheck','synccheck_varlen']:
 assert 'ERROR SUMMARY: 0 errors' in Path(f'artifacts/v289_{suffix}.log').read_text()
assert not Path('artifacts/v289_odd_guarded_equivalence.log').exists()
Path('artifacts/v289_qualification_interruption.json').write_text(json.dumps({
 'dispatch_exit_code':143,
 'completed':['guarded_smoke','guarded_memcheck','synccheck','synccheck_varlen'],
 'observed':'All sanitizer logs end with zero errors; odd-stage log absent; no v289/sanitizer process remains; selected GPU idle.',
 'cause':'Unresolved external termination of dispatch; no observed kernel/sanitizer failure.',
 'action':'Retain completed logs and continue from odd guarded equivalence after fresh idle preflight.'
},indent=2)+'\n')
PY
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u validate_equivalence.py --baseline v287 --candidate v289 --block-k 128 --local-tokens 513 --chunk 0 --seed 5678 --repeats 3 --output-json artifacts/v289_odd_guarded_equivalence.json > artifacts/v289_odd_guarded_equivalence.log 2>&1
tail -4 artifacts/v289_odd_guarded_equivalence.log

/opt/sglang/bin/python -u validate_equivalence.py --baseline v287 --candidate v289 --block-k 128 --output-json artifacts/v289_equivalence.json > artifacts/v289_equivalence.log 2>&1
tail -3 artifacts/v289_equivalence.log
/opt/sglang/bin/python -u validate_equivalence.py --baseline v287 --candidate v289 --block-k 128 --local-tokens 1024 --chunk 0 --seed 5678 --output-json artifacts/v289_short_equivalence.json > artifacts/v289_short_equivalence.log 2>&1
tail -3 artifacts/v289_short_equivalence.log

set +e
/opt/sglang/bin/python -u audit_mask_precision.py --kernel-versions v287 v289 --block-k 128 --output-json artifacts/v289_masks.json > artifacts/v289_masks.log 2>&1
audit_status=$?
set -e
test "$audit_status" -le 1
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/v289_masks.json'))
assert d['pairwise_bitwise_mismatches']['v287/v289']==0
assert d['cases']['v287']['pass'] and d['cases']['v289']['pass']
print('v287/v289 masked exact match and FP32 tolerance PASS')
PY

