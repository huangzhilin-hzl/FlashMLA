#!/usr/bin/env bash
# Historical commands run inside the authorized pod; CUDA jobs use physical GPU1.
# half_plane_compile_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
mkdir -p artifacts/half_plane_audit
CUDA_VISIBLE_DEVICES="" CUTE_DSL_ARCH=sm_103a /opt/sglang/bin/python audit_tmem_normal_layout.py --columns 128 > artifacts/half_plane_audit/mapping.json 2> artifacts/half_plane_audit/mapping.log
/opt/sglang/bin/python - <<'PY'
import json
d=json.load(open('artifacts/half_plane_audit/mapping.json'))
print({k:v for k,v in d.items() if k!='mapping'})
PY
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/half_plane_compile_preflight.log 2>&1
timeout --kill-after=10s 180s /opt/sglang/bin/python -u probe_tmem_half_plane.py --compile-only --blocks 1 --output-dir artifacts/half_plane_compile > artifacts/half_plane_compile.log 2>&1
cat artifacts/half_plane_compile/resources.txt
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re
s=next(Path('artifacts/half_plane_compile').glob('*.ptx')).read_text()
m=re.findall(r'tcgen05.ld.red[^;]+;',s)
assert len(m)==1 and m[0].endswith(', 64;'),m
print('LD.RED x64 half split64 verified')
PY


# half_plane_memcheck_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/half_plane_memcheck_preflight.log 2>&1
timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_half_plane.py --blocks 1 --output-dir artifacts/half_plane_memcheck > artifacts/half_plane_memcheck.log 2>&1
tail -3 artifacts/half_plane_memcheck.log


# half_plane_remaining_command
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
for spec in synccheck:1 memcheck:296 synccheck:296; do
 checktool="${spec%:*}"; blocks="${spec#*:}"
 tag="half_plane_${checktool}_${blocks}"
 /opt/sglang/bin/python check_gpu_idle.py > "artifacts/${tag}_preflight.log" 2>&1
 timeout --kill-after=10s 180s compute-sanitizer --tool "$checktool" --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_half_plane.py --blocks "$blocks" --output-dir "artifacts/$tag" > "artifacts/$tag.log" 2>&1
 tail -2 "artifacts/$tag.log"
done

