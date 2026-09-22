#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,re,json
s=Path('artifacts/snapshots/delayed_pv_wait_source');assert not s.exists();s.mkdir(parents=True)
for name in ('v251_source.csv','v251_source_summary.json','v251_source.log','v251_source_preflight.log'):shutil.copy2(Path('artifacts')/name,s/name)
c=Path('artifacts/dual_score_arbitration_compile/v251')
t=(c/'sass.txt').read_text()
d={'static_native_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',t))//2,'LDL':len(re.findall(r'\bLDL\b',t)),'STL':len(re.findall(r'\bSTL\b',t))}
(c/'resource_summary.json').write_text(json.dumps(d,indent=2)+'\n')
snap=Path('artifacts/snapshots/delayed_pv_wait_compile');assert not snap.exists();(snap/'v251').mkdir(parents=True)
for p in c.iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v251'/p.name)
shutil.copy2('kernel_v251.py',snap/'v251'/'kernel_v251.py')
shutil.copy2('artifacts/v251_compile_preflight.log',snap/'v251_compile_preflight.log')
p=Path('artifacts/snapshots/v251/v251_ncu.ncu-rep')
if p.exists():p.unlink()
print(json.dumps(d))
PY
