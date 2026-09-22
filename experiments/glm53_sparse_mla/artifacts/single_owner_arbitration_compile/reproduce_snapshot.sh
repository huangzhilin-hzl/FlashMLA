#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import shutil,re,json
s=Path('artifacts/snapshots/single_owner_arbitration_source');assert not s.exists();s.mkdir(parents=True)
for name in ('v252_source.csv','v252_source_summary.json','v252_source.log','v252_source_preflight.log'):shutil.copy2(Path('artifacts')/name,s/name)
c=Path('artifacts/dual_score_arbitration_compile/v252')
t=(c/'sass.txt').read_text()
d={'static_native_instructions':len(re.findall(r'/\* 0x[0-9a-f]{16} \*/',t))//2,'LDL':len(re.findall(r'\bLDL\b',t)),'STL':len(re.findall(r'\bSTL\b',t))}
(c/'resource_summary.json').write_text(json.dumps(d,indent=2)+'\n')
snap=Path('artifacts/snapshots/single_owner_arbitration_compile');assert not snap.exists();(snap/'v252').mkdir(parents=True)
for p in c.iterdir():
 if p.suffix in ('.ptx','.txt','.json','.log'):shutil.copy2(p,snap/'v252'/p.name)
shutil.copy2('kernel_v252.py',snap/'v252'/'kernel_v252.py')
shutil.copy2('artifacts/v252_compile_preflight.log',snap/'v252_compile_preflight.log')
p=Path('artifacts/snapshots/v252/v252_ncu.ncu-rep')
if p.exists():p.unlink()
print(json.dumps(d))
PY
