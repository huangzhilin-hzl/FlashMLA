#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import re,json,shutil
Path('artifacts/eight_producer_compile/initial_control_note.json').write_text(json.dumps({'initial_assertion': 'expected maxntid 640', 'observed': 'reqntid 640, 1, 1', 'resolution': 'check the emitted required thread count', 'candidate_launched_before_correction': False}, indent=2)+'\n')
records={}
for version,donor,compute,mmas in [('v265',48,168,34),('v266',32,192,26)]:
 p=Path('artifacts/eight_producer_compile')/version
 ptx=next(p.glob('*.ptx')).read_text()
 sass=(p/'sass.txt').read_text()
 res=(p/'resources.txt').read_text()
 assert re.search(r'REG:96\s+STACK:0',res)
 assert re.search(r'\.reqntid\s+640',ptx)
 assert re.search(r'setmaxnreg\.dec\.sync\.aligned\.u32\s+'+str(donor),ptx)
 assert re.search(r'setmaxnreg\.inc\.sync\.aligned\.u32\s+'+str(compute),ptx)
 assert sass.count('UTCQMMA.WS')==mmas
 assert sass.count('UTMALDG.2D.GATHER4')==20
 assert 256*compute+384*donor==640*96
 records[version]={'threads':640,'initial_registers':96,'donor':donor,'compute':compute,
 'register_pool':61440,'native_gather4_sites':20,'native_mma_sites':mmas,'local_spill':False}
print(json.dumps(records))
Path('artifacts/eight_producer_compile/control.json').write_text(json.dumps(records,indent=2)+'\n')
dest=Path('artifacts/snapshots/eight_producer_compile')
assert not dest.exists()
for p in Path('artifacts/eight_producer_compile').rglob('*'):
 if p.is_file() and p.suffix in {'.json','.txt','.log','.ptx'}:
  out=dest/p.relative_to('artifacts/eight_producer_compile')
  out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(p,out)
for v in ['v265','v266']:
 shutil.copy2('kernel_'+v+'.py',dest/v/('kernel_'+v+'.py'))
PY
