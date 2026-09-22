#!/usr/bin/env bash
set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
/opt/sglang/bin/python - <<'PY'
from pathlib import Path
import csv,json
out={}
for v in ('v190','v254'):
 p=Path('artifacts')/(v+'_ncu_raw.csv')
 if not p.exists():p=Path('artifacts/snapshots')/v/p.name
 units,row,*_=list(csv.DictReader(p.open()))
 keys=[k for k in row if (k.startswith('dram__') and (k.endswith('.sum') or 'bytes_read' in k or 'bytes_write' in k)) or k in ('lts__t_sectors.sum','lts__t_sector_hit_rate.pct','lts__t_sectors_srcunit_tex_op_read.sum','lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum','lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')]
 out[v]={k:{'value':row[k],'unit':units[k]} for k in keys}
Path('artifacts/v254_cache_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
PY
