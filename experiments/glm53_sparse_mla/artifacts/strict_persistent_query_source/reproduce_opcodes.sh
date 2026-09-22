#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
import csv,collections,re,json
root=Path('experiments/glm53_sparse_mla/artifacts')
inputs={'v197':root/'v195_v197_source/v197_source_unlocked.csv','v273':root/'strict_persistent_query_source/v273_source.csv'}
counts={}
for version,p in inputs.items():
 c=collections.Counter()
 with p.open() as f:
  next(f)
  for r in csv.DictReader(f):
   s=r['Source'].strip()
   if s.startswith('@'): s=s.split(None,1)[1]
   if not s: continue
   value=r.get('Instructions Executed','').replace(',','')
   if value:
    c[s.split()[0]]+=int(float(value))
 counts[version]=c
d={k:counts['v273'][k]-counts['v197'][k] for k in counts['v273'].keys()|counts['v197'].keys()}
result={'totals':{v:sum(c.values()) for v,c in counts.items()},'counts':counts,'deltas':d}
(root/'strict_persistent_query_source/v197_v273_opcode_counts.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['totals'])
print('largest deltas',sorted(d.items(),key=lambda kv:abs(kv[1]),reverse=True)[:16])
PY

