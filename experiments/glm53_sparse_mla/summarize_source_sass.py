"""Summarize instruction-level NCU samples; sample shares are not speedup estimates."""
import argparse
import csv
import io
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('source_csv')
parser.add_argument('output_json')
args = parser.parse_args()
lines = Path(args.source_csv).read_text().splitlines()
rows = [r for r in csv.DictReader(io.StringIO('\n'.join(lines[1:])))
        if r.get('Address', '').startswith('0x')]

def number(row, key):
    value = row.get(key, '').replace(',', '')
    return float(value) if value and value != '-' else 0.0

metrics = ['stall_long_sb', 'stall_short_sb', 'stall_barrier', 'stall_mio',
           'stall_wait', 'stall_no_inst', 'stall_not_selected',
           'L1 Wavefronts Shared Excessive']
report = {'source': args.source_csv, 'purpose': 'sample attribution; not a speedup estimate',
          'totals': {}, 'top': {}}
for metric in metrics:
    total = sum(number(row, metric) for row in rows)
    report['totals'][metric] = total
    selected = sorted(enumerate(rows), key=lambda item: number(item[1], metric), reverse=True)[:8]
    report['top'][metric] = [
        {'address': row['Address'], 'sass': row['Source'].strip(),
         'value': number(row, metric), 'share_of_metric_sum': number(row, metric)/total,
         'context': [{'address': r['Address'], 'sass': r['Source'].strip()}
                     for r in rows[max(0,index-4):index+2]]}
        for index,row in selected if number(row, metric)>0 and total>0]
for metric in ('L1 Wavefronts Shared', 'L1 Wavefronts Shared Ideal', 'Instructions Executed'):
    report['totals'][metric] = sum(number(row, metric) for row in rows)
Path(args.output_json).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'totals': report['totals'], 'top_long_sb': report['top']['stall_long_sb'][:3]}, indent=2))
