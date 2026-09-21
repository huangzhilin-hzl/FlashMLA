"""Extract a compact, unit-preserving summary from NCU's wide raw CSV."""
import csv
import json
import sys

KEYS = [
    'gpu__time_duration.sum',
    'launch__registers_per_thread',
    'launch__shared_mem_per_block_dynamic',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'smsp__warps_eligible.avg.per_cycle_active',
    'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
]

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
units, *kernels = rows
print(json.dumps([
    {'kernel': row.get('Kernel Name'),
     'metrics': {key: {'value': row.get(key), 'unit': units.get(key)} for key in KEYS}}
    for row in kernels
], indent=2))
