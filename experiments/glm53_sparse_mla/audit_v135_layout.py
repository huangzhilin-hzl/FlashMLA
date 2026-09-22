"""Static address/ownership audit for v135; not a device or numerical test."""
import json
from pathlib import Path

scores = set()
outputs = set()
shadow = set()
for tid in range(128):
    head = tid % 64
    for j in range(32):
        key = (tid // 64) * 32 + j
        assert 0 <= key < 64
        scores.add((head, key))
        # QK and the final physical output chunk intentionally share columns224..255.
        shadow.add((tid, 224 + j))
    for tile in range(8):
        for j in range(32):
            channel = (tile // 4) * 256 + (tid // 64) * 128 + (tile % 4) * 32 + j
            assert (head, channel) not in outputs
            outputs.add((head, channel))
            assert 0 <= tile * 32 + j < 256
assert scores == {(h,k) for h in range(64) for k in range(64)}
assert outputs == {(h,d) for h in range(64) for d in range(512)}
assert shadow == {(dp,c) for dp in range(128) for c in range(224,256)}
for stage in range(2):
    covered = set()
    for warp in range(2):
        for group in range(8):
            row = warp * 32 + group * 4
            for lane in range(4):
                assert row + lane not in covered
                covered.add(row + lane)
    assert covered == set(range(64))
report = dict(purpose='static assumed-WS-layout ownership audit, not runtime validation',
              score_elements=len(scores), output_elements=len(outputs),
              shadow_elements=len(shadow), tmem_columns=256,
              q_bytes=64*576, kv_bytes=2*64*576, p_bytes=64*64,
              producer_indices='all64 exactly once per stage', pass_=True)
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/v135_layout_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
