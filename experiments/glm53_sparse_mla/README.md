# GLM5.3 sparse MLA experiments

Independent CuTeDSL FP8 QK/PV kernels for the requested B300 workload:
`b=8192, s_q=1, H=64, Dqk=576, Dv=512, TopK=2048`, unit-scale E4M3 Q/KV,
BF16 output. Each numbered file preserves an iteration; the log distinguishes measured
versions from pending prototypes and rejected experiments. These are
experimental kernels, not an installed replacement for FlashMLA.

- [Measured results](../../docs/glm53_sparse_mla/RESULTS.md)
- [Changes, NCU evidence, failures and references](../../docs/glm53_sparse_mla/ITERATIONS.md)

`benchmark_source.py` is the unchanged copy of the user's `sm103_mla.py`, SHA256
`d843320fb5147a807135282a1b87bb4c247cdd7a6a16b9107d546c48c8a8fb66`.
`bench.py` adds the candidate without changing reference calculations or timing.
The source has some stale human-readable `4096` labels; the recorded tensor
shapes and `--local-tokens 8192` configuration identify the actual workload.

## Run in the authorized pod

Run GPU experiments sequentially. `check_gpu_idle.py` performs two read-only
utilization/memory checks before each `run_iteration.sh` invocation and refuses
timing/profiling if GPU1 is busy; it never terminates another process. Run this
preflight manually before other GPU validation commands too. The selected UUID is physical GPU1 of
`molou/molou-glm53-tp8-ep8-3048-0920`, container `server`. CUDA exposes that
selected GPU as logical device 0. The remote directory is
`/tmp/glm53_sparse_mla_dev`; Python is `/opt/sglang/bin/python`.

```bash
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a

# Fast single-P path with closely matching TRTLLM precision; see all-row limits below.
/opt/sglang/bin/python bench.py \
  --kernel-version v125 --block-k 128 \
  --backends trtllm cute --scope native --check-rows 512 \
  --warmup-iters 20 --repeat-iters 100 --cache both --timing cuda-graph \
  --output-json artifacts/v125_accuracy512_graph.json

# Short event-based tuning run, followed by one warmed NCU invocation.
bash run_iteration.sh v125 128
```

`run_iteration.sh` saves raw JSON, logs, NCU details/CSV and an immutable per-run
snapshot under `artifacts/snapshots`. NCU durations are diagnostic; they must not
be used as benchmark latencies. NCU binaries and CUBIN files are retained on disk
but excluded from Git. Text evidence is committed alongside each kernel.

Regenerate the measured summary locally:

```bash
python3 experiments/glm53_sparse_mla/summarize.py > docs/glm53_sparse_mla/RESULTS.md
```

## Validation limits

The full 8192-row/seed1234 audit checks all 268,435,456 output elements:

| Candidate | Original-tolerance failures | Intended comparison |
|---|---:|---|
| TRTLLM | 9 | Baseline FP8 precision |
| v054, P scale448 | 9, identical coordinates/values to TRTLLM | Earlier fast path |
| v086, fused scaled exp2 input | 9 on seed1234 / 6 on seed5678, identical coordinates/values to TRTLLM | Earlier target-workload fast path |
| v090, validity bitmaps | Same as v086 via full bitwise checks on both seeds | Earlier fast path with observed sustained speedup |
| v091, bitmaps plus early stage release | Same as v090 via full bitwise checks on both seeds | Earlier fast path |
| v094, four producer warps | Same as v091 via full bitwise checks on both seeds | Earlier fast path |
| v096, pre-wait index fetching | Same as v091 via full bitwise checks on both seeds | Earlier fast path |
| v097, pre-wait index fetching with four producer warps | Same as v096 via full bitwise checks on both seeds | Earlier fast path |
| v105, dedicated MMA issuer | Same as v097 via full bitwise checks on both seeds | Earlier fast path |
| v108, per-thread P readiness | Same as v105 via full bitwise checks on both seeds | Earlier fast path |
| v112, balanced denominator reduction | 9 on seed1234 / 6 on seed5678, identical coordinates/values to TRTLLM | Earlier fast path |
| v125, direct256-bit output stores | Full bitwise equality to v112 on both seeds, short case and masks | Current fast path |
| v049, P scale256 | 11 | Earlier timing reference |
| v053, residual FP8 | 0 | Original higher-precision path |
| v065, residual FP8 with V collector reuse | 0 via full bitwise equivalence to v053 | Exact-equivalence optimization |
| v067, residual FP8 with bounded scaling anchor | 0 on two independent full-reference seeds | Earlier validated higher-precision path |
| v075, probability scale folded into exp2 | 0 on two independent full-reference seeds | Earlier higher-precision path |
| v098, pre-wait index fetching | Full bitwise equality to v075 on both seeds, short case and masks | Earlier higher-precision path |
| v106, dedicated MMA issuer | Full bitwise equality to v098 on both seeds, short case and masks | Earlier higher-precision path |
| v114, balanced denominator reduction | 0 on two full target seeds, full short case and masks | Earlier higher-precision path |
| v128, direct256-bit output stores | Full bitwise equality to v114 on both seeds, short case and masks | Current higher-precision path |

The current fast path v125 matches v112 bitwise on both full8192-row seeds,
all1024 short-case rows and masked inputs, retaining the FP8 limits below.
Qualified b2 synccheck and b512 memcheck report zero errors. Direct256-bit
output stores replace the shared transpose, with verified32-byte alignment.

Eager20/100 warm/cold medians are 1720.98/1733.12 µs versus TRT
1890.26/1911.52 µs; Graph medians are 1736.62/1720.40 µs versus
1871.82/1914.78 µs. Three rotated orders give warm1688.99–1691.26 µs
versus v1121700.98–1712.13 µs and TRT1871.82–1878.10 µs; cold
1687.54–1693.62 µs versus v1121695.87–1699.86 µs and TRT
1858.26–1859.55 µs. It improves on v112 in each recorded ordering, while
separate-run eager/Graph differences are smaller and variable. Short tuning is
1648.86 µs versus TRT1690.75 µs. These observations use unlocked clocks and
do not establish all-input accuracy or a universal speedup.

Earlier fast path v112 independently retains the same 9/6 full-target
failures as TRTLLM on seeds1234/5678. Only 2464/2601 of 268,435,456 BF16
outputs differ from TRTLLM. The full short case retains 7650 failures for both;
the mask case matches v108 bitwise and retains 86 failures versus TRT's 78.
Qualified b512 memcheck reports zero errors. The denominator uses a balanced
32-value sum tree, which changes rounding and requires these fresh audits.

Its 20/100 eager warm/cold medians are 1716.51/1745.12 µs versus TRT
1888.85/1915.10 µs; Graph medians are 1738.34/1722.18 µs versus
1869.89/1919.95 µs. Three rotated orders give warm 1701.89–1712.13 µs
versus v108 1712.34–1713.34 µs and TRT 1871.36–1876.06 µs; cold
1697.63–1697.71 µs versus v108 1708.10–1719.46 µs and TRT
1857.46–1859.60 µs. It improves on v108 in each recorded ordering, although
one warm pair differs by less than 1 µs. These are observed ranges of round
medians with unlocked clocks, not confidence intervals. The baseline FP8
precision limits remain; v128 is the current higher-precision option.

Earlier fast path v108 matches v105 bitwise on both full 8192-row seeds,
all 1024 short-case rows and the masked input; qualified b2 synccheck/b512
memcheck report zero errors. Its 20/100 eager warm/cold medians are
1753.09/1766.93 µs versus TRT 1876.54/1915.55 µs, and Graph medians are
1716.29/1734.90 µs versus 1872.02/1933.12 µs. Four rotated orders measure warm
1712.40–1720.51 µs versus v105 1722.46–1730.75 µs and TRT
1867.87–1879.95 µs. Warm improves in every ordering; cold ranges overlap v105
and one ordering is slightly slower. Every compute thread releases its own
P/correction writes to the readiness barrier. All baseline FP8 precision limits
remain. Short tuning is 1680.42 versus TRT 1689.89 µs; the small margin alone
does not establish a robust short-run advantage.

Earlier fast path v105 matches v097 bitwise on both full 8192-row seeds,
all 1024 short-case rows and the masked input. Qualified b2 synccheck and b512
memcheck report zero errors. Its 20/100 eager warm/cold medians are
1740.94/1768.85 µs versus TRT 1876.64/1913.94 µs; Graph medians are
1762.86/1742.77 µs versus TRT 1867.81/1916.77 µs. Three rotated orders give
warm 1726.42–1730.53 µs versus v097 1757.30–1760.72 µs and TRT
1851.44–1880.13 µs. A separate MMA warp issues current PV before next QK,
with independent completion and P-readiness barriers. The original FP8
accuracy limits below still apply. Short five-event timing is near parity with
TRT (1693.86 versus 1691.84 µs), not a demonstrated short-run win.

The current higher-precision v128 path matches v114 bitwise on both full8192-row
seeds, all1024 short-case rows and masked inputs, retaining its audited original-
tolerance passes. Qualified b2 synccheck/b512 memcheck report zero errors.
Eager20/100 warm/cold medians are1941.71/1945.70 µs versus TRT1880.22/1916.98 µs;
Graph medians are1970.22/1945.54 µs versus1871.52/1916.35 µs. Three rotated
orders give warm1922.99–1925.22 µs versus v1141939.54–1939.66 µs and TRT
1869.34–1882.32 µs; cold1916.85–1917.02 µs versus v1141927.20–1929.12 µs
and TRT1857.36–1865.76 µs. It improves on v114 in every recorded ordering,
while remaining slower than TRT. Short tuning is1802.37 µs versus TRT1691.71 µs.

Earlier higher-precision v114 path independently passes the original
atol0.01/rtol0.05 tolerance on all 268,435,456 output elements for each of
two seeds, all 33,554,432 short-case elements and the masked input. Qualified
b512 memcheck reports zero errors. Relative RMSE is about 0.001736 on both
full target seeds, roughly 8.6 times lower than the baseline FP8 path.

Eager20/100 warm/cold medians are 1955.95/1957.86 µs versus TRT
1882.72/1915.01 µs; Graph medians are 1996.70/1968.24 µs versus
1867.87/1918.93 µs. Three rotated orders give warm 1939.41–1939.73 µs
versus v106 1951.84–1964.11 µs and TRT 1869.86–1880.32 µs; cold
1927.15–1929.22 µs versus v106 1953.76–1960.18 µs and TRT
1857.52–1867.60 µs. It improves on v106 in every recorded ordering and
remains slower than TRT. Summation rounding changes, so these are independent
accuracy audits rather than a claim of bitwise equality to v106.

Earlier higher-precision v106 path matches v098 bitwise on both full seeds,
all short-case rows and the masked input; qualified b2 synccheck/b512 memcheck
report zero errors. Its 20/100 eager warm/cold medians are 1977.41/1981.01 µs
versus TRT 1874.05/1918.05 µs; Graph medians are 1998.90/1972.21 µs versus
1867.82/1916.93 µs. Rotated warm medians are 1962.75–1964.16 µs versus v098
2005.17–2007.23 µs and TRT 1867.87–1881.60 µs. It improves the higher-precision
path in every recorded ordering, while remaining slower than TRT.

The higher-precision v098 path matches v075 bitwise on both full seeds, all
short-case rows and the masked input, with qualified b512 memcheck reporting
zero errors. Its 20/100 Graph warm/cold medians are 2056.35/2037.58 µs versus
TRT 1871.82/1918.86 µs. Rotating orders measure warm 2005.14–2013.14 µs versus
v075 2043.94–2046.66 µs and TRT 1868.77–1878.14 µs. It improves the audited
higher-precision path while remaining slower than TRT.

v097 retains v096 output bits on both full 8192-row seeds, all 1024 short-case
rows and the masked input; qualified b512 memcheck reports zero errors. Its
20/100 eager-event warm/cold medians are 1787.86/1784.90 µs versus paired TRT
1883.94/1929.81 µs, and Graph medians 1794.08/1783.62 µs versus
1869.82/1918.70 µs. Three rotated orders measure warm 1757.28–1761.38 µs versus
TRT 1867.87–1878.16 µs and v096 1771.58–1771.63 µs. It combines independent
index fetching before stage waits with four producer warps and 256 compute
threads. The inherited FP8 precision limits below remain explicit.

v096 retains v091 output bits on both full 8192-row seeds, all 1024 rows of the
short case and the masked input; qualified b512 memcheck reports zero errors.
Its 20/100 eager-event warm/cold medians are 1806.46/1795.95 µs versus paired
TRT 1886.18/1916.88 µs; Graph medians are 1806.54/1796.08 µs versus
1880.22/1920.86 µs. Three rotated orders give warm 1769.98–1771.74 µs versus
TRT 1869.81–1878.22 µs, improving on v094 in every ordering. It moves the
read-only global index fetch before stage-reuse waiting while keeping shared
publication after that wait. This preserves the documented FP8 precision limits;
it does not establish an all-row FP32-reference pass. v128 is the current validated
higher-precision option.

v094 retains v091 output bits on both full8192-row seeds, the1024-row short
case and the masked input; qualified b512 memcheck reports zero errors. Its
20/100 eager-event warm/cold medians are1839.30/1837.06 µs versus paired TRT
1874.00/1923.28 µs, and Graph medians1839.20/1827.62 µs versus1865.82/1927.14 µs.
A four-order audit gives warm1819.04–1819.97 µs versusTRT1867.92–1870.10 µs,
and improves on v091 in every ordering. It uses384 threads with four producer
warps, keeping the256-thread compute path. The same precision limits below apply.

v091 matches v090 bitwise on both full8192-row seeds, the1024-row short case
and the masked input, with qualified b512 memcheck reporting zero errors.
Eager-event warm/cold medians are1852.51/1851.06 µs versusTRT1886.94/1917.02 µs;
Graph medians1847.62/1832.43 µs versusTRT1863.94/1926.14 µs. Three rotated
orders give warm v0911822.98–1824.83 µs versusTRT1869.98–1879.65 µs; it also
improves on v090 in every ordering. These are the same baseline-precision
outputs, including the failures documented below. v128 is the current higher-precision option on its audited inputs.

v090 matches v086 bitwise on all8192 rows of both seeds and all1024 rows of
the short case, in three repeats each. The mask case also matches v086 bitwise,
and qualified b512 memcheck reports zero errors. These comparisons transfer
v086's documented precision limits, including the shared target failures with
TRT; they do not establish a strict full-reference pass.

Using20 warmups and100 repeats, v090 eager-event warm/cold medians are
1859.63/1845.01 µs versus TRT1882.35/1919.04 µs. Graph medians are
1853.31/1839.36 µs versus TRT1867.95/1931.57 µs. A separate three-round
order-rotation audit measures warm v0901831.07–1831.63 µs versus TRT
1867.86–1878.13 µs, with an advantage in every ordering. These are observed
speedups for this workload and execution regime; the short five-event tuning
run still favors TRT. Clocks are not locked, and raw timing distributions are retained.

v086 matches more than 99.999% of TRTLLM BF16 outputs on both full target
seeds. Graph warm/cold medians are 1873.54/1880.13 µs versus paired TRTLLM
1869.82/1946.53 µs. A subsequent three-round order-rotation audit confirms
warm parity (~1872 µs each), but has v086 ~1878 µs versus TRT ~1857 µs cold;
the earlier cold advantage is not robust. Qualified b512
device memcheck reports zero errors. The full b1024/chunk0 case has7650
original-tolerance failures for both; the internal-hole/short case has86 for
v086 versus78 for TRTLLM. Target-input agreement does not imply a tolerance
pass or agreement on arbitrary masks. v075 passes these additional cases.

v054 matches about 99.96% of TRTLLM BF16 outputs bitwise on this input.
It passes 512 sampled rows for seeds1234/5678/42, but it does **not** pass the
strict all-row tolerance. v053 passes the entire seed1234 target and the
additional sampled, short and masked cases recorded in the iteration log.
v065 matches all v053 output bits in three full-target/seed1234 runs. Its
Graph warm/cold medians are 2279.70/2269.34 µs, versus paired TRTLLM
1871.50/1947.44 µs. This is still slower than TRTLLM, with higher reference
accuracy on the audited input.

v067 independently passes all outputs for seeds1234/5678 and all1024 rows of
a short-sequence case, plus the masked and qualified memcheck tests. Its
Graph warm/cold medians are 2134.14/2107.38 µs versus paired TRTLLM
1869.14/1926.99 µs. Lower probability scaling slightly increases relative
error while passing these original-tolerance audits. None of these tests
proves every possible input or scale.

v075 repeats the two full8192-row audits, full1024-row short case, masked case
and qualified b512 memcheck with zero failures. Graph warm/cold medians are
2097.22/2080.51 µs versus paired TRTLLM1874.14/1925.12 µs. It remains slower
than TRTLLM, while preserving the audited higher precision. Folding the scale
into exp2 changes rounding; no bitwise equivalence to v067 is claimed.

To reproduce the full shared-reference audit (accuracy only):

```bash
/opt/sglang/bin/python validate_full_accuracy.py \
  --kernel-versions v125 v128 --include-trtllm \
  --output-json artifacts/full_accuracy_seed1234.json
```

This audit records every backend's failures before returning nonzero if any
fails. For residual-P performance, use `--kernel-version v075` with `bench.py`;
v053 remains the direct full-reference-audit baseline.
No tolerance is relaxed to obtain a pass. `validate_full_accuracy.py --block-k`
defaults to128; pass64 for the split-head/N64 experiment v082.

Full-target iterations use the original FP32 reference and unchanged tolerances
(`atol=0.01`, `rtol=0.05`). The usual tuning check samples 8 rows. v013, v016, v034, v037, v039, v044, v045 and v049 also
passed 64 sampled rows on the target chunk3, with graph warm/cold measurements.
Those historical sampled checks do not supersede the full audit above.

The extra b1024/chunk0/seed5678 test fails the original tolerance for both v013
and TRTLLM; the failed numerical checks remain explicit in the iteration log.
The added v031 internal-hole/short-sequence test matches v020 bitwise but
fails the original FP32 tolerance; that is not a numerical pass. v033 timing
is non-isolated due to another GPU1 workload; do not rank that event result.
Memory checks and numerical checks are recorded separately. v002/v003 have an
undersized TMEM allocation and are retained only as rejected historical
experiments; `kernel_v004_unsafe.py`, v021/v022 (failed numerical layout experiments), and diagnostic files are also not candidates.

Candidates use online softmax, FP8 probabilities and FP32 accumulators. They
process the supplied sparse indices and do not call TRTLLM, FlashMLA, reference
attention, or expand the KV cache. Current wrappers are scoped to the stated
shape/layout and do not provide general API validation or LSE output.

## Offline compilation

While the authorized GPU is occupied, compile a TMA candidate with fake tensors:

```bash
CUDA_VISIBLE_DEVICES="" CUTE_DSL_ARCH=sm_103a /opt/sglang/bin/python \
  compile_offline.py --kernel-version v034
```

This emits PTX/CUBIN for inspection without allocating or launching GPU work.
Compilation and static resource reports are not runtime validation. v034 and later candidates additionally make cross-thread TMEM ordering explicit with tcgen05
fences. Earlier measured versions lack those explicit fences; their sampled
passes do not establish safety for every compiler or execution schedule.

Explicit `min_blocks_per_mp` launch bounds can require device-attribute queries
in this DSL. For those fake-tensor compilations, select the authorized GPU1,
run `check_gpu_idle.py`, and add `--initialize-cuda` to `compile_offline.py`.
That opt-in creates a CUDA context but launches no candidate kernel; the JSON
record distinguishes it from the default context-free compilation.


Cache-policy follow-up: `v129` changes only v125's output L2 priority to
evict-first. Full2seeds/short/mask bitwise equivalence and qualified sanitizers
pass. Two rotating-order audits favor it in8/10 warm and9/10 cold round medians,
but the gain is small and includes regressions in individual orders. Hardware
counters show about8.8% less DRAM read traffic in the recorded warmed profile.
It is a validated experimental alternative; v125 remains the established default.
KV evict-last (`v130`) has no consistent gain. Mandatory future-QK lookahead
(`v131`) delays PV and regresses to2.038ms; readiness-conditional lookahead
(`v132`) recovers to1.661ms but still does not improvev125. See the iteration log
for the complete sampled accuracy, timing and profiling limits.
