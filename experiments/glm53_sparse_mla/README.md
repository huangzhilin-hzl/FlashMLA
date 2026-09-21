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
  --kernel-version v086 --block-k 128 \
  --backends trtllm cute --scope native --check-rows 512 \
  --warmup-iters 20 --repeat-iters 100 --cache both --timing cuda-graph \
  --output-json artifacts/v086_accuracy512_graph.json

# Short event-based tuning run, followed by one warmed NCU invocation.
bash run_iteration.sh v086 128
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
| v086, fused scaled exp2 input | 9 on seed1234 / 6 on seed5678, identical coordinates/values to TRTLLM | Current target-workload fast path |
| v049, P scale256 | 11 | Earlier timing reference |
| v053, residual FP8 | 0 | Original higher-precision path |
| v065, residual FP8 with V collector reuse | 0 via full bitwise equivalence to v053 | Exact-equivalence optimization |
| v067, residual FP8 with bounded scaling anchor | 0 on two independent full-reference seeds | Earlier validated higher-precision path |
| v075, probability scale folded into exp2 | 0 on two independent full-reference seeds | Fastest validated higher-precision path |

v086 matches more than 99.999% of TRTLLM BF16 outputs on both full target
seeds. Graph warm/cold medians are 1873.54/1880.13 µs versus paired TRTLLM
1869.82/1946.53 µs: near parity warm, faster cold in that run. Qualified b512
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
  --kernel-versions v086 v075 --include-trtllm \
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
