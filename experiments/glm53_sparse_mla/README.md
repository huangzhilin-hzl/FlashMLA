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

# Full target, paired comparison, expanded sampled correctness, stable timing.
/opt/sglang/bin/python bench.py \
  --kernel-version v049 --block-k 128 \
  --backends trtllm cute --scope native --check-rows 64 \
  --warmup-iters 20 --repeat-iters 100 --cache both --timing cuda-graph \
  --output-json artifacts/v049_full_graph.json

# Short event-based tuning run, followed by one warmed NCU invocation.
bash run_iteration.sh v049 128
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

**New expanded check:** v049 fails 1 of 16,777,216 outputs in the full-target
512-row/seed1234 check; v051 fails 2. TRTLLM passes that same check. The v049
command above is a performance reference with a known accuracy failure,
not a generally validated implementation. A residual-FP8 probability path
is being evaluated without loosening the original tolerances.

Full-target iterations use the original FP32 reference and unchanged tolerances
(`atol=0.01`, `rtol=0.05`). The usual tuning check samples 8 rows. v013, v016, v034, v037, v039, v044, v045 and v049 also
passed 64 sampled rows on the target chunk3, with graph warm/cold measurements.
That is not exhaustive validation of all rows, shapes or quantization scales.

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
