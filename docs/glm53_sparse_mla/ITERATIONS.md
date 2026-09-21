# GLM5.3 FP8 sparse MLA on B300 — iteration log

## Scope and evidence rules

Target: b=8192, s_q=1, H=64, Dqk=576, Dv=512, TopK=2048, context=131072,
page size 64, CP4/rank0/chunk3, softmax scale 0.0625. Canonical Q and KV are
unit-scale E4M3; output is BF16. All selected keys must be processed. No reference,
TRTLLM, or other attention backend is called from the candidate kernel.

Branch: `molou/glm53_sparse_mla_dev`, based on upstream `ba89a34`.
Original user benchmark is preserved byte-for-byte in
`experiments/glm53_sparse_mla/benchmark_source.py`; `bench.py` injects a candidate
without changing the reference or timing implementation. Small query counts are
compile/correctness probes only, never performance evidence for the target.
Every reported latency must identify timing mode, workload, correctness result,
GPU and source iteration. NCU-instrumented durations are diagnostic, not benchmark
latencies. Failed iterations and missing measurements remain explicit.

## Environment (2026-09-21)

Pod: `molou/molou-glm53-tp8-ep8-3048-0920`, container `server`.
Only physical GPU1 is used, selected by UUID
`GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2`. Initial memory usage 0 MiB and
utilization 0%. Other GPUs are not modified or reset.
B300 SXM6 AC; PyTorch 2.13.0+cu130; FlashInfer 0.6.18; CuTeDSL 4.6.2;
Nsight Compute 2025.3.0. Remote experiment directory `/tmp/glm53_sparse_mla_dev`.

## Baseline B0

User script, full target shape, correctness 8 rows passed (max abs 0.0076411,
relative RMSE 0.0150462). CUDA events, 20 warmups, 100 repeats:
TRTLLM warm median 1860.70 us, cold median 1854.66 us. Full raw JSON and log
are being retained under `artifacts/baseline`. Reproduces the supplied screenshot
within about 1%. Clock and profiler variation must still be controlled by paired
baseline/candidate measurements in each later run.

## Iteration 001 — establish an independent FP8 implementation

File: `experiments/glm53_sparse_mla/kernel_v001.py`.
Why: existing FlashMLA decodes compressed FP8 KV using BF16 MMA; implement a true
FP8 QK/PV path before optimizing memory and overlap.

Changes: M64/KV64 tiles, one query per CTA, two CTAs for output channel halves;
explicit sparse gather into swizzled shared K, shared-memory transpose into a
K-major FP8 V operand; tcgen05 FP8 MMA with FP32 accumulation; online FP32
softmax, quantized probability scaled by 256 before PV; BF16 final output.
The two output CTAs duplicate QK. The initial pipeline is serialized and copies
partial PV to registers each tile. These are deliberate initial inefficiencies,
not claimed optimizations. No full KV materialization is required.

Validation: compilation/correctness in progress. NCU and target latency pending;
no speedup claimed.

## External expert and implementation references

- NVIDIA tcgen05 programming guide:
  https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html
- NVIDIA CuTeDSL tcgen05 API:
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html
- Local CUTLASS `examples/python/CuTeDSL/blackwell/dense_gemm.py`:
  shared operand layouts, accumulator fragments, TMEM copies and synchronization.
- FlashInfer `flashinfer/cute_dsl/attention/monolithic/mla_decode_fp8.py`:
  FP8 attention, online softmax and correction, FP8 V layout constraints.
- FlashInfer TRTLLM `fmhaKernels.cuh`: sparse generation Q64/KV128 selection.
- Current FlashMLA `sm100/decode/sparse/head64`: baseline pipeline/resource analysis.

External Claude expert tools were attempted twice and returned `Empty output from
Claude CLI`. A direct CLI fallback (including `--version`) exited 137 without
output. This is a tool availability failure; no expert response is fabricated.
Primary NVIDIA documentation and reference sources remain available. Reattempt
an independent expert review after a functioning kernel and NCU evidence exist.

### Iteration 001 bring-up corrections

1. CuTeDSL 4.6.2 requires MMA shared fragments to have affine layouts: move
   the swizzle into the shared pointer (`layout.outer`, `swizzle=layout.inner`).
2. First execution stalled because raw `tcgen05.commit` was executed by all 32
   lanes. The installed NVIDIA API explicitly requires `elect_one()`. The
   experiment's own PID was terminated; no other process or GPU was reset.
3. The original input helper binds `batch=LOCAL_TOKENS` at definition time.
   The probe wrapper now also updates that default when requesting small smoke
   shapes. The unchanged full-target b8192 experiment is unaffected.
4. Shared K-to-V transpose is separated from K loading by a CTA barrier, and
   TMEM reads receive explicit async-view load fences before reuse.

### Iteration 001 smoke result

b=2, full context and TopK, both rows checked: PASS. max_abs=0.0069823,
relative_RMSE=0.0146712. This establishes the FP8 mathematics and full 576/512
channel coverage for the probe. It does not establish full-workload correctness
or performance. Event median was 1970.4 us even at b=2, indicating major initial
implementation overhead; full target and NCU follow next.

### Iteration 001 full target and NCU evidence

Full b8192 correctness passed the original 8-row check: max_abs=0.00686455,
relative_RMSE=0.0147910. Paired warm CUDA-event medians (3 warmups, 5 repeats):
TRTLLM 1691.78 us; v001 268827.58 us; TRT/candidate=0.006293x.
This initial implementation is approximately 159x slower and is rejected for
performance; it remains a reproducible correctness starting point.

NCU (15 passes, default NCU clock control, SM about 1.09 GHz): v001 254 registers
per thread, 111.37 KB dynamic shared memory, achieved occupancy 6.25%, eligible
warps/scheduler 0.11, tensor pipe active 0.664%, DRAM throughput 0.12%, long
scoreboard stall 6.106 cycles per issued instruction (about 65%). The diagnostic
405.92 ms duration is not the unprofiled benchmark latency. Local-spill derived
metrics report `no data`, so spilling is a hypothesis, not yet quantified.

NCU baseline: 128 registers/thread, 219.78 KB dynamic shared memory, achieved
occupancy 24.94%, eligible warps/scheduler 0.52, SM throughput 71.80%, DRAM
throughput 9.34%, diagnostic duration 2.84 ms at the same approximately 1.09 GHz.
NCU warns that TRTLLM Work ID affects some launched-work counts. Do not infer
occupancy from that warning's cluster-count fields.

## Iteration 002 — remove scalar gather serialization and excess TMEM reservation

File: `experiments/glm53_sparse_mla/kernel_v002.py`.
Why: v001 is latency limited, with very low Tensor Core activity and many long
scoreboard stalls, while achieved residency is only one 4-warp CTA per SM.
Changes: replace scalar FP8 Q/KV global loads with aligned 16-byte cp.async
copies; use 4 threads per token row, 9 64-byte groups for the full 576 dimensions;
keep sparse indices unchanged; reserve 256 TMEM columns instead of 512 and move
PV scratch to column 64, allowing two CTAs/SM if other resources permit.
The vector pointers explicitly assert true 16-byte alignment (576-byte rows and
16-byte column offsets). Correctness and performance are pending.
