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

### Iteration 002 result

Full target 8-row correctness PASS with identical errors to v001. Paired warm
event medians: TRTLLM 1692.96 us, v002 26413.31 us; TRT/candidate=0.064095x.
A 10.18x improvement over v001, still 15.60x slower than the paired baseline.

NCU: 255 registers/thread; shared memory 111.372 KB; achieved occupancy 6.252%;
eligible warps/scheduler 0.2152; tensor-pipe active 6.0297%; long-scoreboard
0.5928 cycles/issued instruction. Additional hardware counters measured
194510848 local-load sectors and 164084856 local-store sectors, plus 805534747
shared-load and 402922617 shared-store bank conflicts. These counters establish
that local-memory traffic and shared-memory conflicts are substantial, beyond
just the original scalar global gathers. NCU duration 44.9167 ms is diagnostic.
The smaller TMEM reservation did not increase measured occupancy; the allocation
permit lifetime must also be examined.

## Iteration 003 — reuse K shared memory as V

File: `experiments/glm53_sparse_mla/kernel_v003.py`.
Why: v002 NCU measured approximately 1.21 billion shared-memory bank conflicts.
The scalar FP8 V transpose is an unnecessary candidate source of these conflicts.
Reference inspection of FlashInfer `monolithic/mla_decode_fp8.py:522` shows
FP8 PV can use an MN-major V operand on Blackwell. Replace the K-major PV operand
and explicit transpose buffer with an MN-major view of the already gathered K
buffer, slicing the correct 256 output channels. No arithmetic precision change.
Validation pending; remaining local-memory pressure is the next target.

### Iteration 003 result

Full target 8-row correctness PASS (same errors as v001/v002). Paired warm
event medians: TRTLLM 1691.81 us, v003 19413.22 us; TRT/candidate=0.08715x.
Removing the transpose improved v002 by 1.36x, but remains 11.48x slower than
TRTLLM. NCU: 255 registers/thread; 94.988 KB shared memory; occupancy 6.252%;
eligible warps/scheduler 0.23354; tensor active 8.2868%; long scoreboard 0.75130.
Local-load/store sectors: 195297280 / 138193716. Shared-load/store bank conflicts:
102061 / 402653220. The shared-load conflicts almost disappeared, confirming
the transpose was a bottleneck. Store conflicts and register spilling remain.
NCU diagnostic duration 32.5455 ms.

## Iteration 004 — keep online output in TMEM

File: `experiments/glm53_sparse_mla/kernel_v004.py`.
Why: v003 still generates 333 million local-memory sectors; persistent output
plus temporary PV accumulators exceed the register budget. Keep the running
output in TMEM, load/rescale/store it before the next PV MMA and accumulate
directly into it. Keep the probability factor 256 throughout and divide it out
at final normalization. Also release the TMEM allocation permit immediately
after allocation, as in NVIDIA examples, instead of keeping it for the CTA
lifetime. Smoke b2 passes with the same numeric errors. Full target and NCU
measurements are pending.

### Iteration 004 allocation bug found at full concurrency

v004's b8192 run failed with an illegal memory access, despite b2 passing.
Compile-time layout inspection showed M64 FP32 accumulators consume the full
N columns: S needs 64 and O needs 256, starting at offset 64. The v002/v003
256-column allocation was therefore undersized (320 required). Keeping the
allocation permit until kernel exit happened to hide the overlap; releasing it
made the error visible at concurrency. Their passing numerical checks do not
establish memory safety, and those versions are rejected as unsafe, in addition
to being slow. The failed v004 source is retained as `kernel_v004_unsafe.py`;
its log is retained. v004 now allocates 512 columns, with compile-time column
bounds assertions using NVIDIA's `find_tmem_tensor_col_offset`. Full checks
are rerun before its performance is accepted. The earlier claim that 256
columns could allow two resident CTAs was incorrect for these exact layouts.

External Claude performance consultation retried with the functioning v003
and the v004 failure evidence; again returned `Empty output from Claude CLI`.

### Iteration 004 corrected result

512-column version: full b8192 8-row correctness PASS, max_abs 0.00686455 and
relative_RMSE 0.01479099. Paired warm event medians: TRTLLM 1691.78 us, v004
15984.67 us; TRT/candidate=0.10584x. NCU: 253 registers/thread, 94.988 KB
shared memory, achieved occupancy 12.431%, eligible warps/scheduler 0.26611,
tensor active 10.223%, long scoreboard 0.82261. **Local load/store sectors
are both zero**. Shared-load conflicts 205; shared-store conflicts 405680663.
Keeping O in TMEM eliminated the measured register spilling. Occupancy counts
resident warps, including any waiting for TMEM allocation; it does not prove
two CTAs are computing concurrently. NCU diagnostic duration 26.3702 ms.

## Iteration 005 — vectorize probability stores

File: `experiments/glm53_sparse_mla/kernel_v005.py`.
Why: v004 has no local-memory traffic but still 406 million shared-store
conflicts. Scalar FP8 stores from one thread per head poorly match shared-memory
banks. Convert probabilities into packed FP8 registers, then use 128-bit stores
into the existing swizzled P layout. Same math and safe 512-column allocation.
Smoke b2 passes; full target and NCU pending.

## Iteration 006 — pack complementary TMEM datapath lanes

File: `experiments/glm53_sparse_mla/kernel_v006.py`.
Why: the M64 instruction uses only half of each 32-lane TMEM warp region.
PTX Layout F explicitly supports lane alignment 0 or 16. Store O in lanes
0–15 and S in lanes 16–31 within each warp region, using the same columns
but disjoint cells. This can safely fit both into 256 columns and permit
concurrent TMEM allocations. This is different from the invalid column packing
in v002/v003. Compile-time column bounds are asserted. Validation pending.
Primary ISA reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout

### Iteration 005 result

Full b8192 8-row correctness PASS (same errors as corrected v004). Paired
warm event medians: TRTLLM 1693.66 us, v005 14399.74 us; TRT/candidate=0.11762x.
NCU: registers fall from 253 to 186/thread; shared memory 94.988 KB; achieved
occupancy 12.430%; tensor active 11.467%; eligible warps/scheduler 0.28741;
long scoreboard 0.86799. Local sectors remain zero. Shared-load/store conflicts
428 / 203079187: probability vectorization removed roughly half the remaining
store conflicts. The other major path is scalar TMEM-to-shared score writes.
Diagnostic NCU duration 23.5537 ms. Next softmax layout should bypass that
intermediate score buffer and distribute reduction directly over TMEM fragments.

### Iteration 006 result

Full b8192 8-row correctness PASS, same errors as v004/v005. Paired warm
event medians: TRTLLM 1690.59 us, v006 8459.26 us; TRT/candidate=0.19985x,
1.70x faster than v005 but still 5.00x slower than TRTLLM. NCU: 186 registers,
shared memory 94.988 KB; occupancy 12.427%;
tensor active 19.176%; eligible warps/scheduler 0.53872; long scoreboard
0.98258. Local sectors remain zero. Shared-load/store conflicts 34247027 /
203727957. Similar residency but twice the eligible warps and higher tensor
activity support that concurrent TMEM use now works. NCU duration 14.0830 ms.
Compute Sanitizer memcheck is being run at b512 to exercise concurrent CTAs.

## Iteration 007 — direct TMEM softmax

File: `experiments/glm53_sparse_mla/kernel_v007.py`.
Why: the intermediate S buffer still receives scalar stores with severe bank
conflicts. A diagnostic compile of `Ld16x32bx2Op` confirmed each thread owns one
head and 32 contiguous key columns, and lanes differing by 16 cover the other
half of that head. Remove the shared S buffer entirely; reduce local max/sum
in registers and exchange with an XOR-16 shuffle. All 128 threads participate,
then vector-store FP8 P. Keep O correction and MMA unchanged. Diagnostic source
and its coordinate mappings are preserved as `kernel_v007_diag.py`.
Validation pending.

### Concurrent memory-safety check

Compute Sanitizer memcheck on v006, b512, full TopK/context, checked rows 0/511:
PASS, **ERROR SUMMARY: 0 errors**. The sanitizer covers >1000 CTAs, exercising
TMEM reuse and concurrent allocations. It does not replace full target numerical
checks. Sanitizer timing is not a performance result.

### Iteration 007 result

Full b8192 8-row correctness PASS, max_abs 0.00686455, relative_RMSE 0.01479099.
Paired warm event medians: TRTLLM 1691.71 us, v007 7694.37 us, ratio 0.21986x.
NCU: 255 registers/thread, 78.604 KB shared memory, occupancy 12.439%, tensor
active 22.513%, eligible warps/scheduler 0.45178, long scoreboard 1.66207.
Shared store conflicts fall to 3288796 (98.4% lower than v006); shared load
conflicts 14907443. But local load/store sectors rise to 579862528 / 22973288.
Register spilling returned, offsetting much of the reduced shared traffic.
NCU diagnostic duration 11.9897 ms.

## Iteration 008 — smaller output register fragments

File: `experiments/glm53_sparse_mla/kernel_v008.py`.
Why: direct softmax leaves more per-thread state alive across the loop; loading
128 FP32 output elements/thread at once crosses the register budget. Correct
and write out O in 64-column subtiles (32 elements/thread), retaining O in
TMEM between subtiles. The MMA shapes, number of CTAs and arithmetic stay
unchanged. Hypothesis: eliminate local-memory traffic without restoring the
conflicting shared-score buffer. Validation pending.

## Iteration 009 — eliminate duplicated QK and sparse gather

File: `experiments/glm53_sparse_mla/kernel_v009.py`.
Why: two output CTAs repeat QK and gather the same selected KV. Assign the full
512 outputs to one CTA, using two PV N tiles and one QK/softmax computation.
Retain v008's small correction/epilogue fragments. O now needs 512 TMEM columns;
S uses the complementary lanes. This trades concurrent TMEM users for less
work, so measurement must decide whether it helps. Validation pending.

### Iteration 008 result

Initial compilation required an explicit true 64-column alignment assertion on
the subtile TMEM pointer; the verifier failure log is retained. Corrected full
b8192 8-row check PASS with the same errors as v007. Paired warm event medians:
TRTLLM 1691.01 us, v008 7083.84 us; ratio 0.23871x. NCU: 255 registers/thread,
78.604 KB shared memory, occupancy 12.418%, tensor active 23.513%, eligible
warps/scheduler 0.42921, long scoreboard 1.59163. Local-load/store sectors
373293056 / 11952196. Shared-load/store conflicts 10383563 / 3279318.
Smaller O fragments reduce spilling but do not eliminate it; the original
hypothesis was incomplete. The next register analysis must inspect generated
code and compiler liveness rather than assuming O alone causes the pressure.
NCU diagnostic duration 11.4620 ms.

### Iteration 009 bring-up and result

First full-output smoke failed: the default CuTe M64 fragment packs the second
256-column MMA N tile into the other TMEM lane half (stride 1048576), which
was reserved for S. Set the MMA N-tile stride explicitly to 256 columns. The
failing diagnostic layout and log are retained. Corrected b2 and full b8192
8-row checks PASS with the same error as v008.

Paired warm medians: TRTLLM 1691.78 us, v009 7310.24 us; ratio 0.23143x.
This is slower than v008's 7083.84 us: removing duplicate work did not overcome
reduced concurrency. NCU: 255 registers/thread, 78.604 KB shared memory;
occupancy 12.372%, tensor active 14.813%, eligible warps 0.21598, long scoreboard
1.40750. Local-load/store sectors 205651968 / 6627664; shared-load/store conflicts
530 / 172505. Diagnostic duration 11.9067 ms.

## Iteration 010 — vectorize the masked KV zero fill

File: `experiments/glm53_sparse_mla/kernel_v010.py`, based on v008.
Why: a b2 diagnostic PTX/SASS build reveals no explicit PTX local arrays, but
90 static LDL/STL instructions after register allocation. The invalid-slot
branch expands to 18 x 16 scalar byte stores with separate swizzled addresses;
these addresses can be hoisted and remain live even when runtime slots are
valid. Replace each 16-byte zero fill with one vector store, retaining all
masking semantics. Full-target NCU will decide whether this removes spilling.
Text codegen and analysis are retained under `artifacts/v008_codegen`.

## Iteration 011 — full output with vectorized zero fill

File: `experiments/glm53_sparse_mla/kernel_v011.py`, based on corrected v009.
Apply the same zero-fill change to the full-output CTA design, to reassess the
QK-duplication versus concurrency tradeoff after register pressure is reduced.
Validation pending for v010/v011.

### Iteration 010 result — zero-fill liveness hypothesis confirmed

Full b8192 8-row correctness PASS, same error as v008. Paired warm event medians:
TRTLLM 1691.87 us, v010 5277.73 us; ratio 0.32057x. This is 1.34x faster than
v008. Registers fall from 255 to 128/thread and local load/store sectors both
fall to zero. The zero-fill path's address liveness was a major spill cause,
even though the selected chunk has valid indices throughout.

NCU: 78.604 KB shared memory; occupancy 12.413%; tensor active 32.250%; eligible
warps/scheduler 0.28713; long scoreboard 3.23416. Shared-load/store conflicts
26863016 / 6124491. Diagnostic duration 8.35981 ms. With spilling removed,
serialized loads/computation and duplicate QK remain major optimization targets.

### Iteration 011 result

Full b8192 8-row correctness PASS. Paired warm medians: TRTLLM 1691.90 us,
v011 5798.05 us, ratio 0.29181x. Still slower than v010 (5277.73 us), so
retain the split-output design as the optimization base. NCU: 128 registers,
78.604 KB shared memory, occupancy 12.368%, tensor active 19.304%, eligible
warps 0.16600, long scoreboard 2.46240. Local-load/store sectors 2097152 /
134664 (much reduced but nonzero). Shared-load/store conflicts 364 / 236013.
Diagnostic duration 9.13267 ms. Reduced work alone is insufficient while the
pipeline has long serialized phases and only one CTA can own the 512 columns.

## Iteration 012 — overlap sparse KV loads with attention computation

File: `experiments/glm53_sparse_mla/kernel_v012.py`, based on v010.
Why: with spills eliminated, v010 spends more cycles waiting for global loads.
Allocate two K/valid buffers. Gather tile 0 before the loop; while QK/softmax/PV
use the current tile, issue cp.async loads for the next tile into the other
buffer. Wait for pending copies after PV, before switching buffers. Retain
all existing masking/zero-fill logic. Expected shared usage stays just under
the limit for two resident CTAs. Full correctness and NCU are pending.
