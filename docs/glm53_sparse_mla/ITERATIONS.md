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

### Iteration 012 result — occupancy regression

Full b8192 8-row correctness PASS. Paired medians: TRTLLM 1693.89 us, v012
7994.75 us; ratio 0.21188x, slower than v010. NCU: 128 registers, no local
sectors and no shared bank conflicts, but shared memory 115724 bytes/CTA and
occupancy only 6.240% (one CTA/SM). Tensor active 20.813%, eligible warps
0.17831, long scoreboard 1.86402; diagnostic duration 13.0101 ms.

NCU launch data reports an additional 1024 driver-reserved shared bytes/CTA;
the physical SM limit is 233472 bytes. Two CTAs require
`2 * (115724 + 1024) = 233496` bytes, exceeding that limit by 24 bytes even
before any allocation rounding. The prior estimate ignored driver reservation.
The runtime selected only 135168 bytes of shared carveout, sufficient for one
CTA, and the profiler explicitly reports a shared-memory block limit of one.

## Iteration 013 — reuse softmax metadata storage

File: `experiments/glm53_sparse_mla/kernel_v013.py`, based on v012.
Alpha factors are needed during online O correction; denominators are only
needed during the final epilogue. Alias their 64-float shared buffers, and write
the denominators once after the final PV and a synchronization. This saves
256 bytes/CTA and removes unnecessary per-tile denominator stores. The goal
is to restore two resident CTAs without changing the double-buffered pipeline.
Full target correctness, NCU and partial-tile checks are pending.

### Iteration 013 result

Full b8192 8-row correctness PASS. Paired medians: TRTLLM 1691.78 us, v013
4482.37 us; ratio 0.37743x. NCU confirms two-CTA residency: occupancy 12.405%,
shared memory 115468 bytes, 126 registers/thread, no local sectors. Tensor
active 36.693%, eligible warps 0.37793, long scoreboard 2.08951. Shared-load/
store conflicts 5668630 / 160100; diagnostic duration 7.34963 ms. The 256-byte
saving restores concurrency; the overlapping-load pipeline now improves on
v010 by 1.18x. Still 2.65x slower than the paired TRTLLM baseline.

Extended validation: memcheck at b1024/chunk0/seed5678 reports zero memory
errors, but the unchanged numerical tolerance fails on 6 of 524288 checked
elements (16 rows; maximum offending absolute error 0.0154580). This is a
**failed numerical check**, not an accepted validation. Compare TRTLLM on the
same inputs and inspect FP8 probability quantization before claiming broader
shape/seed coverage. The target b8192/chunk3 is being rechecked with 64 rows.
No tolerance has been relaxed.

## Iteration 014 — dedicated loader warpgroup

File: `experiments/glm53_sparse_mla/kernel_v014.py`, based on v013.
Use a 256-thread CTA: four warps load Q/KV and four execute QK/softmax/PV.
Two full/empty mbarrier pairs protect the two KV stages; producer and consumer
use separate 128-thread named barriers. Producers can prepare the next stage
while compute warps continue, without requiring the same warps to issue loads.
Retain alpha/denominator aliasing to stay within the two-CTA shared-memory limit.
Numerical behavior is unchanged by construction but still requires validation.

### Extended validation and stable graph measurements for v013

The exact target b8192/chunk3/seed1234 passes the expanded 64-row FP32 check:
v013 max_abs 0.00975490, relative_RMSE 0.0146073; TRTLLM max_abs 0.0101305,
relative_RMSE 0.0148794. Tolerances remain atol=0.01, rtol=0.05.

CUDA Graph, 20 warmups/100 repeats: warm TRTLLM 1879.84 us versus v013
4482.27 us (0.41939x); cold TRTLLM 1896.74 us versus v013 4485.31 us
(0.42288x). These use the original graph timing/eviction code and are kept
separate from the short CUDA-event tuning table. Both cache modes confirm
v013 remains substantially slower; cache eviction does not explain the gap.

For the extra b1024/chunk0/seed5678 partial-TopK test, TRTLLM also fails the
unchanged tolerance: 19/524288 elements, largest offending absolute error
0.0221396, at the same row/head/channel as v013's largest offending value.
v013 has 6 failing elements with 0.0154580. This supports a shared FP8
approximation limitation rather than evidence of a pipeline-specific memory
error; it does not turn either failed check into a pass. Broader numerical
coverage remains limited and needs separate treatment. Memcheck reports zero
addressing errors for that case. Raw failures are preserved under
`artifacts/v013_validation`.

### Iteration 014 result

Full b8192 8-row correctness PASS, same errors as v013. Paired medians: TRTLLM
1691.81 us, v014 3567.87 us; ratio 0.47418x. Loader/compute specialization
improves v013 by 1.26x. NCU: 124 registers/thread, shared memory 115504 bytes,
occupancy 23.224%, tensor active 43.641%, eligible warps 0.58263, long
scoreboard 4.11929. Local sectors remain zero; shared-load/store conflicts
22254878 / 11457038. Diagnostic duration 6.18675 ms. Increased warp residency
and overlapping work improve throughput, but the kernel still takes 2.11x
the paired TRTLLM time.

## Iteration 015 — revisit full-output CTAs with specialized loading

File: `experiments/glm53_sparse_mla/kernel_v015.py`, based on v014.
Assign all 512 output channels to one CTA while keeping a separate producer
warpgroup. Use the explicit nonoverlapping 256-column N-tile stride learned
in v009. This tests whether better loading overlap changes the earlier
full-output versus duplicated-QK tradeoff. Validation pending.

### Iteration 015 result

Full b8192 8-row correctness PASS. Paired medians: TRTLLM 1691.84 us, v015
3767.58 us; ratio 0.44905x, still slower than split-output v014 (3567.87 us).
NCU: 124 registers, 115504 shared bytes, occupancy 23.692%, tensor active
27.473%, eligible warps 0.29738, long scoreboard 3.30076. Local sectors zero;
shared-load/store conflicts 9660785 / 12387311. Diagnostic duration 6.44310 ms.
Resident warps include the CTA waiting for its 512-column TMEM allocation;
the lower eligible-warp and tensor activity remain consistent with that limit.

## Iteration 016 — full output with 128-key tiles

File: `experiments/glm53_sparse_mla/kernel_v016.py`, based on v015.
Use KV128 instead of KV64 to halve loop/barrier and online O-correction counts.
The full-output design already has only one active TMEM user; larger shared
buffers therefore do not sacrifice a second computing CTA. QK uses N128;
softmax threads each process 64 keys. The wrapper requires `--block-k 128`;
the reproducible run script forwards that value to both benchmark and NCU.
Accuracy and resource effects are pending.

### Iteration 016 result

Full b8192 8-row correctness PASS: max_abs 0.00673771, relative_RMSE 0.0148667.
Paired warm event medians: TRTLLM 1691.78 us, v016 2996.35 us; ratio 0.56461x.
KV128 improves on v015 by 1.26x and on v014 by 1.19x. NCU: 207 registers,
193840 shared bytes, occupancy 11.009%, tensor active 34.298%, eligible warps
0.26279, long scoreboard 3.69279. Local sectors zero; shared-load/store
conflicts 2548516 / 262695. Diagnostic duration 5.15706 ms.

The paired baseline's tensor activity is 62.491%. The next target is to
overlap QK for tile i+1 with softmax/correction for tile i, and PV for tile i
with softmax for tile i+1, rather than serially waiting after both MMAs.

## Iteration 017 — overlap adjacent attention tiles

File: `experiments/glm53_sparse_mla/kernel_v017.py`, based on v016.
Allocate two S regions in the unused TMEM lane half and two shared P buffers.
Prime QK for tile 0, then issue QK for tile i+1 before computing tile i's
softmax. Signal the KV/P stage's empty barrier directly from PV completion,
rather than making compute warps wait immediately after PV. Wait for previous
PV only before updating the running O accumulator; wait for the last PV before
epilogue. This removes some serialized QK/PV waits, within the two-KV-stage
capacity constraint. Producer/consumer phase arithmetic and resource pressure
require fresh smoke, full-target, and memory validation. Measurements pending.

### Iteration 017 result

Full-target 8-row numerical check PASS, same errors as v016. Paired event
medians: TRTLLM 1691.78 us, v017 4062.53 us (0.41644x), a regression.
NCU: 202 registers, 202040 shared bytes, occupancy 11.318%, tensor active
26.875%, eligible warps 0.18981, long scoreboard 5.64460. Local sectors zero;
shared-load/store conflicts 3599303 / 12550. Diagnostic duration 6.56237 ms.
With only two KV stages, early waiting for the next tile couples the consumer
back to the previous PV and producer refill. The measured lower tensor activity
rejects this ordering; no memory-safety certification is claimed for it.

## Iteration 018 — defer next-QK scheduling until after softmax

File: `experiments/glm53_sparse_mla/kernel_v018.py`, based on v017.
Move the wait for the next KV tile and its QK issue after current softmax/P
stores but before O correction. This lets softmax execute while the producer
refills the next stage, while retaining two S/P regions and hardware PV-to-empty
barrier signaling. Full-target 8-row check PASS. Paired event medians: TRTLLM
1693.79 us, v018 3589.12 us (0.47192x). Better than v017 but still slower than
v016, so v016 remains the performance base.

NCU: 200 registers, 202040 shared bytes, occupancy 11.209%, tensor active
30.689%, eligible warps 0.22110, long scoreboard 4.00437. Local sectors zero;
shared-load/store conflicts 10590851 / 311222. Diagnostic duration 5.74925 ms.
The intended overlap does not offset extra scheduling/waiting overhead with
this two-stage design. Keep the unsuccessful variants as evidence.

## Iteration 019 — skip identity O correction exactly

File: `experiments/glm53_sparse_mla/kernel_v019.py`, based on v016.
Online softmax only rescales the running O accumulator when a tile increases
the running maximum. Test all head indices addressed by each O fragment and
use a warp-wide all vote. A warp whose factors are exactly 1 skips its eight
TMEM load/multiply/store sequences. No approximation threshold is introduced;
TMEM instructions remain warp-converged. This tests unnecessary correction
traffic as a bottleneck without changing the QK/PV scheduling. Pending checks.

### Iteration 019 result

Full-target 8-row check PASS, same errors as v016. Paired warm medians:
TRTLLM 1691.87 us, v019 3022.98 us (0.55967x). There is no measured benefit
from the exact-skip test alone. NCU: 207 registers, 193840 shared bytes,
occupancy 11.013%, tensor active 34.009%, eligible warps 0.25488, long
scoreboard 3.81900. Local sectors zero; shared-load/store conflicts
2557275 / 305320. Diagnostic duration 5.19382 ms.

## Iteration 020 — align correction and softmax head ownership

File: `experiments/glm53_sparse_mla/kernel_v020.py`, based on v019.
Use Ld/St16x32bx2 for the correction fragments so each thread owns the same
head as its softmax fragment. Multiply by the thread-local correction value;
remove shared alpha stores, reads, and their named barrier. Keep the original
Ld16x128b layout in the epilogue for coalesced global output stores. Exact
warp-uniform identity skipping remains; no numerical approximation was added.

Full-target 8-row check PASS, same errors as v016. Paired warm medians:
TRTLLM 1691.87 us, v020 3011.68 us (0.56177x). Registers drop from 207 to 130,
but runtime is essentially unchanged relative to v016. NCU: shared 193840
bytes, occupancy 10.953%, tensor active 34.196%, eligible warps 0.23839,
long scoreboard 4.27853. Local sectors zero; shared-load/store conflicts
2501426 / 185423. Diagnostic duration 5.16640 ms. With shared memory and TMEM
still restricting computing CTAs, register reduction alone does not increase
residency or explain the remaining performance gap.

## Iteration 021 — weight-stationary M64 MMA and Layout E

File: `experiments/glm53_sparse_mla/kernel_v021.py`, based on v020.
Test tcgen05.mma.ws for FP8 QK and PV. This is the hardware weight-stationary
instruction form, separate from software loader/compute warp specialization.
The PTX documentation and CUTLASS tmem_frg_ws implementation specify a 2x2
layout for M64: N halves occupy DP[0:64] and DP[64:128]. O occupies 256 columns
and S occupies another 64, within a 512-column allocation. Emit inline PTX from CuTeDSL using CuTe-generated SMEM descriptors; retain FP32
accumulation and FP8 P. Rewrite TMEM fragment layouts and output column offsets
accordingly. This is a new layout hypothesis requiring smoke, full numerical,
and memory checks; no performance claim is made before measurements.

References for this experiment:
- https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout
- https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-mma-ws
- CUTLASS include/cute/atom/mma_traits_sm100.hpp, tmem_frg_ws<M_MMA=64>.

### Iteration 021 rejected; isolated QK diagnosis

Initial descriptor generation required the MMA-vector mode to be a nested
rank-2 layout; wrapping the logical matrix layout fixed the verification error.
The installed NVVM weight-stationary op then failed in libNVVM with no useful
backend explanation. Emitting the documented instruction through DSL inline
PTX compiled successfully. Both compiler logs are retained.

The b2 complete-attention check FAILED (90.9% elements mismatched); no timing or
NCU comparison is accepted. `kernel_v021_diag.py` isolates just the first QK
tile and dumps its unscaled FP32 result. It passes against FP32 matmul with
maximum absolute error 7.63e-6. The descriptor and raw Layout-E QK values are
therefore correct for this sample. Printed copy coordinates show that automatic
TMEM tiling gives each thread TWO heads and distributes N halves across warps.
The inherited single-head max/sum/correction logic is invalid for that mapping.
This rejects v021 as a candidate; it is kept only as a reproducible failed
experiment, alongside `diagnose_v021.py` and its coordinate/result log.

## Iteration 022 — explicit warp-local head assignment for Layout E

File: `experiments/glm53_sparse_mla/kernel_v022.py`, based on v021.
Assign 16 consecutive heads to each compute warp. Use a 16x64 TMEM copy view,
then load both Layout-E N halves into each thread's registers. Each thread now
has one head and 64 score elements; XOR16 combines the two lanes per head.
Apply the same warp-local addressing to O correction and epilogue. Output
column offsets still follow Layout E's split N halves. Pending validation.

### Baseline code-generation evidence

Exported SASS directly from the existing TRTLLM NCU report, without another
GPU profiling run. It contains `UTMALDG.2D.GATHER4` for sparse KV gathering and
`UTCQMMA` for FP8 matrix operations. The former confirms TMA gather4 as a
concrete loading difference from the candidate's 16-byte cp.async instructions.
The latter mnemonic alone does not establish whether the baseline uses PTX
weight-stationary mode; no such claim is made. Full export is retained under
`artifacts/baseline_codegen`.

### Iteration 022 rejected

The new pointer arithmetic first required explicit alignment hints. After that
fix it compiled, but b2 attention still FAILED (90.4% mismatch). A QK-only dump
showed exactly 75% mismatch: warp 0 read correctly, the other three did not.
The explicit pointer offsets do not override the warp's TMEM datapath ownership;
this approach cannot simply move each warp to 16 contiguous Layout-E rows.
No full performance run or NCU result is reported for this rejected version.
`diagnose_v022.py` preserves this finding. The one-block attention diagnostic
was prepared but not executed because QK alone already disproved the mapping.

## Iteration 023 — respect Layout E and reduce across warp pairs

File: `experiments/glm53_sparse_mla/kernel_v023.py`, based on v021.
Keep the automatically partitioned, independently verified QK read layout.
Each thread processes two heads separately. XOR16 merges lanes within a warp;
shared max/sum arrays then merge the two warps owning different N halves.
Maintain two running max/sum values per thread. O correction and epilogue use
128-channel views spanning all 128 datapaths, with Layout-E channel mapping.
This corrects both single-head reduction and partial-datapath copy assumptions.

Smoke and full-target 8-row numerical checks PASS. Paired warm events: TRTLLM
1691.94 us, v023 3019.84 us (0.56027x). NCU: 121 registers, 194864 shared bytes,
occupancy 10.966%, tensor active 17.043%, eligible warps 0.28044, long scoreboard
3.11800. Local sectors zero; shared-load/store conflicts 1708850 / 468982.
Diagnostic duration 5.18675 ms. Tensor-active cycles approximately halve versus
v020 while overall runtime remains similar. This is consistent with faster MMA
execution being offset by other critical-path work, including loading and the
additional softmax synchronization; it is not a demonstrated runtime gain.

## Iteration 024 — TMA gather4 into SW64 KV buffers

File: `experiments/glm53_sparse_mla/kernel_v024.py`, based on v020 to isolate
the loading change from the weight-stationary layout experiment. Encode a
2D UINT8 tensor map over the existing FP8 KV storage, with 64-byte SW64 tiles.
A producer warp issues 32 gather4 groups for each of nine column tiles; TMA
completion bytes signal the existing full barrier. Invalid row indices use
hardware out-of-bounds zero fill and retain the softmax mask. Q loading remains
unchanged. Descriptor preparation and storage allocation occur outside timing;
the kernel still gathers only the supplied sparse indices. Validation pending.

### Iteration 024 result

CUDA Python required explicit cuuint32_t/cuuint64_t array entries when encoding
the tensor map; the original binding failure log is retained. Smoke and full
8-row checks then PASS, with the same numerical errors as v020. Paired event
medians: TRTLLM 1691.58 us, v024 8754.30 us (0.19323x), a large regression.
NCU: 130 registers, 193840 shared bytes, occupancy 11.972%, tensor active
11.727%, eligible warps 0.11834, long scoreboard 7.40876. Local sectors zero;
shared-load/store conflicts 101893 / 10852. Diagnostic duration 15.04928 ms.

SASS export shows an ELECT/R2UR/BRA loop around each TMA instruction to
serialize lane-varying coordinates. All 32 gather groups were assigned to a
single producer warp, causing costly repeated coordinate broadcasts and
uniform-register spill/fill moves. Those SASS MOV.SPILL/R2UR.FILL operations
are not local-memory spills: measured local sectors remain zero. TMA itself
is not sufficient; its instruction issue organization matters.

## Iteration 025 — distribute TMA across four producer warps

File: `experiments/glm53_sparse_mla/kernel_v025.py`, based on v024.
Select one lane in each group of four across all 128 producer threads, instead
of using only the first 32 threads. The same 32 gather4 groups are distributed
as eight per warp. All tensor maps, buffers and numerical math are unchanged.

Smoke/full 8-row checks PASS. Paired medians: TRTLLM 1692.19 us, v025 3482.69 us
(0.48589x). This improves the first TMA version by 2.51x but still loses to the
cp.async base. NCU: 130 registers, 193840 shared bytes, occupancy 11.137%, tensor
active 30.264%, eligible warps 0.34098, long scoreboard 2.01679. Local sectors
zero; shared-load/store conflicts 498476 / 2205093. Diagnostic duration 5.836 ms.

## Iteration 026 — explicit elected TMA issuer per warp

File: `experiments/glm53_sparse_mla/kernel_v026.py`, based on v025.
Elect one lane per producer warp and explicitly unroll its eight gather groups.
This avoids asking the compiler to serialize eight active lanes separately for
each of the nine TMA column instructions. Keep the four producer warps and
existing SW64 layout to isolate issue overhead. Validation pending.

### Iteration 026 result

Smoke and full 8-row checks PASS. Paired warm medians: TRTLLM 1691.84 us,
v026 3114.18 us (0.54327x). Explicit elected issue improves v025 by 1.12x,
but is still slightly slower than the best cp.async version. NCU: 130 registers,
193840 shared bytes, occupancy 10.875%, tensor active 33.113%, eligible warps
0.30700, long scoreboard 3.37771. Local sectors zero; shared-load/store
conflicts 1997049 / 708552. Diagnostic duration 5.33619 ms.

## Iteration 027 — compose uniform TMA with weight-stationary MMA

File: `experiments/glm53_sparse_mla/kernel_v027.py`, based on v023, with the
validated TMA producer from v026. Both changes individually reduce specific
instruction/cycle costs but have not improved the end-to-end best time. Test
whether faster loading exposes the faster Layout-E MMA path, rather than
assuming their benefits compose. Softmax reduction and output mapping remain
those of v023. Validation pending.

### Iteration 027 result

Smoke and full 8-row checks PASS. Paired warm medians: TRTLLM 1691.84 us,
v027 3458.37 us (0.48920x). Combining TMA with weight-stationary MMA does not
improve the candidate. NCU: 121 registers, 194864 shared bytes, occupancy
11.225%, tensor active 15.368%, eligible warps 0.40178, long scoreboard 1.32199.
Local sectors zero; shared-load/store conflicts 2473850 / 1695965. Diagnostic
duration 5.74912 ms. The reduction in some stall/cycle metrics does not imply
an end-to-end gain; measured runtime is worse than both parent designs.

## Iteration 028 — dedicated MMA warp and independent correction progress

File: `experiments/glm53_sparse_mla/kernel_v028.py`, based on v026.
Use 288 threads: 128 compute, 128 producer, and one dedicated 32-thread MMA
warp. Allocate two P buffers and two nonoverlapping S regions in the upper
TMEM lane half. The MMA warp primes QK0, issues next-QK before current PV,
waits for a compute-ready barrier, and signals stage reuse directly from PV
completion. Compute warps can run softmax and O correction while the MMA warp
waits for the next KV stage, avoiding the specific dependency exposed by
v017/v018. Producer stage reuse still waits for PV completion. The compute
warps wait for previous PV before O correction and final PV before epilogue.
This changes concurrency and requires fresh numerical and memory validation.

### Iteration 028 result and validation

The initial CuTeDSL compile reported an SSA dominance error when mutable MMA
state was shared across conditional helper calls. Creating local QK/PV MMA
objects in the issuing branch/helper fixes it; the failed compiler log is kept.
Smoke/full 8-row numerical checks PASS. Paired warm medians: TRTLLM 1689.98 us,
v028 3247.42 us (0.52041x), still slower than v026 and the best v016.
NCU: 168 registers, 202056 shared bytes, occupancy 12.445%, tensor active
34.056%, eligible warps 0.29561, long scoreboard 4.08047. Local sectors zero;
shared-load/store conflicts 8084040 / 689638. Diagnostic duration 5.19114 ms.

The b512/chunk0 test covers lengths 1 through 2045, including invalid indices
and partial final tiles. Checked rows 0 and 511 PASS numerically (max_abs
0.00618249). Default Compute Sanitizer reports 34 CUDA_ERROR_INVALID_VALUE
errors, all in CUDA Python's cuGetProcAddress_v2 feature lookup during tensor
map binding initialization; no device access error appears in that log. A
separate memcheck run with --report-api-errors no retains device memory
instrumentation and returns ERROR SUMMARY: 0 errors. Preserve both logs and
do not describe the original default run as an unconditional sanitizer pass.
An earlier misspelled --chunk-idx invocation did not instrument any API and
is retained separately as a CLI failure, not as validation evidence.

### Stable Graph and expanded target check for v016

The current best event candidate v016 passes 64 sampled rows on full b8192,
chunk3, seed1234. max_abs 0.00975490, relative_RMSE 0.01470397. TRTLLM also
passes with max_abs 0.01013046, relative_RMSE 0.01487935. Tolerances unchanged.
20 warmups/100 Graph repeats: warm TRTLLM 1873.92 us versus v016 2997.23 us
(0.62522x); cold TRTLLM 1876.53 us versus v016 3000.50 us (0.62541x).
The long run confirms the candidate latency and continuing 1.60x gap to TRTLLM;
short event ratios and Graph ratios remain separately identified.

## Iteration 029 — separate softmax and correction warpgroups

File: `experiments/glm53_sparse_mla/kernel_v029.py`, based on v028.
Use 416 threads: 128 softmax, 128 TMA producer, 128 output correction, 32 MMA.
Softmax publishes P and per-head correction to a two-stage ready barrier. The
correction group waits for previous PV, updates O, then releases current PV.
Softmax can progress to the already-issued next QK while correction is running.
Separate alpha and denominator storage prevents epilogue/last-correction aliasing.
This further tests pipeline overlap while retaining two KV/S/P stages. Pending
numerical, resource, and performance checks.

### Iteration 029 result

Smoke and full 8-row checks PASS. Paired warm medians: TRTLLM 1690.02 us,
v029 3284.10 us (0.51461x). More warpgroup separation does not improve runtime.
NCU: 128 registers, 202584 shared bytes, occupancy 17.468%, tensor active
33.733%, eligible warps 0.30565, long scoreboard 7.00844. Local sectors zero;
shared-load/store conflicts 7977472 / 47823. Diagnostic duration 5.23386 ms.
Higher occupancy here includes more waiting/synchronizing work; it is not a
throughput improvement. Keep v016 as the best measured candidate.

## Iteration 030 — split 512-dimensional main storage from 64-dimensional tail

File: `experiments/glm53_sparse_mla/kernel_v030.py`, based on v026.
Use SW128 for Q/K's first 512 dimensions and SW64 for the remaining 64. Each
four-index group now needs four 128-byte main gathers and one 64-byte tail
gather, instead of nine 64-byte gathers. Two tensor maps share the same KV
allocation; there is no expanded input copy. QK performs 16 K32 MMAs on the
main descriptors and two on the tail descriptors, preserving accumulation
order. PV consumes only the main 512-dimensional buffer. Total KV shared
storage is unchanged. This isolates TMA issue count and memory-layout effects
from the larger warpgroup pipeline experiments. Validation pending.

### Iteration 030 result

Smoke/full 8-row checks PASS with unchanged numerical errors. Paired warm
medians: TRTLLM 1691.90 us, v030 3041.60 us (approximately 0.556x). Reducing
TMA issue count helps v026 modestly but does not beat the stable v016 result.
NCU: 126 registers, 193840 shared bytes, occupancy 10.810%, tensor active
33.831%, eligible warps 0.23108, long scoreboard 4.61079; local sectors zero.
Shared-load/store conflicts 5357811 / 17526; diagnostic duration 5.221376 ms.

## Iteration 031 — ballot-based fully valid tile path

File: `experiments/glm53_sparse_mla/kernel_v031.py`, based on v030.
Producer warps ballot their valid indices into eight shared Uint32 words
(two stages). A compute thread whose two words are all ones skips the original
64 individual shared validity reads; otherwise it retains per-element masking.
This preserves internal negative-index holes, not just partial final tiles.

Smoke/full 8-row checks PASS. Paired warm medians: TRTLLM 1691.46 us,
v031 2994.08 us (0.56493x). The 2 us difference from v016 is not evidence of
a meaningful improvement. NCU: 192 registers (up from 126), 193872 shared
bytes, occupancy 10.953%, tensor active 34.414%, eligible warps 0.24699,
long scoreboard 4.30201; local sectors zero. Shared-load/store conflicts
4775440 / 17219; diagnostic duration 5.134272 ms. The extra branch reduces
validity reads but substantially increases compiler register requirements.

### Explicit holes and short-sequence numerical check

`validate_masks.py --kernel-version v031` constructs b2/chunk3 with ten internal
holes in row0 and a length129 row1 with three holes (valid counts 2038 and 126).
v031 and v020 produce bitwise identical outputs: zero unequal elements.
The FP32-reference check at the unchanged atol=0.01/rtol=0.05 FAILS for 44 of
65536 elements, max_abs 0.02133679 at row 1 / head 55 / channel 109. This establishes
that the new mask path preserves the prior computation, not that this added
short-sequence case meets tolerance. Retain the failure log under
`artifacts/v031_validation`; no tolerance was relaxed. FP8-P numerical accuracy
on short cases remains a limitation requiring a separate improvement.

## Iteration 032 — 256-column output correction chunks

File: `experiments/glm53_sparse_mla/kernel_v032.py`, based on v031.
Replace eight 64-column TMEM correction chunks with two 256-column chunks,
using Ld/St16x32bx2 Rep128. This reduces load/store wait boundaries while
raising the correction fragment to 128 FP32 values per thread. The final
coalesced epilogue is unchanged.

Smoke/full 8-row checks PASS. Paired warm medians: TRTLLM 1691.65 us,
v032 3031.30 us (0.55806x). NCU: 255 registers and local-load/store sectors
30408704 / 13269340, demonstrating compiler spills. Shared 193872 bytes,
occupancy 10.916%, tensor 33.978%, eligible 0.23941, long scoreboard 4.33177.
Shared conflicts 4974824 / 17533; diagnostic 5.196512 ms. Larger chunks do not
improve runtime and are not selected.

## Iteration 033 — isolate correction size from mask-branch pressure

File: `experiments/glm53_sparse_mla/kernel_v033.py`, based on v032.
Remove the ballot/fully-valid branch and restore v030 elementwise validity
checks while retaining 256-column correction chunks. This isolates whether
the mask branch caused the large-chunk spill regression.

Smoke/full 8-row checks PASS with the same errors as v030. NCU still reports
255 registers and local sectors 33685504 / 18082152; the large correction
fragments spill even without the mask branch. Shared 193840 bytes,
occupancy 10.951%, tensor 34.899%, eligible 0.24119, long scoreboard 4.36837,
shared conflicts 5242053 / 24798, diagnostic 5.091104 ms.

**Timing contaminated by another GPU workload:** paired event medians jump
to TRTLLM 4419.74 us and v033 8477.82 us; baseline p05 is 2242.16 us. After our
commands completed, two read-only checks at 18:14:50/18:15:14 pod time found
physical GPU1 at 100% utilization and 257665 MiB used, while this container
had no Python/NCU process and NVML exposed no owning process. Treat this as
non-isolated timing; do not rank its latency or interpret its profiled
utilization against prior isolated runs. Preserve the raw evidence. GPU
benchmarking waits for an available authorized GPU1; no other GPU or process
is modified.

## Iteration 034 — two compute warpgroups per tile

File: `experiments/glm53_sparse_mla/kernel_v034.py`, based on v030.
Use 384 threads: two 128-thread compute groups and the original 128-thread TMA
producer. Each compute group owns all 64 heads and half of each 128-key score
tile, reducing its score fragment to 32 FP32 values/thread. Two shared
reductions merge head maxima and sums across the groups. Output correction
and epilogue each split 512 channels into two 256-channel halves, retaining
small 64-column TMEM copies. This trades two extra compute barriers for less
per-thread softmax/correction work and more independently schedulable warps.
The existing TMEM datapath ownership repeats across four-warp groups; only
column offsets differ. Numerical/resource/performance validation is pending
GPU1 availability. This file is not yet a validated candidate.

### Offline compilation while GPU1 is occupied

The new `compile_offline.py` uses CuTe fake descriptors matching full b8192
shapes/strides and `CUDA_VISIBLE_DEVICES=""`, with explicit SM103a. It allocates
no input tensors and never calls the compiled function. v034 compiles
successfully; cuobjdump reports REG90, STACK0, LOCAL0. The static SHARED1024
field excludes dynamically allocated buffers and is not the total CTA shared
memory size. Numerical and runtime checks remain pending. The GPU preflight
correctly refused the v034 timing/profiling command while 257665 MiB remained
allocated on GPU1; its log is preserved with the offline compilation evidence.

## Iteration 035 — TMEM probabilities with matching A/D lane alignment

File: `experiments/glm53_sparse_mla/kernel_v035.py`, based on v030.
The preserved TRTLLM SASS contains PV-like MMA groups using a `tmem[...]`
A operand (for example lines 3974–3978), while our candidates use SMEM A.
This is a concrete data-path difference; the mnemonic alone still does not
identify the PTX .ws qualifier or the full baseline algorithm.

Use explicitly interleaved M64/N256 output fragments: the two N tiles occupy
opposite 16-lane halves at columns [0,256), instead of 512 columns in one half.
QK scores move to the lower lane half at [256,384). FP8 probabilities are packed
into [384,416), duplicated across both lane halves; each PV N tile reads the
copy matching its output lane alignment. The shared P buffer and its stores
are removed. Correction/epilogue use the corresponding interleaved output
addresses. MMA order within each output element and softmax math are unchanged.

[NVIDIA PTX data-path layout documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout)
requires matching A/D lane alignment for Layout F. Therefore simply placing
P in the previously unused opposite lane half would be invalid. This version
explicitly reserves disjoint S/P/O storage within 512 allocated TMEM columns.
Offline layout/compile checks and GPU numerical/memory/performance validation
are pending; no speedup is claimed.

### Iteration 035 compile and layout audit corrections

The initial compile failed because slicing only B's N mode left A/B with
different ranks. Indexing the singleton M modes explicitly fixes the call.
The first successful compile used 126 registers, zero stack/local bytes, and
its probability packing audit matched all 128 threads × 64 logical values.
However, inspecting the emitted output layout revealed that TMEM-A mode's
default C fragment is NON-interleaved (N-tile stride 256), unlike the SMEM-A
fragment used earlier. That would overlap the planned S/P region. Before
any GPU launch, set the N-tile stride explicitly to 16<<16 and assert the
output's physical column footprint is 256. The audit now checks this bound
in addition to every score/probability coordinate. Preserve the initial
compiler error and initial successful compile/audit logs, distinguished from
the corrected candidate; initial compilation was not runtime validation.

### Explicit cross-thread tcgen05 ordering

The installed CuTe `fence_view_async_tmem_load/store` emits tcgen05.wait,
which tracks completion. It does not insert the before/after-thread-sync
fences described in the [PTX synchronization patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-consistency-model).
The emitted prototype PTX confirmed those fences were absent. Add explicit
before/after fences around compute named barriers and an after fence after
MMA-completion mbarrier waits in pending v034/v035. This makes cross-thread
TMEM dependencies explicit; both versions need fresh compile and runtime
validation. Earlier sampled numerical passes remain historical evidence,
not proof that missing ordering is safe for all schedules/compiler versions.

### Corrected offline results and next GPU checks

Corrected v034 SHA256:
`3af357d415de853e5557de0833df313bdc5828e90efcd207e4030351c4a45f3e`.
Corrected v035 SHA256:
`92bc7ab5a0a8af6e806c09e2d60859ea711b1b1d3ebb80febc6ed5aba70fc2bd`.
Both compile with GPU visibility disabled. Static resources remain 90 and 126
registers respectively, with STACK0/LOCAL0. PTX extracts preserve the explicit
fences and TMEM instructions. The corrected v035 audit passes 8192 element
coordinate comparisons and the 256-column O bound; its printed N-tile stride
is 1048576 (16<<16), as intended. These checks establish compiler/layout
properties only. Full GPU validation is still pending because GPU1 remains
occupied by an apparent external workload.

After GPU1 is available, the next bounded sequence is: preflight; b2 smoke
checks for v034/v035; full b8192 paired events plus NCU for numerical survivors;
expanded 64-row warm/cold Graph checks and device memcheck for any candidate
that improves latency. Preserve original FP32 tolerances and explicitly
retain the existing short-sequence numerical limitation. Re-run v033 timing
only if it becomes useful for comparison; its spilled correction path is not
currently preferred.

### GPU1 availability and iteration 034/035 measured results

GPU1 became idle again at 18:36:07 pod time (two samples: utilization 0%,
allocated memory 0 MiB); isolated GPU work resumed under the original UUID.
Both corrected candidates pass b2 smoke and full-target 8-row checks with the
original tolerances, max_abs 0.006737709 and relative_RMSE 0.014866708.

v034 paired warm events: TRTLLM 1693.82 us versus candidate 2878.69 us,
approximately 0.5884x. This is about 4% faster than the previous 3.0 ms best.
NCU: registers 90, shared 194864 bytes, occupancy 17.190%, tensor 35.800%,
eligible 0.35235, long scoreboard 5.88091, local sectors 0/0, shared conflicts
5119552/191873, diagnostic 4.947520 ms. Splitting compute work improves
end-to-end time even with additional cross-group reductions.

v035 paired warm events: TRTLLM 1691.90 us versus candidate 3045.25 us,
approximately 0.5556x. NCU: registers 126, shared 185648 bytes, occupancy 10.813%,
tensor 33.832%, eligible 0.23408, long scoreboard 4.63940, local sectors 0/0,
shared conflicts 5675146/11780, diagnostic 5.223104 ms. Removing shared P saves
8192 shared bytes but does not improve runtime by itself. Keep this as a
validated layout/data-path building block rather than the best candidate.

### Iteration 034 expanded Graph and device memory validation

Full b8192/chunk3/seed1234 passes 64 checked rows, max_abs 0.009754896 and
relative_RMSE 0.014703961. TRTLLM passes with max_abs 0.010130458. The original
atol=0.01/rtol=0.05 criteria are unchanged and this remains sampled validation.
20 warmups/100 Graph repeats: warm v034 2879.57 us versus TRTLLM 1878.14 us;
cold v034 2883.65 us versus TRTLLM 1870.34 us. The stable run confirms an
approximately 4% improvement over v016, while TRTLLM is still about 1.53x faster.

Compute Sanitizer memcheck on b512/chunk0 (lengths 1..2045), checked rows 0/511,
passes numerical comparison (max_abs 0.006182492) and reports 0 device memory
errors. `--report-api-errors no` is explicitly used to bypass the previously
identified CUDA Python feature-lookup API errors, while preserving device
memory instrumentation. It is a qualified device-memory pass, not a claim
that the default sanitizer mode or every possible shape was validated.

## Iteration 036 — combine two compute groups with TMEM probabilities

File: `experiments/glm53_sparse_mla/kernel_v036.py`, based on v035 with v034's
compute partition. Each of two compute groups handles 64 score columns and
256 output channels. Packed probabilities use 16 TMEM columns per group and
are duplicated into matching A/D lane halves. Each output group owns one
interleaved N256 fragment. Retain explicit tcgen05 fences and the cross-group
head-max/sum reductions. This tests whether the smaller TMEM store fragments
and broader compute distribution make the TMEM-P path worthwhile; v035 alone
was not faster, so composition must be measured. Validation pending.

## Iteration 037 — defer cross-group denominator reduction

File: `experiments/glm53_sparse_mla/kernel_v037.py`, based on v034.
Both compute groups use the same per-tile running maximum and correction.
Therefore each can update only its own denominator contribution throughout
the tile loop; the full denominator is their sum at exit. Remove per-tile
shared head-sum exchange and its barrier, and perform that reduction once
before epilogue. Reuse the now-dead head-max storage for the final partial
sums, saving 512 shared bytes. This preserves the real-arithmetic softmax
formula but changes FP32 summation order, requiring fresh numerical checks.
The per-tile maximum exchange and all required TC memory ordering remain.
Validation pending.

### Iteration 036 result

The two-group compile-time probability-coordinate audit passes; smoke and
full-target 8-row numerical checks pass with unchanged errors. Paired warm
medians: TRTLLM 1691.74 us, v036 2926.85 us (approximately 0.5780x), slower than
v034. NCU reports only 80 registers but local-load/store sectors 8912896 /
13117008. Lower register count alone is not evidence of a better implementation.
Shared 186672 bytes, occupancy 17.168%, tensor 35.093%, eligible 0.34443,
long scoreboard 6.16955, shared conflicts 5365693/115454, diagnostic 5.033600 ms.
Continue from v034's shared-probability path for the next latency experiments.

## Iteration 038 — one-head Layout-E softmax/correction and weight-stationary MMA

File: `experiments/glm53_sparse_mla/kernel_v038.py`, based on v037, using the
validated inline .ws MMA helper from v023. Unlike v023's two-head/thread
Ld16x32bx2 mapping, use Ld32x32b Rep32 on a physical 128-DP × 32-column view.
Each compute thread then owns one head and 32 keys. The two compute groups
cover complementary physical column halves; four partial maxima per head
are merged once per tile. Denominator contributions remain independent until
the final reduction, as in v037. Correction uses the same one-head mapping;
epilogue retains v023's coalesced Layout-E copy.

QK uses the SW128 main 512 plus SW64 tail 64 layouts and TMA producer from v030.
Both QK and PV use hardware weight-stationary MMA. The goal is to expose its
lower tensor-cycle cost while avoiding the two-head bookkeeping and repeated
head-sum synchronization that limited v023/v027. TMEM O occupies columns 0..255
across all datapaths, S 256..319; P remains in shared memory. This changes both
layout and reduction order and requires fresh numerical/memory validation.
No performance result is yet claimed.

### Iteration 037 short-run result

Smoke/full 8-row checks PASS. Full-target max_abs remains 0.006737709;
relative_RMSE is 0.014866682 after the changed denominator summation order.
Paired warm medians: TRTLLM 1691.78 us, v037 2777.12 us (approximately 0.6092x).
This improves v034 by approximately 3.5%. NCU: registers 114, shared 194352 bytes,
occupancy 17.161%, tensor 37.309%, eligible 0.34573, long scoreboard 6.18059,
local sectors 0/0, shared conflicts 5777572/382800, diagnostic 4.737184 ms.
Expanded Graph and device-memory validation is in progress.

### Iteration 037 expanded validation

Full b8192/chunk3/seed1234 passes 64 sampled rows, max_abs 0.009754896 and
relative_RMSE 0.014703972, unchanged tolerances. 20 warmups/100 Graph repeats:
warm candidate 2777.20 us versus TRTLLM 1879.97 us; cold candidate 2781.15 us
versus TRTLLM 1874.18 us. This confirms the short-run improvement, with TRTLLM
still approximately 1.48x faster in the warm Graph run.

The b512/chunk0 device memcheck, using the explicitly qualified
`--report-api-errors no` setting, reports 0 device memory errors. Checked
rows 0/511 pass numerical comparison with max_abs 0.006182492. This checks
partial tiles and the reused shared denominator storage on that input, not
all possible masks or shapes. v037 is the current best validated candidate.

### Iteration 038 rejection and iteration 039 address correction

v038 offline compilation passes with REG122/STACK0/LOCAL0 and its score
coordinate audit covers every head/key. However, the first b2 GPU check
fails: 30151/65536 mismatches, max_abs 0.604953766. No performance/NCU run
was accepted for this numerically invalid candidate. Inspecting the earlier
v023 epilogue reveals the cause: each N256 Layout-E tile puts channels
0..127 and 128..255 into opposite 64-DP halves. A 64-column epilogue slice
therefore contains non-contiguous channel blocks, not 128 consecutive channels.

v039 changes only that output address: group*256 + tile*64 + (d//64)*128 + d%64.
Extend audit_ws_layout.py to compare each epilogue physical address with the
N256 MMA address and verify complete, unique coverage of all 64*512 outputs.
The score/correction mapping and algorithm remain unchanged. GPU validation
and profiling are pending. Preserve v038's failure rather than counting it
as a performance candidate.

### Iteration 039 validation and profiling

The corrected output mapping passes all coordinate audits and b2/full-target
8-row numerical checks. Paired warm events: candidate 2328.74 us versus TRTLLM
1691.81 us, approximately 0.7265x. NCU: 118 registers, 194864 shared bytes,
occupancy 17.071%, tensor active 22.316%, eligible 0.43145, long scoreboard
4.52277, local sectors 0/0, shared load/store conflicts 4421204/1777016,
diagnostic 3.962400 ms. Weight-stationary MMA changes tensor-cycle accounting;
its lower tensor-active percentage versus v037 does not mean lower useful
throughput. End-to-end latency improves approximately 16%.

Full b8192 Graph validation (20 warmups/100 repeats/64 checked rows) passes
with max_abs 0.009754896 and relative_RMSE 0.014703953. Warm: 2332.50 us versus
TRTLLM 1872.05 us; cold: 2338.51 us versus TRTLLM 1912.77 us. The warm gap is
now approximately 1.25x. b512/chunk0 device memcheck with the explicitly
qualified --report-api-errors no setting reports zero device errors, and
rows 0/511 pass with max_abs 0.006182492. v039 becomes the best validated
candidate; this remains sampled validation within the existing scope.

## Iteration 040 — packed output correction multiplication

Based on v039, change only output correction's scalar FP32 multiplies into
explicit cute.arch.mul_packed_f32x2 pairs. FlashInfer's sparse blk128
flash_fwd_sm100.py uses this operation in its correction path. The
[PTX packed FP32 multiplication specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul)
keeps each element's FP32 rounding; this introduces no approximation threshold.
The aim is fewer CUDA-core instructions while retaining the same TMEM
load/store sizes and synchronization. Numerical checks, code-generation
inspection and performance profiling are pending.

### Iteration 040 result

Smoke and full-target 8-row checks pass with unchanged numerical errors.
Paired events: candidate 2326.69 us versus TRTLLM 1691.78 us. NCU reports
118 registers, shared 194864 bytes, occupancy 17.066%, tensor active 22.305%,
eligible 0.43163, long scoreboard 4.52265, local sectors 0/0, shared conflicts
4418255/1801227, diagnostic 3.964032 ms. This is effectively tied with v039;
explicit packed multiplication alone has no demonstrated meaningful gain.

## Iteration 041 — one store-completion wait per correction phase

Based on v040, move tcgen05.wait::st from every disjoint output correction
chunk to the end of that correction loop. The consumer PV MMA still follows
the completion wait, before-thread-sync fence, full compute barrier and
after-thread-sync fence. Every TMEM load remains followed by wait::ld before
its registers are consumed. FlashInfer's correction loops also batch stores
with one completion wait at the end. No matrix layout or softmax math changes.
This tests whether four serialized store waits contribute to latency; checks
and NCU are pending.

## Iteration 042 — widen one-head correction fragments

Based on v041, use Ld/St32x32b Rep64 for output correction: each thread handles
64 contiguous values per load instead of 32, and each group makes two chunks
instead of four. The score loader, P stores and coalesced output epilogue are
unchanged. The previous non-.ws v032/v033 attempt needed 128 correction values
per thread and incurred heavy local traffic; this version needs only 64, but
register/local-traffic checks remain essential. Goal: amortize TMEM load waits
without repeating that register-pressure regression. Validation pending.

### Iteration 041 result

Smoke/full-target 8-row checks pass with unchanged errors. Paired warm median
2316.58 us versus TRTLLM 1691.84 us: a small approximately 0.4% improvement,
not yet an independently established stable gain. NCU: 118 registers, 194864
shared bytes, occupancy 17.066%, tensor 22.411%, eligible 0.43267, long
scoreboard 4.52469, local sectors 0/0, shared conflicts 4429451/1812978,
diagnostic 3.947232 ms.

## Iteration 043 — overlap next softmax with current PV

Based on v041, retain two compute groups and the one-head .ws layout, but
prime QK0 and submit QK(next) before PV(current). Separate QK/PV completion
barriers allow the next softmax to execute while the preceding PV finishes.
Wait for that preceding PV before correcting its O accumulator. Two shared
P buffers protect outstanding PV reads; a single TMEM S buffer remains safe
because all score readers have passed the pre-PV compute barrier before the
next QK overwrites it. The producer's empty-stage release is now a tcgen05
completion commit after PV, rather than a manual arrival after an all-thread
PV wait. The final PV is explicitly drained before epilogue.

This revisits v028's overlap hypothesis with the faster .ws data path and
without adding an independent MMA warp. Required before/after-thread-sync
fences remain, and QK/PV phases now each advance once per tile. This is a
synchronization change requiring smoke, full-target, expanded and device
memory validation before promotion. Compile/runtime/profiling pending.

### Iteration 042 result

Smoke/full-target 8-row checks pass, but latency regresses to 2367.33 us versus
TRTLLM 1689.82 us. NCU: 168 registers, 194864 shared bytes, occupancy 17.011%,
tensor 21.980%, eligible 0.41415, long scoreboard 4.59918, local sectors
6291456/1583320, shared conflicts 5314480/1549018, diagnostic 4.024064 ms.
The wider fragment introduces local traffic and is not promoted.

### Iteration 043 result

Smoke/full-target checks pass with unchanged errors, but paired warm latency
2712.74 us versus TRTLLM 1691.68 us regresses substantially. NCU: 110 registers,
203064 shared bytes, occupancy 17.244%, tensor 20.817%, eligible 0.38426, long
scoreboard 5.09995, zero local sectors, shared conflicts 8530638/980345,
diagnostic 4.244512 ms. The intended overlap is insufficient to offset its
changed scheduling/wait behavior. Retain the evidence, return to v041.

### v039 instruction-level sampling findings

NCU SourceCounters plus WarpStateStats were collected separately, preserving
the SASS/source CSV. The first command omitted profile_ncu.py's required
--backend argument and exited before profiling; both the command-error log
and the corrected run are retained. The largest long-scoreboard samples
occur at branches consuming SYNCS.PHASECHK.TRYWAIT results, including the
QK-completion wait and producer empty-stage wait. This metric must not be
interpreted simply as HBM load latency. v039 already emits FMUL2 in correction
(for example CSV lines 1003/1023/1043/1063), explaining why explicitly requesting
packed multiplies in v040 has little effect. Its epilogue also expands into
many scalar division/refinement and STG.E.U16 instructions.

## Iteration 044 — compute normalization reciprocal once per head

Based on v041, store 1/(256*total_sum) in the final shared denominator slot
once per head, then normalize all output values with a multiplication.
Previously every output value used a division by the same head denominator;
the emitted SASS contains repeated reciprocal refinement sequences. This
reduces 512 divisions per head to one without changing the attention formula.
FP32 evaluation order changes slightly, so unchanged-tolerance numerical
validation is required. No approximate reciprocal intrinsic is introduced.
Validation/profiling pending.

### Iteration 044 result and expanded validation

Smoke/full-target 8-row checks pass: max_abs 0.006737709, relative_RMSE
0.014866707. Paired warm events 2277.73 us versus TRTLLM 1691.84 us.
NCU: 120 registers, 194864 shared bytes, occupancy 17.152%, tensor 22.839%,
eligible 0.39933, long scoreboard 5.17866, local sectors 0/0, shared conflicts
4439682/1780919, diagnostic 3.875360 ms.

64-row Graph validation passes with max_abs 0.009754896 and relative_RMSE
0.014703941. Warm: 2283.34 us versus TRTLLM 1844.90 us; cold: 2289.49 us versus
TRTLLM 1896.56 us. The candidate is about 2.1% faster than v039's Graph run;
the paired TRT result and its quantiles are preserved because clocks are not
locked. Qualified b512/chunk0 device memcheck reports zero device errors and
rows 0/511 pass at max_abs 0.006182492. v044 is the new best validated version.

The offline epilogue-coordinate inspection shows each thread owns two heads
and strided columns (e.g. thread0: (head0,col0), (head8,col0), (head0,col4),
(head8,col4), ...). Therefore simply replacing scalar output stores with a
contiguous vector store would be incorrect; a real layout redistribution
is required.

## Iteration 045 — stage BF16 output for vectorized global writes

Based on v044, after the final PV and compute barrier, reinterpret the first
64 KiB KV-main stage as a SW128 BF16 [64,512] output buffer. All KV reads have
completed and the producer has no further payload writes. Load TMEM output
using the already validated one-head physical [128,32] copy, normalize and
convert groups of 32 values, and write 128-bit BF16 vectors to shared memory.
After a compute barrier, redistribute contiguous groups of eight BF16 values
across threads and issue 128-bit global stores. This avoids pretending the
old epilogue's per-thread strided coordinates are contiguous.

The alias footprint is statically checked against the old KV stage. No
additional shared allocation is introduced. The extra shared round trip and
barrier may offset the better global transaction pattern; this is a measured
hypothesis, not a claimed gain. Numerical/memory validation and NCU pending.

### Iteration 045 short-run result

Smoke and full-target 8-row checks pass with the same errors as v044. Paired
warm median 1988.80 us versus TRTLLM 1691.94 us: approximately 12.7% faster
than v044. NCU: 122 registers, 194864 shared bytes, occupancy 17.862%, tensor
26.330%, eligible 0.45347, long scoreboard 4.85273, local sectors 0/0, shared
conflicts 4321759/1935330, diagnostic 3.356544 ms. The extra shared staging
round trip is outweighed by the improved output transaction/instruction
pattern. Expanded numerical/Graph/device-memory validation is running.

## Iteration 046 — eight TMA producer warps

Based on v045, increase CTA size from 384 to 512 threads while retaining
256 compute threads. Eight producer warps each issue four gather4 groups,
instead of four warps each issuing eight groups. The first 128 producer
threads still load the 128 sparse indices and initial Q; all 256 producers
participate in the named producer barriers. Gather coverage is unchanged:
warp-local row = (warp-8)*16 + group*4, covering rows 0..127 exactly once.
No additional shared allocation or numerical operation is introduced.
This tests reduction of per-warp gather descriptor/issue serialization
against the extra active-warp/register costs. Validation/NCU pending.

### Iteration 045 expanded validation

64-row Graph checks pass with unchanged max_abs 0.009754896 and relative_RMSE
0.014703941. Warm candidate 1992.83 us versus TRTLLM 1874.18 us; cold candidate
1998.91 us versus TRTLLM 1927.30 us. Qualified b512/chunk0 device memcheck
reports zero errors, and rows 0/511 pass unchanged tolerances. This validates
the shared KV/output alias on that test. v045 becomes the best validated
version. The warm Graph gap is about 6.3%; the short-event gap is larger,
so these measurement regimes must remain separately reported.

## Iteration 047 — load Q through five tiled TMA transfers

Based on v045, replace the initial 2304 per-thread 16-byte Q copies per CTA
with four 128x64-byte main-dimension TMA tiles and one 64x64-byte tail tile.
New Q tensor-map descriptors use the unchanged global 576-byte row stride
and 64-row boxes, preserving the SW128 main / SW64 tail shared layout. A
dedicated transaction barrier tracks all 36864 bytes and is drained before
compute begins. Descriptor construction occurs once in runner setup, just
like the existing KV maps; inputs are not expanded or repacked.

The v039 source profile attributes substantial excessive shared wavefronts
to the initial LDGSTS sequence. This change targets that startup cost and
instruction count while preserving all mainloop arithmetic. The map storage
grows from 256 to 512 bytes; offline compiler descriptors now follow the
module's TENSOR_MAP_BYTES constant. Layout/runtime checks and NCU pending.
Reference: [PTX tensor tile copies](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).

### Iterations 046 and 047 measured results

Both pass b2 and full-target 8-row checks with unchanged v045 errors.
v046 paired median 1976.22 us versus TRTLLM 1691.87 us. NCU: 120 registers,
194864 shared bytes, occupancy 23.342%, tensor 26.573%, eligible 0.51079,
long scoreboard 5.82319, local sectors 0/0, shared conflicts 3544225/3347238,
diagnostic 3.324192 ms. Extra producer warps give a small approximately 0.6%
short-run improvement; increased occupancy alone does not establish a gain.

v047 paired median 1972.42 us versus TRTLLM 1691.81 us. NCU: 122 registers,
194872 shared bytes, occupancy 17.861%, tensor 26.509%, eligible 0.45579,
long scoreboard 5.00835, local sectors 0/0, shared conflicts 4474251/1903455,
diagnostic 3.334432 ms. Q TMA gives a small approximately 0.8% short-run
improvement; expanded checks will be applied to the combined survivor.

The v045 source profile confirms vectorized output code generation:
1048576 dynamic warp STG.E.128 instructions, versus v039's 8388608 STG.E.U16.
Static SASS rows also shrink from 4040 to 1672 (including normalization changes).
This directly supports the output-path diagnosis rather than relying solely
on aggregate memory counters. Source sampling still concentrates at MMA and
producer-stage completion waits.

## Iteration 048 — compose Q TMA and eight producer warps

Based on v047, apply the same eight-warp KV-gather partition as v046. This
tests whether the two independently small gains compose. Q still completes
before all threads start the mainloop. Validation pending.

## Iteration 049 — overlap initial Q and KV transfers

Based on v048, let compute warp0 issue Q TMA while producer warps immediately
start filling KV stages. Only compute threads wait for the Q transaction
barrier, before consuming Q. Remove the post-Q all-CTA barrier; the initial
barrier following mbarrier initialization and TMEM allocation remains. This
allows the independent Q/KV transfers to overlap at startup, with unchanged
mainloop/softmax/epilogue behavior. Correctness and memory checks are required
for this synchronization change; no performance claim yet.

### Iteration 048/049 short-run results

Both pass smoke and full-target 8-row checks with unchanged v045 errors.
v048 paired warm median 1951.87 us versus TRTLLM 1691.78 us. NCU: 126 registers,
194872 shared bytes, occupancy 23.328%, tensor 26.882%, eligible 0.50284,
long scoreboard 5.99545, zero local sectors, shared conflicts 3529134/2546646,
diagnostic 3.285472 ms. The two small improvements compose.

v049 paired warm median 1902.72 us versus TRTLLM 1689.82 us. NCU: 126 registers,
194872 shared bytes, occupancy 23.272%, tensor 27.436%, eligible 0.51447,
long scoreboard 5.72411, zero local sectors, shared conflicts 3662030/2344613,
diagnostic 3.219232 ms. Initial Q/KV overlap reduces a further approximately
2.5%. Expanded Graph and qualified device-memory checks are running.

## Iteration 050 — TMA stores from the shared output buffer

Based on v049, replace the shared-to-register-to-global vector-store loop
with eight 64x64-element BF16 TMA stores directly from the same SW128 shared
output buffer. Each writer fences generic shared stores to the async proxy
before the compute barrier. An elected thread then submits the eight tiles,
commits the bulk group and waits for full completion before the final barrier
and exit. The new UINT16 output tensor map preserves BF16 bits and is created
once in runner setup; total map storage is 640 bytes.

Instruction syntax follows CUTLASS SM90_TMA_STORE_2D in
cute/arch/copy_sm90_tma.hpp. Use full wait_group 0 (not only the .read variant)
for this initial candidate. The goal is to remove the extra shared loads and
warp global-store instructions while retaining the validated output layout.
Compilation, numerical checks, device memory checks and NCU are pending.

### Iteration 049 expanded validation

64-row numerical checks pass with max_abs 0.009754896 and relative_RMSE
0.014703941. Graph warm: candidate 1911.01 us versus TRTLLM 1875.97 us; cold:
candidate 1911.07 us versus TRTLLM 1923.18 us. Warm is about 1.9% slower and
cold is approximately tied; the short-event regime still shows about a 13%
latency gap, so no blanket speedup claim is made. Qualified b512/chunk0
device memcheck reports zero errors and rows 0/511 pass unchanged tolerances.
v049 is the current best expanded-validated candidate.

## Iteration 051 — bounded softmax scaling anchor

Based on v049, retain the old exponential scaling anchor when the current
tile maximum is at most 0.75 above it in log2 units. Otherwise update the
anchor and rescale O/denominator exactly as before. The anchor need not be
the exact running maximum for the real-arithmetic softmax identity: both
numerator and denominator use the same exponential shift. The chosen bound
ensures P*256 <= 256*2^0.75, approximately 430.54, below E4M3FN's finite 448
limit. No probability is discarded and no tolerance is loosened.

This may let more warps skip output correction with the existing exact
correction==1 check. It changes FP8 quantization and FP32 accumulation order,
so it is an accuracy experiment as well as a performance experiment. First
run the original smoke/full-target checks; reject failures, and require
expanded validation before promotion. Numerical/NCU results pending.

### Iteration 050 result

Smoke/full-target 8-row checks pass with unchanged v049 errors. Paired warm
median 1908.67 us versus TRTLLM 1691.49 us: no improvement over v049. NCU:
126 registers, 194872 shared bytes, occupancy 23.263%, tensor 27.380%,
eligible 0.50750, long scoreboard 5.83432, zero local sectors, shared conflicts
3611791/2408662, diagnostic 3.228352 ms. TMA output stores are not promoted.

### Iteration 051 initial accuracy/performance tradeoff

Smoke/full-target 8-row checks pass unchanged tolerances, but errors increase:
full max_abs 0.011151507 and relative_RMSE 0.016216585 versus v049's
0.006737709 / 0.014866707. Paired warm 1892.42 us versus TRTLLM 1691.74 us is
only about 0.5% faster than v049. NCU: 126 registers, 194872 shared bytes,
occupancy 23.297%, tensor 27.589%, eligible 0.51497, long scoreboard 5.78050,
zero local sectors, shared conflicts 3521720/2365347, diagnostic 3.202656 ms.
Do not promote this small gain with increased quantization error; expand
numerical checking to characterize the tradeoff and continue from v049.

## Iteration 052 — one warp waits for MMA completion

Based on v049, only warp0 performs the QK/PV completion mbarrier waits. It
propagates completion to all compute threads through a named barrier with
explicit tcgen05 before/after-thread-sync fences. PV reuses the existing
post-PV compute barrier; QK adds a compute barrier before score reads. This
changes scheduling of waits, not arithmetic or layouts. It tests whether
reducing the number of mbarrier waiters helps the sampled wait bottleneck
enough to offset the extra QK barrier. Validation and profiling pending.

### Expanded 512-row accuracy failure: v049 and v051

Full-target seed1234, original tolerances, 512 selected rows: v051 FAILS
2/16777216 elements, greatest failing absolute difference 0.013399132 at
selected-row index354/head15/channel293. The v049 control also FAILS
1/16777216 elements, difference 0.012872629 at selected-row index112/head33/
channel458. TRTLLM PASSES the identical 512-row check (overall max_abs
0.014106080, relative_RMSE 0.015002753; tolerance is combined absolute/relative).
Do not mislabel these failures as passes or broaden the earlier 64-row claims.
The known full-target precision limitation now takes priority over further
small timing gains. v051 is rejected; v049 remains a timing reference with
a documented 512-row numerical failure, not a generally validated solution.

## Iteration 053 — residual FP8 probabilities for a second PV term

Based on v049, represent each scaled probability x as hi=FP8(x) and
lo=FP8(x-FP32(hi)). Store both FP8 matrices and accumulate their two PV
products into the same FP32 output. Q/K/V and both tensor-core PV inputs
remain FP8; no reference/dense fallback or higher-precision KV expansion is
used. The denominator and original tolerance remain unchanged.

The second FP8 term recovers most probability-rounding error instead of
tuning a scale to one observed outlier. It costs 8 KiB shared memory and a
second set of PV MMAs; register use, latency, and 512-row/multiple-seed
accuracy must be measured. The priority is a robust numerical candidate;
subsequent work can reduce this added cost. Validation pending.

### Iteration 052 scheduling result

Smoke/full-target 8-row checks pass with unchanged v049 errors. Paired warm
median 1894.46 us versus TRTLLM 1691.90 us, a small approximately 0.4% gain.
NCU: 112 registers, 194872 shared bytes, occupancy 23.293%, tensor 27.579%,
eligible 0.52178, long scoreboard 3.49455, zero local sectors, shared conflicts
3126336/2487041, diagnostic 3.204992 ms. Reducing the number of waiters shifts
stall accounting substantially without a comparable latency gain. This
does not address the probability-quantization failure found in v049; keep
it only as a scheduling building block for the accuracy repair.

### Iteration 053 initial compile correction

The first compile rejects subtraction between the score TensorSSA shape
(32,1) and the flat packed-probability shape (32). No GPU kernel ran. Make
the FP8 register fragment inherit the score fragment's shape so conversion
back to FP32 preserves the arithmetic profile. Preserve the initial compile
log; numerical/performance validation still pending for the corrected file.

## Iteration 054 — use the full E4M3 finite probability range

Based on v049, change only probability scale and matching output denominator
from 256 to 448. This is motivated by concrete baseline evidence: preserved
TRTLLM SASS line2495 selects 8.807354927 (log2(448)) immediately before
pre-exponential FMAs, and line2874 selects 448 as a scaling value. The
predicates/full baseline algorithm are not reconstructed, so this does not
claim exact TRTLLM numerical equivalence.

The standard E4M3 finite maximum uses more available range without the second
PV of v053. It changes rounding-bin alignment and may or may not eliminate
rare tolerance failures; no scale search tuned to the failing element will
be performed. Require 512-row and additional-seed checks before judging it.
Keep v053's residual path as the robust precision option while this is tested.

### Iteration 053 corrected results

Smoke and full-target 8-row checks pass. Full max_abs improves to 0.001884818
and relative_RMSE to 0.001690517. Paired warm events: 2316.42 us versus TRTLLM
1687.81 us. NCU: 101 registers, 203064 shared bytes, occupancy 23.313%, tensor
32.942%, eligible 0.49948, long scoreboard 6.86680, zero local sectors, shared
conflicts 3469229/2488035, diagnostic 3.943296 ms. The second PV costs about
22% versus v049's short run, while removing most probability-rounding error.

The original failing 512-row/seed1234 case now PASSES: max_abs 0.003831148 and
relative_RMSE 0.001693299, versus TRTLLM 0.014106080 / 0.015002753. This is
sampled evidence, not an all-shapes accuracy guarantee. Paired Graph20/100
warm candidate 2356.21 us versus TRTLLM 1861.66 us; cold 2332.77 us versus
1939.60 us. The expanded Graph table now shows sampled row counts explicitly.
Additional seed, short-sequence/mask and device-memory checks are running.

### Iteration 053 additional validation

Qualified b512/chunk0 device memcheck: zero errors; two checked rows pass
with max_abs 0.001525640. The previously failing holes/partial-tile case now
passes the independent FP32 check: valid counts 2038/126, max_abs 0.003903151.
validate_masks.py gains --reference-only for precision-changing candidates;
the default historical bitwise-v020 mode is unchanged.

Full-target seed5678, 512 checked rows: PASS, max_abs 0.003912091 and
relative_RMSE 0.001692613. b1024/chunk0/seed5678, 64 rows: PASS, max_abs
0.006079197 and relative_RMSE 0.001140024, covering the earlier short-input
numerical failure.

### Iteration 054 initial results and broader accuracy checks

Smoke/full-target 8-row checks pass. Full max_abs 0.007641070 and relative_RMSE
0.015046225 nearly match the paired TRTLLM statistics, supporting the scale
diagnosis. Warm events: 1900.90 us versus TRTLLM 1691.84 us. NCU: 126 registers,
194872 shared bytes, occupancy 23.283%, tensor 27.433%, eligible 0.51443,
long scoreboard 5.72494, zero local sectors, shared conflicts 3693140/2291661,
diagnostic 3.221664 ms.

512-row checks PASS independently for seeds1234,5678,42; max_abs respectively
0.014106080, 0.012659281, 0.012925267, relative_RMSE 0.015002714, 0.015003934,
0.014990480. This still does not establish all-row correctness.

The new validate_full_accuracy.py is auditing all 8192 rows in batches using
the unchanged source.reference_rows FP32 expression, TF32 disabled, original
atol=0.01/rtol=0.05. It records failures for every backend before returning,
including example coordinates and allowed error; it never reports its wall
time as a kernel benchmark. TRTLLM, v049, v053 and v054 share exactly the same
inputs/reference for this run. Full-target audit is in progress.

## Iteration 055 — preserve the probability fragment shape

Based on v054, make the FP8 probability fragment inherit the score fragment's
shape instead of flattening it. v053 required this shape preservation for
residual arithmetic and compiled with fewer registers, though other changes
prevent attributing that difference to layout alone. This isolated experiment
checks code-generation/register effects without changing probability values,
MMA order, synchronization or storage coordinates. Validation/NCU pending.

### Full 8192-row seed1234 accuracy audit

All four backends were checked on the same 268435456 output elements, with
the unchanged FP32 reference and original combined tolerance. Results:

| Backend | Tolerance failures | Max absolute error | Relative RMSE |
|---|---:|---:|---:|
| TRTLLM | 9 | 0.019369811 | 0.014979338 |
| v049, P scale256 | 11 | 0.020210505 | 0.014776585 |
| v054, P scale448 | 9 | 0.019369811 | 0.014979332 |
| v053, residual FP8 | 0 | 0.005918741 | 0.001693324 |

v054's nine failure coordinates and values exactly match TRTLLM's recorded
examples. Only 108223/268435456 BF16 outputs differ from TRTLLM, approximately
0.0403%. This establishes closely matching baseline precision for this input,
not a strict all-row tolerance pass. v053 passes every output on this full
input, plus the additional sampled/short/mask cases above; other inputs are
not proven. Continue the fast single-P path with explicit baseline-equivalent
accuracy limits, and retain the residual path for the stricter full-reference
criterion. No tolerances or failing counts are hidden/changed.

### Iteration 055 result

Shape preservation alone has no measurable effect: 1900.77 us versus TRTLLM
1691.94 us, unchanged 8-row errors and 126 registers. NCU: shared 194872 bytes,
occupancy 23.277%, tensor 27.450%, eligible 0.51423, long scoreboard 5.72394,
zero local sectors, shared conflicts 3666848/2324535, diagnostic 3.220000 ms.
The v053 register reduction cannot be attributed to this change alone.

## Iteration 056 — four compute groups with 16 scores per thread

Based on v055, distribute score processing across four compute groups, each
reading 16 physical TMEM columns and holding one head per thread. Eight
partial maxima/sums per head replace four. Each group corrects 64 physical
output columns, corresponding to two disjoint logical channel blocks;
shared output staging uses the exact N256 Layout-E address mapping.

Use four producer warps and 640 total threads to limit the added register
footprint, returning to the prior four-warp gather partition. The final
shared-output redistribution uses 512 compute threads. This changes both
compute distribution and producer count, so performance cannot be attributed
to one factor alone. Coordinate coverage, register/spill behavior and fresh
numerical/performance checks are required. Validation pending.

### Iteration 054 stable timing and device-memory check

512-row Graph20/100 validation passes its sampled rows. Warm candidate
1912.83 us versus TRTLLM 1869.98 us; cold 1910.88 us versus TRTLLM 1931.20 us.
The full-target nine-error limitation remains explicit; this sampled pass
does not override it. Qualified b512/chunk0 device memcheck reports zero
device errors and its two numerical rows pass. The README now presents both
v054's baseline-precision path and v053's full-audit-passing residual path,
instead of treating the earlier 64-row result as a general accuracy claim.

### Iteration 056 result

Score/output coordinate audits, smoke and full-target 8-row checks pass;
full relative_RMSE 0.015046224 is unchanged to the shown precision. Warm
events 1958.11 us versus TRTLLM 1691.84 us regress from v054. NCU: 96 registers,
195896 shared bytes, occupancy 30.260%, tensor 26.552%, eligible 0.71029,
long scoreboard 6.83057, local read/write sectors 6291456/6398904, shared
conflicts 6133068/2818160, diagnostic 3.331168 ms. Higher occupancy/eligibility
does not offset local traffic and extra cross-group bookkeeping. Do not
promote the four-group configuration as measured.

## Iteration 057 — redistribute registers across four compute groups

Based on v056, producer threads release registers to a 48-register budget
and compute threads request 112. The role budgets total
128*48 + 512*112 = 63488 registers, below the 65536-register SM budget; each
role covers whole warp groups and producers release before entering their
loop, so compute register acquisition does not depend on later stage release.
The pattern follows FlashInfer's explicit setmaxregister_decrease/increase
warp-specialized roles. No numerical operation or memory layout changes.

This tests whether the v056 local traffic comes from the static per-thread
budget and can be reduced by moving registers from its producer role. That
cause is a hypothesis, not established solely by the 96-register metric.
Compiler/runtime validation and measured local sectors are pending.

### Iteration 057 failed runtime validation

The b2 smoke test stopped at its bounded 90-second timeout before producing
a numerical result. GPU1 returned to zero utilization/memory afterward; no
reset was performed. No timing or NCU result is accepted for this version.

The earlier budget argument was incomplete: setmaxnreg.inc draws from a
**per-CTA register pool**, not all unused SM registers, and blocks until the
requested registers are available. The 63488-register role total exceeds
640*96=61440 if the CTA starts at v056's 96 registers/thread. This is a
plausible deadlock mechanism; v057's own compiled resource usage is being
checked before treating that initial allocation as established. See the
[NVIDIA PTX setmaxnreg specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg).

## Iteration 058 — constrain redistribution to the CTA pool

Based on v057, lower compute roles from112 to104 registers, retaining48 for
producer threads. The role total is 512*104+128*48=59392 registers, within a
61440-register CTA allocation. All arithmetic/layouts are unchanged. Check
the actual compilation resource budget before a bounded smoke launch, then
collect performance/NCU only if numerical validation completes. Pending.

Offline v057 CUBIN resource inspection confirms REG96, STACK0, LOCAL0.
Thus its initial CTA allocation is indeed 61440 registers, below its63488
role request. This and PTX blocking semantics support the register-pool
deadlock diagnosis; no debugger PC capture was taken during the timeout.

## Iteration 059 — N256 scores with a single KV stage

Based on v058, double keys per tile from128 to256, reducing the main loop
from16 to8 iterations atTopK2048. Four compute groups process32 scores per
thread; the score TMEM region grows from64 to128 physical columns while
output remains in columns0..255. Each producer thread loads two indices,
and each producer warp issues16 gather4 groups. PV has eight K32 steps.

A single256-key KV stage keeps shared allocation below the hardware limit.
The producer waits for the preceding tile's PV completion before overwriting
the stage; full/empty barriers alternate parity every tile. This sacrifices
KV prefetch overlap but halves score reductions and output-correction rounds.
These competing effects and any register spills must be measured.

The wider tile changes floating-point reduction/quantization order, so
accuracy cannot be inferred from v054 even with the same448 probability
scale. Coordinate audit, offline compile, bounded smoke, memory check and
fresh reference tests are required before promotion. Pending.

### Iteration 058 result

Offline CUBIN REG96 confirms the initial61440-register pool. The corrected
role total59392 completes smoke/full-target8-row numerical checks, supporting
the v057 resource-deadlock diagnosis. Warm events1949.73us versusTRT1691.78us
remain slower than v054. NCU:96 registers,195896 shared bytes,occupancy30.240%,
tensor26.670%,eligible0.712712,long-scoreboard6.73514,local read/write sectors
2097152/2131436,shared conflicts6032957/2653887,diagnostic3.312896ms.
Redistribution reduces local traffic by about two thirds versusv056, but
does not remove it or produce a competitive speedup.

## Iteration 060 — finish the four-group register-budget experiment

Based on v057, use112 compute registers and32 producer registers. The total
512*112+128*32=61440 exactly fits the confirmed96-register initial CTA pool.
This isolates whether eliminating the remaining compute-side local traffic
is worth additional pressure on the producer role. Compile resource inspection
and bounded smoke precede full profiling. All arithmetic/layouts unchanged.

### Iteration 059 result — wider single-stage tile rejected

Layout audit, offline compile, b2 and full-target8-row numerical checks pass.
Full relative_RMSE0.015157504 differs from the N128 path as expected. Warm
events2580.70us versusTRT1689.86us are substantially slower. NCU:96 registers,
204072 shared bytes,occupancy30.341%,tensor21.474%,eligible0.417342,
long-scoreboard14.96478,local read/write sectors18350080/29362272,shared
conflicts984884/49579,diagnostic4.107456ms. Offline CUBIN reportsSTACK224.
Although shared conflicts fall, local traffic and lost prefetch overlap
outweigh fewer loop rounds. Do not promote this single-stage configuration;
expanded accuracy/memory validation is not needed for the rejected candidate.

## Iteration 061 — persistent query scheduling on the two-group path

Based on v054, launch min(batch,148) CTAs, reflecting this B300's148SMs. Each
CTA handles query rows separated by the grid size, retaining its512-column
TMEM allocation and initialized barriers. All arithmetic and tile shapes
remain unchanged. A full CTA barrier drains output staging and both producer
and compute roles before the next query starts.

Full/empty KV barrier generations use a cumulative tile count across queries;
Q barrier parity alternates per query. The MMA barrier still completes two
phases per key tile, so each query starts at phase0. First-tile output MMA
continues to overwrite its accumulator. The change trades per-query launch
and TMEM allocation overhead against a new full-CTA boundary and less flexible
query scheduling. Validate multiple queries per CTA, varying sequence lengths,
and exact v054 output equivalence before promotion. Pending.

### Iteration 060 result — no local traffic, still slower

Smoke/full-target8-row checks pass with unchanged v056 errors. CUBIN REG96,
STACK0; NCU confirms zero local load/store sectors. Warm events1937.54us
versusTRT1691.84us improve onv058 but remain slower than the two-group v054.
NCU:195896 shared bytes,occupancy30.252%,tensor26.838%,eligible0.713861,
long-scoreboard6.834383,shared conflicts4475404/2608651,diagnostic3.293888ms.
This isolates local traffic as a real cost in the previous variants, while
also showing that removing it does not justify the four-group design.

## Iteration 062 — specialize persistent-CTA register budgets

v061's offline CUBIN reports128 registers and144 stack bytes, suggesting
that the outer query loop increased register pressure. Based onv061, release
producer registers to48 and allow compute threads192 once before the outer
query loop. Their total61440 fits a128-register512-thread CTA's65536-register
pool; confirm v062's own allocation before runtime. No per-query repeated
setmaxnreg operation is added. Numerical mapping and synchronization remain
unchanged. This tests the spill hypothesis separately from scheduling. Pending.

### Iteration 061 initial result

The b300/chunk0 smoke exercises multiple differently sized queries per CTA
and passes its two sampled rows. Full-target8-row check also passes with
v054's displayed errors. Warm events1988.83us versusTRT1691.94us regress.
NCU:128 registers,194872 shared bytes,occupancy25.003%,tensor26.511%,
eligible0.549299,long-scoreboard5.766146,local read/write11026432/398384
sectors,shared conflicts3504375/3677948,diagnostic3.327936ms.
The register-specialized follow-up is intended to separate that local-memory
penalty from persistent scheduling itself. No promotion at this point.

## Iteration 063 — one producer barrier per KV tile

Based onv054, move the single transaction-expectation arrival before the
index-publication barrier. That barrier then publishes both the indices and
expectation before any gather4 is issued, eliminating the second barrier.
Remove the post-gather producer barrier: the next tile uses separate indices
and KV storage; reuse of a stage waits for its compute/PV empty notification.
The next tile's publication barrier still waits for all producer threads,
and transaction completion accounts for all current-tile DMA before compute
can read it. Producer synchronization falls from three to one per tile.

No score arithmetic, addresses, byte counts or compute ordering changes.
This synchronization optimization requires fresh smoke, equivalence and
qualified device-memory checks if it improves performance. Pending.

### Iteration 062 offline resource rejection

The compiler reduced initial CUBIN REG usage to110 (STACK0), invalidating the
assumed128-register starting pool. Even rounding110 up to112 gives57344
registers, below the requested61440. **Do not launch this configuration.**
There is no GPU timing, numerical result or NCU measurement forv062. Preserve
the compile/resource evidence and inspect generated register-control SASS.

## Iteration 064 — reduce the persistent role budget

Based onv062, request176 compute registers and32 producer registers:53248
total, within a104-register initial512-thread pool. Compilation may again
change initial resource usage; check it before any launch. This keeps the
persistent schedule and numerical operations unchanged. Pending.

### Iteration 062 SASS inspection supersedes the preliminary rejection

The PTX contains the requested48/192 setmaxnreg operations, but the final
CUBIN SASS contains **zero USETMAXREG instructions**. By contrast v058 emits
USETMAXREG.DEALLOC.CTAPOOL and ALLOC.CTAPOOL. Therefore v062 does not actually
request the nominal61440-register role total at runtime, and the preliminary
resource-deadlock rejection above does not apply to this compiled binary.
In this placement, the role branches immediately reconverge before the query
loop; the final optimizer output has no register-control instructions. Its REG110/STACK0 code generation is still
worth measuring, but any improvement must not be attributed to dynamic
register redistribution without evidence in the generated instructions.
A bounded runtime check is now appropriate after SASS inspection.

### Iteration 063 initial result

Smoke/full-target8-row checks pass with the same displayed v054 error values.
Warm events1890.69us versusTRT1691.94us are about0.5% belowv054, a small gain
requiring stable paired Graph measurement. NCU:126 registers,194872 shared
bytes,occupancy23.253%,tensor27.586%,eligible0.510583,long-scoreboard5.86779,
zero local sectors,shared conflicts3947070/2334346,diagnostic3.200544ms.
Expanded equivalence, reference/Graph and qualified memcheck pending.

### Iteration 062 measured result

After confirming no dynamic register-control instructions in final SASS,
the b300/chunk0 and full-target8-row checks pass. Warm events1904.77us versus
TRT1691.94us recover most ofv061's regression, but do not beatv054/v063.
NCU:110 registers,194872 shared bytes,occupancy24.995%,tensor27.487%,
eligible0.546396,long-scoreboard5.519212,zero local sectors,shared conflicts
4315995/2179636,diagnostic3.215936ms. Attribute the change to generated code
and removal of local traffic, not an executed setmaxnreg mechanism.

v064 likewise emits no USETMAXREG instructions; offline resources areREG91,
STACK0. Its nominal role-budget arithmetic is therefore not a runtime
allocation requirement for the inspected binary. Runtime validation pending.

### Iteration 063 expanded validation

Full8192/seed1234 equivalence versusv054 passes all268435456 BF16 bit patterns
in each of three runs. It therefore inherits v054's nine original-tolerance
failures on this input; it is not a strict-reference pass. The independent
512-row Graph check passes. Warm1913.23us versusTRT1869.90us and cold1900.56us
versusTRT1941.54us do not establish a stable warm gain overv054. Qualified
b512/chunk0 memcheck with --report-api-errors no reports zero device errors,
and its two reference rows pass. Keep the default/reference versionv054.

### Iteration 064 result

The b300/chunk0 and full-target8-row checks pass. Warm2121.76us versus
TRT1693.92us regress despite lower registers and zero local traffic.
NCU:91 registers,194872 shared bytes,occupancy24.987%,tensor24.271%,
eligible0.492440,long-scoreboard6.187685,shared conflicts4337694/566972,
diagnostic3.639552ms. Lower static register count is not sufficient evidence
of better code generation. No promotion.

## Iteration 065 — cache V across high/residual PV operations

Based on the full-audit-passing residual v053, use the four weight-stationary
B collector buffers. For each output N256 tile, the four high-P K32 MMAs fill
buffers b0..b3, then the four residual-P MMAs consume their matching buffer
with lastuse. The B matrices/descriptors are identical between matching
high/residual operations, and no intervening operation overwrites a buffer.
Each output element retains its accumulation order; the independent N tiles
are scheduled high/residual together to prevent collector overwrite. QK keeps its default discard
behavior, after the previous PV has completed. No memory layout or numerical
formula changes, and v063's producer change is not composed yet.

This targets repeated V reads into the Tensor Core operand collector for the
second PV term. The weight-stationary b0..b3 mechanism is distinct from the
newer ordinary-MMA collector::b qualifiers that requireSM107f; the WS form is
available in PTX8.6/SM100. See [NVIDIA tcgen05.mma.ws](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-ws).
Offline syntax/resource inspection, bounded smoke, equivalence and NCU pending.

### Iteration 065 initial result

Offline compile, b2 and full-target8-row reference checks pass. Full max_abs
0.001884818 and relative_RMSE0.001690517 retain the residual path's accuracy.
Warm events2224.06us versusTRT1693.57us improve about4.0% fromv053.
NCU:102 registers,203064 shared bytes,occupancy23.315%,tensor34.389%,
eligible0.523675,long-scoreboard6.429123,zero local sectors,shared conflicts
3731712/3012490,diagnostic3.776832ms. Full-output equivalence, stable Graph
and final SASS collector inspection are in progress.

### Iteration 065 full equivalence and stable timing

All268435456 BF16 output bit patterns matchv053 in three full-target/seed1234
runs. Consequently the full-reference pass ofv053 applies to these identical
outputs; this is transitive evidence for that input, not a new independent
FP32 audit or a claim about arbitrary inputs. The512-row Graph check passes.
Warm2279.70us versusTRT1871.50us and cold2269.34us versusTRT1947.44us improve
overv053's2356.21/2332.77us, respectively. Final SASS explicitly contains
B_KEEP followed by B_REUSE for collector buffers0..3 on both output tiles,
confirming the intended mechanism. Qualified memory/short-input checks pending.

## Iteration 066 — bounded scaling anchor with residual probabilities

Based onv065, retain the running softmax anchor while the next tile maximum
is at most2.5 log2 units above it. Use probability scale64 instead of256,
leaving maximum high-P input64*2^2.5≈362.04 below E4M3's448 finite limit.
When the bound is exceeded, update the anchor to the new maximum and perform
the usual output/denominator correction. Numerator and denominator always
share the same anchor. This preserves the real-arithmetic attention formula.

Earlier single-P anchor versionv051 failed expanded accuracy; this version
retains residual FP8 PV to recover probability rounding error. Lower scaling
can still increase underflow/quantization error, so it requires independent
full-reference validation, additional seeds and short/masked inputs. No
accuracy pass is inherited fromv065. The purpose is to skip more identity
output corrections while keeping strict-reference accuracy. Pending.

### Iteration 065 short-input and memory validation

b1024/chunk0/seed5678 matchesv053 on all33554432 output bit patterns in each
of three runs. Qualified b512/chunk0 device memcheck with --report-api-errors
no reports zero errors and passes its two sampled reference rows. The README
now points residual performance runs tov065, retainingv053 as the independently
audited full-reference baseline andv054 as the fast baseline-precision path.

## Iteration 067 — wider bounded anchor window

Based onv066, use probability scale16 and a4.5-log2 anchor window. The maximum
high-P input remains16*2^4.5≈362.04, so overflow headroom is unchanged. This
may avoid additional output corrections, at the cost of coarser FP8 subnormal
resolution for the high/residual terms. This is a distinct speed/accuracy
experiment, not an assumed improvement. No tolerance or reference changes.
Independent full-reference and additional-input checks are required before
any accuracy claim or promotion. Pending.

### Iteration 066 initial performance and independent full audit

Smoke/full-target8-row checks pass; warm events2160.61us versusTRT1691.81us
improve about2.9% overv065. NCU:102 registers,203064 shared bytes,
occupancy23.327%,tensor35.395%,eligible0.524663,long-scoreboard6.604001,
zero local sectors,shared conflicts3599362/2882458,diagnostic3.670752ms.

An independent full8192-row/seed1234 FP32-reference audit passes every one
of268435456 output elements with the unchanged tolerance. Max_abs0.005918741,
relative_RMSE0.001709720; all finite, zero mismatches. The lower probability
scale slightly increases relative error fromv053's0.001693324 while preserving
this full-input pass. Other seeds/short/masked inputs and stable timing pending.

## Iteration 068 — N256 with two compute groups and a larger role budget

Revisit the rejectedv059 single-stage N256 idea onv066's residual path. Use
two compute groups, each processing64 scores per thread, plus eight producer
warps. Compute requests208 registers and producers32 (61440 total), subject
to inspection of the actual compiled pool. This targetsv059's heavy local
traffic by allowing larger per-thread compute fragments; it changes multiple
resource choices and is not an isolated measurement of tile size alone.

A single256-key KV stage and two16KiB probability buffers fit shared memory.
Each query has eight iterations. QK usesN256; PV uses eightK32 steps. Four
collector buffers cover one K-group of128 keys at a time, scheduling its
high/residual pair before reusing collectors for the next K-group. Each output
now interleaves high/residual terms across these K-groups, so FP32 accumulation
order changes and an independent accuracy audit is necessary.

Score coordinate audit, offline registers/spills, bounded smoke and profiling
precede any promotion. The producer loses double-buffer prefetch; reduced
synchronization may or may not offset that cost. Pending.

### Iteration 067 initial performance and two full-reference seeds

Smoke and8-row target checks pass. Warm events2048.38us versusTRT1693.86us,
about5.2% belowv066. NCU:102 registers,203064 shared bytes,occupancy23.314%,
tensor37.469%,eligible0.539711,long-scoreboard6.569879,zero local sectors,
shared conflicts3586383/2894575,diagnostic3.468544ms.

All268435456 elements independently pass for both seed1234 and seed5678.
Seed1234:max_abs0.005918741,relative_RMSE0.001735970; seed5678:max_abs
0.005017400,relative_RMSE0.001735982. The same seed5678 audit independently
passesv066 with max_abs0.004050493,relative_RMSE0.001709954. These numerical
comparisons show a small accuracy tradeoff, with both still passing the
original tolerance. Stable timing and additional edge checks pending.

## Iteration 069 — scale4 with a6.5-log2 bounded anchor

Based onv067, lower high/residual probability scale to4 and widen the anchor
window to6.5 log2 units, retaining the same maximum input≈362.04. This tests
additional correction avoidance against increased subnormal rounding error.
Independent reference audits, including additional seeds, remain mandatory;
passingv067 does not validate this candidate. No tolerance changes. Pending.

### Iteration 067 stable timing and edge validation

512-row Graph checks pass: warm2134.14us versusTRT1869.14us, cold2107.38us
versusTRT1926.99us. These sustained timings are slower than the short2048us
event result and must not be conflated. Both regimes improve overv065.

The full1024-row/chunk0/seed5678 audit passes all33554432 elements forv066
andv067. v066:max_abs0.007934809,relative_RMSE0.001657998;v067:max_abs
0.008034229,relative_RMSE0.001678572. v067's internal-hole/partial case passes
the unchanged reference with max_abs0.004001856. Qualified b512/chunk0
memcheck (--report-api-errors no) reports zero device errors and passes its
two numerical rows. v067 is the best validated higher-precision path so far.

v068's offline coordinate audit passes. CUBIN REG128,STACK176, with actual
USETMAXREG32/208 instructions; its61440-register role sum fits the65536
initial pool. The nonzero stack suggests local traffic may still limit it;
runtime profiling is required before evaluating the N256 hypothesis.

### Iteration 068 result — N256 rejected again

Smoke/full-target8-row checks pass, but warm2820.26us versusTRT1691.87us
regresses substantially. NCU:128 registers,219432 shared bytes,occupancy
23.358%,tensor28.350%,eligible0.313090,long-scoreboard12.293159,local read/
write sectors28049420/11543824,shared conflicts32813/112796,diagnostic
4.578528ms. Even the larger compute register budget does not remove local
traffic; single-stage N256 is not promising in this form. No promotion or
expanded reference claim for this rejected performance experiment.

## Iteration 070 — single-P speed/accuracy boundary experiment

Based onv054, use the same scale16/window4.5 finite-range anchor rule asv067,
with a single FP8 probability term. This isolates the speed/accuracy tradeoff
of reduced output correction without the extra residual PV. It is **not**
a higher-precision candidate and cannot inheritv067's all-row passes or
v054's close TRTLLM equivalence. Earlier bounded single-Pv051 had more
reference failures, so full auditing against both FP32 and TRTLLM is essential.

Keep original tolerances, retain every failure count, and do not promote based
on the original eight sampled rows alone. The current validated fast and
higher-precision paths remainv054 andv067. Pending.

### Iteration 069 result — little speed benefit for lower precision

Smoke/full-target8-row checks pass. Warm2043.84us versusTRT1693.66us improves
only4.5us over v067's short run, insufficient to favor the precision tradeoff.
NCU:102 registers,203064 shared bytes,occupancy23.319%,tensor37.648%,
eligible0.540857,long-scoreboard6.562892,zero local sectors,shared conflicts
3600192/2890032,diagnostic3.454304ms.

Full8192-row audits pass the original combined tolerance for seed1234 and
seed5678. Their max_abs errors rise to0.011162430/0.010025859 and relative_RMSE
to0.001953739/0.001952327. A max absolute error over0.01 can still pass the
unchanged atol+rtol criterion; no threshold was altered. Prefer v067's
better precision and already completed stable/edge validation.

### Iteration 070 smoke rejection

Original tolerance fails on1/65536 elements in b2 smoke:row0/head1/channel486,
absolute error0.011013217,relative error1.646965. The runner stops before
benchmark/NCU; no speed is claimed and no tolerance is relaxed. The planned
full performance run is not executed. Wide anchors need the residual term.

## Iteration 071 — M32 head tiles with weight-stationary B reuse

Based onv054, keep exact running maxima and P scale448, but use two M32 head
tiles instead of one M64 tile. Each compute group owns32 heads and all128
keys. PTX Layout G maps key quarters across four32-DP regions. Output uses
two N256 tiles per head group, still256 physical TMEM columns. Probability
stores, partial reductions, denominator ownership and output staging are
remapped explicitly.

For each QK/PV K32 step, fill B collector0 for the first head tile and reuse
it for the second. This doubles MMA instruction count at halfM, keeping
useful arithmetic unchanged; savings may not offset issue overhead. Each
output retains MMA accumulation order, while denominator sum order changes.
Coordinate audit, offline compile, bounded smoke and NCU precede promotion.
The retained M64 CuTe MMA objects are unused wrapper scaffolding; actual
inline idesc encodesM32. Source:[NVIDIA Layout G](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-g).

### Iteration 071 result — spills obscure the M32 reuse hypothesis

Coordinate audit, offline compile, b2 smoke and target eight-row checks pass.
Warm events2228.32us versusTRT1691.90us regress fromv054. NCU:128 registers,
194872 shared bytes,occupancy23.258%,tensor23.336%,eligible0.457546,
long-scoreboard7.073924,local read/write sectors20185088/2391696,shared
conflicts5955718/852765,diagnostic3.786240ms. CUBIN STACK40 agrees with the
large local traffic. No expanded accuracy or promotion for this experiment.

## Iteration 072 — isolate M32 performance with explicit role registers

Based onv071, add producer32/compute208 register limits inside the long-lived
role branches. The total requested budget is61440 registers for512 threads;
inspect the actual CUBIN pool and USETMAXREG instructions before launching.
This tests whether removing local traffic can make the M32 B-collector design
competitive. Arithmetic, tile layout and synchronization are unchanged.
Offline resource inspection, bounded smoke, paired events and NCU pending.

### Iteration 072 result — hints removed; M32 remains slower

PTX contains32/208 setmaxnreg but SASS contains no USETMAXREG. CUBIN REG128,
STACK32: this is a static code-generation change, not verified dynamic role
allocation. Smoke/eight-row checks pass. Warm2218.21us versusTRT1691.55us.
NCU:128 registers,194872 shared bytes,occupancy23.265%,tensor23.405%,eligible
0.471148,long-scoreboard6.935508,local sectors11010048/1861796,shared
conflicts6187629/928314,diagnostic3.774368ms. Local loads fall but speed barely
changes. The experiment does not establish spill-free M32 performance.

## Iteration 073 — revisit normal MMA and TMEM probabilities

Composev036's normal-MMA Layout F/TMEM-P implementation with later changes:
Q TMA concurrent with KV gathering, eight producer warps, deferred cross-group
denominator summation, one shared reciprocal per head, packed correction with
one store wait per four fragments, and shared-memory output staging followed
by128-bit global stores. Use single-P scale448 as in the baseline-precision
path. Producer32/compute208 hints are included but require SASS verification.

Packed probabilities retainv036's audited duplication into both16-lane DP
halves. The epilogue uses the one-head correction copy layout and separately
audits contiguous8-element BF16 vectors. Denominator accumulation order changes;
no bitwise or full-accuracy claim is inherited. This tests whether the old
TMEM-P result was dominated by its old scalar epilogue and spill behavior.
Coordinate audit, offline inspection, bounded smoke, events and NCU pending.

### Iteration 073 result — no spills, but normal MMA remains slower

Layout/compile/smoke/eight-row checks pass. CUBIN REG104,STACK0; setmax hints
again disappear from SASS. Warm2326.69us versusTRT1691.74us. NCU:104 registers,
186168 shared bytes,occupancy23.264%,tensor44.488%,eligible0.416636,
long-scoreboard7.806350,zero local sectors,shared conflicts4739152/623184,
diagnostic3.967360ms. This is faster than oldv036 but slower thanv054; higher
tensor-active percentage across different MMA modes does not imply better
useful throughput. No promotion or expanded numerical claim.

## Iteration 074 — dedicated MMA warp with QK lookahead

Based onv054. Keep256 compute threads, dedicate warp8 to MMA issue, and use
seven producer warps for32 gather4 tasks (round-robin, five groups with a
bound check). Total512 threads. Two score buffers occupy TMEM columns256–383;
output remains0–255 and probabilities stay in shared memory.

MMA prologue issuesQK0; each iteration issuesQK(i+1) before waiting for the
current probability/correction-ready barrier and issuingPVi. QK completion
has two stage barriers; PV has one alternating-phase barrier. Compute waits
for QKi, preparesPi and rescalesO, then signals readiness and waits forPVi.
MMA releases the KV stage only afterPV completes. This keeps O correction
ordered while overlapping nextQK with currentsoftmax. Additional score buffers
are disjoint; no early overwrite of probability or KV buffers is permitted.

TRT SASS shows a distinct warp8 path issuing QK/PV (trtllm_sass.txt:3430,3562,
3974) while softmax occurs earlier in different branches. The exact TRT schedule
is not inferred from names alone; this experiment independently measures one
concrete overlap schedule. Bounded smoke, memcheck, numerical equivalence and
performance profiling are required before promotion. Pending.

### Iteration 074 result — explicit lookahead did not improve throughput

Offline REG93,STACK0. b2/eight-row target checks pass, but warm2371.71us
versusTRT1689.98us regresses. NCU:93 registers,194904 shared bytes,occupancy
23.616%,tensor24.414%,eligible0.404843,long-scoreboard8.743213,zero local
sectors,shared conflicts77320/2409117,diagnostic3.613792ms. Lower shared-load
conflict count does not compensate for the scheduling cost. No promotion;
expanded equivalence/memcheck deferred for this rejected performance branch.
The measured result does not disprove other overlap schedules or TRT's finer
pipeline organization.

## Iteration 075 — incorporate P scale into exp2 and denominator

Based onv067's fully audited residual path. Compute exp2(score + (4-anchor))
directly, quantize that scaled value into high/residual terms, and accumulate
the denominator at the same scale. Final reciprocal removes the extra16
factor. This removes explicit per-element probability multiplication and
keeps the bounded-anchor finite-range argument unchanged. The reassociation
changes FP32 rounding and requires independent reference audits; it does not
inheritv067's accuracy results. Offline codegen, bounded smoke, paired events,
NCU and full-reference checks pending.

## Iteration 076 — one compute warpgroup with a larger register budget

Based onv067. One128-thread compute group reads64 scores per thread, with
256 producer threads retained (384 total). Two partial maxima/sums per head
replace four. Output correction/epilogue cover eight32-column chunks per
thread instead of four; the shared output buffer still enables128-bit stores.
Producer32/compute240 hints request38912 registers, subject to actual CUBIN
pool and instruction inspection. No block size, probability scale or residual
PV changes. The reduction tree changes, requiring independent accuracy checks.

This tests less synchronization and fewer resident threads against more
per-thread softmax work and register pressure. Layout audit, offline resources,
bounded smoke and profiling pending. It is independent ofv075's exp2 change.

### Iteration 075 measured result and expanded validation

Offline REG118,STACK0. Smoke/eight-row checks pass. Warm events1990.94us
versusTRT1691.81us, about2.8% belowv067's2048.38us short run. NCU:118 registers,
203064 shared bytes,occupancy23.324%,tensor38.514%,eligible0.546450,
long-scoreboard6.370164,zero local sectors,shared conflicts3534521/4216127,
diagnostic3.374176ms. The extra register count causes no measured spills.

Independent full8192 audits pass all268435456 elements for seeds1234/5678:
max_abs0.005918741/0.005017400,relative_RMSE0.001735970/0.001735984. Full1024
chunk0/seed5678 passes all33554432 elements,max_abs0.008034229,relative_RMSE
0.001678574. Internal-hole/partial test passes,max_abs0.004001856. Qualified
b512 memcheck (--report-api-errors no) reports zero device errors and passes
its two numerical rows. No tolerance changes or inherited bitwise claim.

512-row Graph validation: warm2097.22us versusTRT1874.14us,cold2080.51us
versus1925.12us. This improves the sustained v067 result by about1.7% warm and
1.3% cold, smaller than the short-event improvement. v075 becomes the current
validated higher-precision performance candidate; it still trails TRTLLM.

v076 offline audit passes. CUBIN REG168,STACK0; SASS retains real USETMAXREG
32/240. The38912-register requested role sum fits the64512-register launch
pool. Runtime validation remains pending.

### Iteration 076 result — lower softmax parallelism loses throughput

b2/eight-row checks pass. Warm2146.50us versusTRT1691.58us. NCU:168 registers,
202552 shared bytes,occupancy17.014%,tensor35.458%,eligible0.390134,
long-scoreboard5.316915,zero local sectors,shared conflicts3944280/2214780,
diagnostic3.668512ms. Dynamic role allocation works and avoids spilling, but
more work per compute thread is slower thanv067's two-group version. No
expanded accuracy claim or promotion.

## Iteration 077 — raw QK maxima and fused affine exp2 inputs

Based onv075. Reduce maxima in raw QK units, express the anchor window in those
units, and convert each exp2 input with score*log2_scale+(4-anchor*log2_scale).
This should combine the old per-element score multiply and later offset into
one FMA; verify final SASS instead of assuming contraction. Correction factors
now convert the raw-max difference to log2 units. Probability scale16 and both
residual PV passes remain unchanged. Rounding and possibly anchor-boundary
decisions change, so independent full-reference audits are required. Pending.

## Iteration 078 — combine reduced producer synchronization with residual P

Applyv063's three-to-one producer-barrier change tov075. Publish gather indices
and expected transaction bytes together before the remaining producer barrier.
The other KV stage has separate indices, and a reused stage still waits for PV
completion. Arithmetic is unchanged, so full bitwise comparison withv075 is
appropriate in addition to device memcheck. The same change did not improve
v054's sustained warm timing; this tests whether the residual path has a
different balance. Offline, smoke, events/NCU and validation pending.

### Iteration 077 initial result — tiny short-run gain

Offline REG118,STACK0. SASS FFMA-family lines rise from10 in v075 to27 in v077
(the latter includes packed FMA); source-level fusion is reflected in codegen.
Smoke/eight-row checks pass. Warm1984.70us versusTRT1691.84us, only6.24us below
v075, not enough alone for promotion. NCU:118 registers,203064 shared bytes,
occupancy23.304%,tensor38.701%,eligible0.493488,long-scoreboard6.698395,zero
local sectors,shared conflicts3562355/4689615,diagnostic3.358144ms.

Independent seed1234 full8192 audit passes all268435456 elements,max_abs
0.005918741,relative_RMSE0.001735970. No second-seed/edge/stable claim yet;
v075 remains the default higher-precision candidate.

## Iteration 079 — widen probability swizzling to128 bytes

Based onv075, change only the high/residual P layout atom fromK_SW64 toK_SW128.
Each P buffer is still64x128 FP8 and8KiB; Q/KV layouts, MMA arithmetic and
thread ownership remain unchanged. This tests whether the roughly4.2M
shared-store conflicts include avoidable conflicts from128-byte head strides
under a64-byte swizzle. Manual MMA descriptors derive their swizzle from the
new layout. Compile/layout checks and bounded smoke precede performance;
full bitwise comparison withv075 and qualified memcheck are needed before
promotion. This is independent ofv077/v078. Pending.

### Iteration 078 initial result — producer-barrier reduction nearly ties

Offline REG118,STACK0. Smoke/eight-row checks pass. Warm1988.64us versus
TRT1693.76us, only2.3us belowv075. NCU:118 registers,203064 shared bytes,
occupancy23.297%,tensor38.615%,eligible0.537665,long-scoreboard6.560339,zero
local sectors,shared conflicts3929262/3889222,diagnostic3.364384ms. No stable
speedup established, so no promotion or expanded equivalence claim yet.

### Iteration 079 result — SW128 does not change the observed conflicts

Offline REG118,STACK0. Smoke/eight-row checks pass. Warm1992.90us versus
TRT1691.81us tiesv075 within short-run variation. NCU:118 registers,203064
shared bytes,occupancy23.308%,tensor38.526%,eligible0.545882,long-scoreboard
6.362201,zero local sectors,shared conflicts3534640/4217042,diagnostic
3.374592ms. The store-conflict count is essentially unchanged fromv075's
4216127, so this measurement does not support the proposed P-swizzle cause.
No promotion or expanded equivalence/memcheck claim. Collect per-instruction
SourceCounters onv075 before attributing the remaining aggregate conflicts.

## Iteration 080 — exact all-valid tile detection for mask bypass

Based onv075. The128 index-loading threads vote within four full warps, publish
four validity flags per KV stage, and combine them after the existing producer
barrier. The combined flag is stored before arrive-and-expect-tx. Compute
bypasses per-element index loads/selects only when every one of the128 slots
is nonnegative. Out-of-length slots remain-1; internal holes force the original
mask loop. Unlike rejectedv031, no endpoint or contiguity assumption is used.

Probability representation, arithmetic and all MMA accumulation orders remain
unchanged; vectorizing score scaling can still affect compiler scheduling.
Bounded smoke, the explicit internal-hole/partial reference case, full bitwise
comparison withv075 and device memcheck precede any promotion. Pending.

### v075 source-level profile — distinguish arbitration from address conflicts

SourceCounters reports52248576 shared wavefronts and exactly52248576 ideal
wavefronts, with zero excessive wavefronts across all instrumented instructions.
This does not support an intra-warp shared-address conflict problem despite
aggregate hardware conflict counters of3.53M loads/4.22M stores. Preserve those
raw counters, but do not equate them with removable address-bank conflicts.
NVIDIA explains that hardware counters also include lost arbitration against
TMA fills, tensor-core reads and other clients; source excessive counters
isolate instruction address behavior ([NVIDIA clarification](https://forums.developer.nvidia.com/t/nsight-compute-h100-questions-on-l1-bank-conflict-statistic-discrepancies-between-details-and-source-pages/351780/3)).

The top three long-scoreboard sample locations are conditional branches
consuming mbarrier phase-check predicates:29005,15470 and12235 samples out of
71535 total (79.28%). The first is the producer's KV-stage-reuse wait; the other
two follow the compute MMA-completion waits. The count is sampled stall
attribution, not a direct estimate of speedup or proof of DRAM bandwidth
saturation. Source CSV and derived summaries are saved underartifacts/v075_source.

### Iteration 080 result — exact mask bypass adds more overhead than it saves

Offline REG110,STACK0. Smoke/eight-row checks and explicit hole/partial reference
case pass (max_abs0.004001856). Warm2064.54us versusTRT1691.90us regresses
fromv075. NCU:110 registers,203104 shared bytes,occupancy23.331%,tensor37.216%,
eligible0.642488,long-scoreboard5.725958,zero local sectors,aggregate shared
conflicts3927426/2716004,diagnostic3.492352ms. Better eligible-warp and stall
ratios do not imply lower total latency. No promotion or full bitwise claim.

## Iteration 081 — packed FP16 intermediate for probability residuals

Based onv075. Round scaled FP32 probabilities to FP16, quantize high FP8 from
those values, and form the low FP8 residual using FP16 subtraction. Keep the
FP32 denominator based on the original exp2 values. Q, KV and both tensor-core
PV operands remain FP8; there is no KV expansion or BF16 attention fallback.

This intentionally trades some probability precision for cheaper packed
conversion/subtraction: inspect whether SASS uses half2 arithmetic and avoids
the FP8-to-FP16-to-FP32 reconstruction present inv075. The extra rounding and
double-rounding boundaries require independent full-reference audits on both
seeds and short/masked inputs. No inherited numerical claim. Pending.

### Iteration 081 result — packed half arithmetic is not faster

Offline REG109,STACK0. SASS has packed FP16 conversion and HADD2 subtraction,
confirming the intended intermediate precision. b2/eight-row checks pass.
Warm2021.57us versusTRT1691.81us is slower thanv075. NCU:109 registers,203064
shared bytes,occupancy23.310%,tensor38.015%,eligible0.530033,long-scoreboard
6.715946,zero local sectors,aggregate shared conflicts3618296/2990988,
diagnostic3.422016ms.

Independent full8192 audits pass both seeds1234/5678, with max_abs
0.005918741/0.004814863 and relative_RMSE0.001750538/0.001750498. Slightly
higher error and no speed benefit: no promotion or further edge validation.

## Iteration 082 — M32/N64 split-head CTAs targeting two-CTA residency

Each query launches two CTAs, each owning32 heads and all512 output channels.
Use64-key tiles,128 compute threads and128 producers, and256 TMEM columns.
Q takes18KiB, double KV72KiB, and two P buffers4KiB, allowing two CTAs in the
B300 shared-memory budget if register/TMEM limits also permit it. Explicit
producer32/compute192 requests28672 registers per CTA; verify the compiled
initial pool before launching. Occupancy must be measured, not assumed.

Q TMA descriptors fetch32 rows atquery*64+head_tile*32. LayoutG scores use16
physical columns across128 DP lanes; output uses128 columns across twoN256
tiles. Each compute thread handles16 scores and four32-value output chunks.
PV uses two high/residualK32 steps perN tile, with two B collectors. The final
32KiB output staging buffer aliases one KV-main stage.

This doubles gathered KV traffic per query and increases MMA instruction count
at smallerM/N, but permits independently scheduled CTAs. The experiment tests
whether concurrency can outweigh those costs. Probability math followsv075;
smaller tiles change online reduction and accumulation order. Layout audit,
offline resource inspection, bounded smoke and full-reference checks pending.

The initialv082 offline build (SHA c923dc13...) had REG64/STACK64 and no actual
USETMAXREG instructions. Before any GPU launch, add min_blocks_per_mp=2 to
make the intended residency explicit. Retain initial compile/resource/SASS
records with an initial suffix; only the subsequent source SHA can receive
runtime results. The launch-parameter change needs a fresh resource inspection.

The explicit launch-bound compile requires CUDA device-attribute queries in
this DSL and fails with CUDA_ERROR_NOT_INITIALIZED in the fully offline setup.
Preserve that failure, and add an opt-in --initialize-cuda flag to the fake-
tensor compiler. It creates a context only on the selected authorized GPU and
launches no candidate kernel. Default compilation remains fully offline.
This is a compilation-environment requirement, not a kernel correctness result.

With the authorized device context and explicit .minnctapersm2, v082 compiles
as REG128/STACK0 and retains actual USETMAXREG32/192. The28672-register role
sum fits its32768-register launch pool. Runtime source SHA is05cc8776..., and
all future measurements must use that SHA. This confirms the static resource
plan only; actual two-CTA residency and throughput remain to be profiled.

### Iteration 082 result — two CTAs fit, but the smaller-tile work is slower

b2/eight-row target checks pass. Warm2861.12us versusTRT1691.68us. NCU:128
registers,97464 shared bytes,occupancy24.040%,tensor26.109%,eligible0.492241,
long-scoreboard6.966612,zero local sectors,aggregate shared conflicts
21527486/22240316,diagnostic4.964096ms. Eight warps per CTA and achieved
occupancy near24% demonstrate more than one CTA resident (one CTA alone
would peak at12.5%). Concurrency does not offset this design's extra gather/
MMA/loop work. No promotion or expanded accuracy claim for the rejected branch.

## Iteration 083 — explicit launch bound with role register redistribution

Based onv075, request one CTA per SM and place producer32/compute192 register
hints inside the persistent role branches. Total requested57344 registers fits
a65536-register CTA pool if the final CUBIN allocates128 per512-thread launch.
Earlier hints without explicit launch bounds sometimes disappeared in SASS;
inspect both instructions and launch resources before testing. Arithmetic and
memory layout remain unchanged. Full bitwise comparison can assess numerical
equivalence if the candidate improves performance. Pending.

## Iteration 084 — balanced denominator contribution tree

Based onv075. Replace the sequential32-value probability sum with an explicit
five-level balanced pairwise tree. Thev075 source SASS shows a long FADD chain,
some of which the compiler overlaps with PV completion. A shorter dependency
chain may improve latency, but its scheduling/register cost must be measured.
Probability quantization and PV accumulation are unchanged; denominator rounding
changes, requiring independent FP32-reference validation. Pending.

## Iteration 085 — exp2 scale folding on the baseline-precision path

Applyv075's scale-folding idea tov054's single-P scale448 path. Compute
exp2(score+(log2(448)-max)), accumulate the scaled denominator, quantize P
directly, and normalize by the scaled sum. This removes the explicit P multiply.
It uses exact running maxima, with no residual or relaxed-anchor window.

This targets comparison at TRTLLM's FP8 probability precision. It is not a
higher-precision replacement forv075. Non-power-of-two448 and exp2 rounding
can change P quantization; full audits must report every failure alongside
TRTLLM, without assumingv054's identical nine failures or changing tolerances.
Offline, smoke, paired performance and full-reference comparison pending.

v083 preflight: REG128,STACK0, actual USETMAXREG32/192, .minnctapersm1.
The57344 role budget fits65536 initial registers. v084 preflight: REG118,
STACK0. Both are ready for bounded runtime evaluation; no timing claim yet.

### Iteration 083 result — register redistribution regresses

The explicit launch bound retains USETMAXREG32/192 and REG128/STACK0, but
warm latency rises to 2027.90 us versus paired TRT 1693.70 us. Smoke and the
eight-row target check pass. NCU: 203064 shared bytes, occupancy 23.256%,
tensor active 37.622%, eligible 0.545481, long-scoreboard 6.428361, zero local
sectors, aggregate shared conflicts 4153511/4262370, diagnostic 3.456352 ms.
No expanded numerical validation or promotion for this slower candidate.

### Iteration 084 result — balanced sum gives no measured benefit

Warm 1995.17 us versus TRT 1692.06 us, close to v075's 1990.94 us. NCU:
118 registers, 203064 shared bytes, occupancy 23.314%, tensor active 38.311%,
eligible 0.553669, long-scoreboard 6.533945, zero local sectors, aggregate
shared conflicts 3442682/4653482, diagnostic 3.395424 ms. Independent full
8192-row seed1234 audit passes all 268435456 elements: max_abs 0.005918741,
relative_RMSE 0.001735969251. No promotion or second-seed/edge claim.

### Iteration 085 initial result — scale folding improves the fast path

Offline REG119/STACK0. Smoke/eight-row checks pass. Warm 1878.05 us versus
TRT 1691.97 us improves on v054's 1900.90 us short run. NCU: 119 registers,
194872 shared bytes, occupancy 23.307%, tensor active 27.763%, eligible
0.530916, long-scoreboard 5.727432, zero local sectors, aggregate shared
conflicts 3542133/4154715, diagnostic 3.182944 ms.

Full 8192-row seed1234 audit has exactly the same nine failing coordinates
and actual values as paired TRT, with unchanged atol0.01/rtol0.05. Both have
max_abs 0.019369811; relative_RMSE is 0.01497933784 versus TRT 0.01497933767.
79767 out of 268435456 BF16 elements differ from TRT (99.9703% bitwise equal).
This is baseline-level FP8 precision, not a strict full-reference pass.
Sustained Graph, a second full seed, device memcheck and masks are pending.

## Iteration 086 — raw-score maxima and fused exp2 input on v085

Keep maxima in raw QK units, scale their difference for accumulator correction,
and form each exp2 input as raw_score*log2_scale + (log2(448)-max*log2_scale).
This applies v077's arithmetic reassociation to the single-P fast path and
may generate packed FFMA2 instead of separate multiply/add instructions.
Exact running maxima and all synchronization/layouts remain unchanged.
Changed FP32 rounding can affect quantization; full independent audits against
FP32 and paired TRT are required for any promotion. Compilation and tests pending.

## Iteration 087 — one producer barrier on v085

Combine index publication and transaction expectation before one producer
barrier and remove the end-of-iteration producer barrier, following v063.
The two stages have disjoint indices, and reuse is protected by PV completion.
Probability arithmetic and MMA ordering remain as v085. This tests composition
of two small improvements; full bitwise comparison with v085 and memcheck will
be required if timing improves. Compilation and tests pending.

### Iteration 085 sustained timing and explicit accuracy limits

512-row Graph validation passes: warm 1888.18 us versus TRT 1877.54 us;
cold 1887.87 us versus TRT 1928.16 us. These are paired measurements with
unlocked clocks: approximately tied warm and 2.1% faster cold in this run,
not proof of a consistent warm speedup. Qualified b512 device memcheck
(--report-api-errors no, as documented for this CUDA-Python environment)
reports zero errors; its two sampled numerical rows also pass.

The second full 8192-row seed5678 audit records six failures for both v085 and
TRT, again with identical failing coordinates/values. max_abs 0.018071592,
relative_RMSE 0.01498275861 versus TRT 0.01498275871; 84654 BF16 outputs differ
(99.9685% equal). Neither full audit is a strict tolerance pass.

The extra internal-hole/partial-tile input fails: 86/65536 elements for v085
and v054, versus 78/65536 for TRT. Their max_abs values are 0.021799020 and
TRT 0.025040984. v075 passes this input with max_abs 0.004001856. The fast-path
agreement on the target workload must not be extrapolated to this case.
The original failed validate_masks log is retained; audit_mask_precision.py
adds paired diagnostics without suppressing failures or changing tolerance.

## Iteration 088 — queue next QK after current PV commit

Based on v085. Issue QK0 in a prologue. In each subsequent iteration warp0
issues current PV and commits its completion barrier, then immediately queues
QK(i+1) once the next KV stage is ready, before the compute warps wait and
reconverge after PV. QK and PV have separate alternating-phase barriers.
The next QK reuses the score buffer only after all compute threads have
completed its TMEM loads and synchronized before current PV.

Unlike v043/v074's QK-before-PV lookahead, this preserves PV priority and the
eight-producer-warp layout. It targets the gap around PV completion, producer
stage release and next-loop QK issue. Waiting for next KV could instead delay
warp0's reconvergence and hurt producer progress; this is an explicit risk to
measure. Arithmetic and per-output MMA accumulation order are unchanged.
Offline resources, bounded smoke, full bitwise comparison and device memcheck
are required before any promotion. Pending.

### Iteration 086 result — near parity with TRT and closer output agreement

Offline REG119/STACK0, with 16 packed FFMA2 instructions in SASS. Smoke and
eight-row checks pass. Warm short events 1870.05 us versus TRT 1691.68 us.
NCU: 119 registers, 194872 shared bytes, occupancy 23.286%, tensor active
27.894%, eligible 0.470247, long-scoreboard 6.020044, zero local sectors,
aggregate shared conflicts 3537829/4511887, diagnostic 3.168512 ms.

512-row Graph: warm 1873.54 us versus TRT 1869.82 us (0.9980x); cold 1880.13 us
versus 1946.53 us (1.0353x). These runs establish near parity warm and a cold
advantage in this measurement, not a consistent warm speedup. The event/Graph
baseline difference remains visible; do not mix ratios across timing regimes.

Both full 8192-row seeds have the same failures as TRT: nine for1234 and six
for5678, including every failing coordinate and actual value. Relative_RMSE
0.01497933883/0.01498275746; max_abs 0.019369811/0.018071592. Only2503/2672
of268435456 BF16 outputs differ from TRT, over99.999% bitwise equal. This is
a strong equivalence observation for these inputs, not a full FP32 tolerance pass.

The full1024-row chunk0/seed5678 test has7650 tolerance failures for both,
max_abs0.057887435, relative_RMSE0.01523154806 versusTRT0.01523154989; only300
of33554432 outputs differ. The ten stored failure examples match; the audit
does not store every failure coordinate in this case. The masked input still
fails86 elements versus TRT78. Qualified b512 memcheck reports zero errors,
with both sampled numerical rows passing.

Promote v086 as the target-workload baseline-precision performance candidate,
with these explicit numerical limits. v075 remains the independently audited
higher-precision option. Neither candidate establishes a sustained warm win.

### Iteration 087 result — small event gain does not improve sustained timing

Offline REG119/STACK0. Warm1872.22 us versusTRT1693.86 us; NCU:119 registers,
194872 shared bytes, occupancy23.284%, tensor27.850%, eligible0.528822,
long-scoreboard5.896641, zero local sectors, aggregate shared conflicts
3973729/3797282, diagnostic3.173920 ms. Full8192-row seed1234 output matches
v085 bitwise in all three repeats. Graph warm1889.58 us versusTRT1867.81 us,
cold1882.14 us versusTRT1923.95 us. This does not improve on v085's sustained
warm result. No separate short/masked/memcheck or second-seed claim; no promotion.

v088 preflight: offline REG112/STACK0. Bounded smoke and device memcheck are
running before the full performance/NCU evaluation.

### Iteration 088 result — improved low-clock profile, worse measured latency

Smoke/eight-row checks and qualified b512 memcheck pass. Full8192-row seed1234
outputs match v085 bitwise in three repeats. Warm events2037.79 us versus
TRT1690.85 us regress. Graph confirms warm2037.95 us versusTRT1872.11 us,
cold2045.87 us versusTRT1924.62 us. No promotion.

NCU at its diagnostic clock regime gives3.078816 ms, lower than v085's3.182944 ms,
with112 registers,194880 shared bytes, occupancy23.446%, tensor28.719%, eligible
0.553978, long-scoreboard4.584788, zero local sectors and aggregate shared
conflicts435266/2654711. Improved profiled duration/stall ratios do not establish
an unprofiled speedup. A rotating-order same-process benchmark of v086/v088/TRT
will check this discrepancy and the small v086/TRT gap without changing clocks.

bench_round_robin.py reuses the original measure_case implementation, rotates
backend order each round, checks512 rows before timing, and records raw samples
plus GPU telemetry at measurement endpoints. Endpoint clocks do not reveal
frequency throughout a kernel. This supplemental experiment must remain separate
from the original paired results and cannot replace them selectively.

### Rotating-order audit — v086 ties warm; cold advantage does not reproduce

Three rounds in one process rotate all three backends through each position.
All512 sampled rows pass. Warm medians: TRT1872.00/1871.82/1871.87 us;
v0861871.92/1872.00/1872.16 us; v0882037.79/2037.89/2037.84 us. This supports
warm parity for v086 and confirms v088's regression independently of order.
Cold medians: TRT1857.04/1857.66/1855.78 us; v0861878.30/1878.19/1878.13 us;
v0882045.86/2045.89/2045.90 us. Earlier cold advantages do not reproduce and
must not be claimed as robust. Both sets of raw evidence remain available.

GPU clock endpoints for v086/v088 are2032 MHz after every measurement; TRT
endpoints vary1710–2025 MHz. These sparse snapshots cannot attribute in-kernel
frequency changes. Default NCU's lower diagnostic clock regime can change the
relative cost of compute and memory. Profile with --clock-control none next to
check whether the v088/v086 ranking and stalls follow the measured runtime.

## Iteration 089 — remove the post-PV compute barrier

Based on v086. Every compute warp retains the PV-completion mbarrier wait and
tcgen05 after-thread-sync fence. Remove the following256-thread named barrier;
thread0 releases the consumed KV stage immediately after the before-thread-sync
fence. The pre-PV barrier already drains each thread's current index/P reads,
and PV completion drains asynchronous KV reads. Next-loop compute uses the
other KV stage. Cross-group max reduction and pre-PV synchronization remain.

This tests whether earlier stage release and one fewer per-tile rendezvous
improve producer overlap. Arithmetic is unchanged. Barrier/lifetime reasoning
must be backed by bounded smoke, device memcheck and full bitwise comparison;
no numerical or safety claim is inherited solely from the small diff. Pending.

### Profiler regime audit — clock control and Tensor Core boost are separate

Installed NCU2025.3 CLI reports default --clock-control base and default
--pipeline-boost-state stable. NVIDIA's GPU tools expert explains that supported
GPUs normally use dynamic Tensor Core boost and that NCU disables the dynamic
option for more consistent multi-pass replay ([NVIDIA expert explanation](https://forums.developer.nvidia.com/t/tensor-core-boost-state-api/340228/2)).
The current online UI documentation uses an auto default, so the installed
CLI help, rather than a newer release's default, is authoritative for this run.

Preserve existing base/stable results. profile_regimes.sh reproduces supplemental
none/stable, base/dynamic and none/dynamic profiles for v086/v088, changing the
two controls independently. Collect SpeedOfLight, SchedulerStats and WarpStateStats
with the same10 warmups and one profiled invocation as profile_ncu.py. Clock and
boost variants are diagnostic controls, never replacements for unprofiled timing.
The matrix is running; no causal attribution or performance claim yet.

### Profiler regime result — ranking reversal follows clock control

| Clock / TC boost | v086 diagnostic ms | v088 diagnostic ms |
|---|---:|---:|
| base / stable (original) | 3.168512 | 3.078816 |
| base / dynamic | 3.165664 | 3.074496 |
| none / stable | 1.870112 | 2.039680 |
| none / dynamic | 1.869984 | 2.039232 |

The ranking follows the clock-control variable. Switching TC boost modes has
negligible measured impact in these profiles; do not attribute this reversal
to Tensor Core boost. Reported GPC frequency is ~1.090 GHz with base control
and ~1.902 GHz without it; DRAM remains ~3.996 GHz. None/dynamic reproduces
the unprofiled latency ordering. This is evidence of clock-regime sensitivity,
not yet proof of a particular memory/compute dependency causing that sensitivity.

At none/dynamic, v086/v088 tensor-active is27.076%/24.810%, eligible-warp
0.493968/0.446367 and long-scoreboard5.935144/5.448281. Even here v088's lower
stall ratio coexists with worse total latency. Original/default profiles remain
useful resource diagnostics but cannot alone rank these pipelines.

### Iteration 089 result — safe on audited cases, sustained gain unestablished

Offline REG119/STACK0. Smoke/eight-row checks and qualified b512 memcheck pass.
Warm events1861.44 us versusTRT1691.74 us. NCU base/stable:119 registers,
194872 shared bytes, occupancy23.282%, tensor28.067%, eligible0.472762,
long-scoreboard6.019016, zero local sectors, aggregate shared conflicts
3460674/4528813, diagnostic3.147072 ms.

All8192-row seed1234 output bits match v086 in three repeats, as do all1024
short-case rows at seed5678 in three repeats. The masked audit retains86
failures, with the same reported errors as v086; neither passes its FP32
reference tolerance. Graph warm1882.21 us versusTRT1871.89 us, cold1871.87 us
versusTRT1927.14 us. Its short-event gain does not establish a sustained warm
gain; keep v086 as the promoted fast path pending any further controlled evidence.

## Iteration 090 — one validity bitmap load per score fragment

Based on v086. Each of the four index-loading warps ballots its32 validity
predicates, and lane0 stores a Uint32 bitmap. Existing producer synchronization
publishes those bits before TMA completion. Each compute thread loads the bitmap
for its aligned32-key fragment once, then uses constant bit tests instead of32
shared-memory index loads. Four words per stage add32 shared bytes.

The audited Layout-E copy has physical DP=local thread ID and column=element
index, so bitmap index is stage*4+(ctid//64)*2+cgroup and bit index is j.
Unlike v080 there is no all-valid branch or cross-warp flag reduction. Internal
holes and partial tiles follow the same per-element predicate as v086. Compile,
bounded smoke, masked comparison, full equivalence and profiling are pending.

### Iteration 090 initial result — validity bitmaps improve the fast path

Offline REG118/STACK0. Smoke/eight-row checks and qualified b512 memcheck pass.
Warm events1828.80 us versusTRT1691.49 us. NCU base/stable:118 registers,
194904 shared bytes, occupancy23.269%, tensor28.584%, eligible0.475524,
long-scoreboard6.176494, zero local sectors, aggregate shared conflicts
4152780/5476619, diagnostic3.092896 ms. The hardware conflict totals increase
while latency decreases; they remain inappropriate as isolated address-layout
or performance rankings.

All8192 rows match v086 bitwise in three repeats at each of seeds1234 and5678;
all1024 short-case rows match at seed5678 in three repeats. The explicit masked
input also matches every v086 output bit, retaining its86 reference failures
versusTRT78. Thus the audited baseline-precision limits transfer exactly for
these inputs; no strict all-row FP32 pass is claimed.

512-row Graph: warm1853.31 us versusTRT1867.95 us, cold1839.36 us versusTRT1931.57 us.
The rotating-order audit gives warm v0901831.18/1831.63/1831.07 us versus paired
TRT1867.86/1876.34/1878.13 us, and cold1838.98/1837.39/1838.82 us versusTRT
1857.84/1855.66/1859.52 us. Each of the three orderings preserves an advantage.
This supports an observed sustained Graph warm speedup of about0.8–2.6%, with
baseline-level FP8 precision. The short five-event tuning run remains slower
than its exceptionally fast paired TRT measurement; do not mix timing regimes.
A20-warmup/100-repeat eager-event run will additionally match the original
benchmark execution regime before updating the fast-path recommendation.

## Iteration 091 — combine bitmaps and early stage release

Based on v090, apply v089's removal of the post-PV compute rendezvous while
retaining all PV waits and required tcgen05 fences. Probability arithmetic
and bitmap semantics are unchanged. Test whether stage-release latency now
composes with reduced mask-load overhead. Full bitwise/memory checks are
required for any promotion. Compilation and measurement pending.

## Iteration 092 — validity bitmaps on the higher-precision path

Based on v075, publish and consume the same per32-key validity bitmaps as v090.
Keep v075's score scaling, bounded anchor, residual FP8 probabilities, denominator
order and all MMAs unchanged. This isolates the memory-access optimization from
v086's raw-score reassociation. Require full output equivalence to v075 on both
seeds, short and masked cases, plus device memcheck before promotion. Pending.

### Iteration 090 promotion — original eager-event regime also improves

With20 warmups/100 repeats and512 reference rows, eager CUDA events measure
warm1859.63 us versusTRT1882.35 us (1.0122x) and cold1845.01 us versus
TRT1919.04 us (1.0401x). This complements the Graph and rotating-order results;
none replaces the preserved short-run result. Promote v090 as the current
baseline-precision fast path for the specified workload. v075 remains the
higher-precision option. Original tolerance is unchanged; shared FP8 accuracy
failures and the more difficult masked/short cases remain explicit.

v091/v092 preflight: REG118/STACK0 and REG96/STACK0 respectively. Their source
SHAs and compiled resources are saved. Runtime validation is in progress.

### Iteration 091 initial result — small compositional gain

Smoke/eight-row checks and qualified b512 memcheck pass. Full8192-row seed1234
outputs match v090 bitwise in three repeats. Warm events1820.90 us versus
TRT1691.81 us, about8 us below v090. NCU base/stable:118 registers,194904 shared
bytes, occupancy23.274%, tensor28.731%, eligible0.477240, long-scoreboard6.175109,
zero local sectors, aggregate shared conflicts4147358/5428684, diagnostic3.077440 ms.
Further full-seed/short/masked and sustained/order-rotation checks are pending.

### Iteration 092 result — bitmaps do not improve the residual path

REG96/STACK0. Smoke/eight-row checks and qualified b512 memcheck pass; all8192
seed1234 output bits match v075 in three repeats. Warm events2005.18 us versus
TRT1691.81 us, slower than v075's1990.94 us. NCU base/stable:96 registers,
203096 shared bytes, occupancy23.305%, tensor38.146%, eligible0.541818,
long-scoreboard6.816829, zero local sectors, aggregate shared conflicts
3732829/3215774, diagnostic3.410016 ms. No promotion or expanded-seed/edge claim.
Reducing mask loads and register count is insufficient to predict end-to-end
performance when the residual-probability schedule is different.

## Iteration 093 — all-valid fragment bypass using the existing bitmap

Based on v090. If a compute warp's32-key bitmap is all ones, keep its raw
score fragment without per-element predicates/stores. Otherwise execute the
identical bit-test mask loop. Each warp shares the same fragment bitmap, so
this branch is uniform. No extra producer votes, flag reduction or barrier is
added, unlike v080's CTA-wide all-valid test. Interior holes and partial tiles
still take the precise mask path. Compile, masked/bitwise/memcheck and timing
validation are pending; no assumption about contiguous valid endpoints is used.

### Iteration 091 promotion — compositional gain repeats across backend orders

All8192-row seed5678 outputs and all1024-row short-case seed5678 outputs match
v090 bitwise in three repeats each. Combined with the seed1234 full audit,
masked bitwise equality and qualified memcheck, this transfers v090's documented
accuracy limits on the audited inputs without altering tolerances.

512-row extended eager events: warm1852.51 us versusTRT1886.94 us, cold1851.06 us
versusTRT1917.02 us. Graph: warm1847.62 us versusTRT1863.94 us, cold1832.43 us
versusTRT1926.14 us. Rotation warm medians are v0911822.98/1824.50/1824.83 us,
v0901832.45/1832.91/1833.10 us, TRT1869.98/1875.98/1879.65 us. Cold medians
v0911830.98/1830.94/1830.98 us, v0901838.37/1838.14/1839.01 us, TRT1855.63/
1857.44/1861.79 us. v091 improves on v090 in every recorded ordering. Promote
v091 as the current baseline-precision fast path; keep v075 for the separately
audited higher-precision option. The five-event tuning run still favors TRT.

v093 preflight: REG124/STACK0. Bounded runtime, masked comparison and memcheck
are in progress before the full benchmark and equivalence check.

### Iteration 093 result — all-valid branching regresses

Smoke/eight-row checks and qualified b512 memcheck pass. The masked case matches
v090 bitwise, as do all8192 seed1234 output elements in three repeats. Warm
1876.19 us versusTRT1691.84 us is slower than v090's1828.80 us. NCU base/stable:
124 registers,194904 shared bytes, occupancy23.289%, tensor27.792%, eligible
0.514020, long-scoreboard5.818145, zero local sectors, aggregate shared conflicts
3777818/3725962, diagnostic3.177664 ms. No promotion or expanded accuracy/Graph
claim. Removing predicates through a branch does not preserve the original
compiler schedule or guarantee lower latency.

## Iteration 094 — four producer warps with eight compute warps

Based on v091. Reduce the CTA from512 to384 threads, retaining256 compute threads
and128 index/TMA producers. Four producer warps each issue eight gather4 row groups
instead of eight warps issuing four groups. Every one of128 rows still receives
four main-column transfers and one tail transfer; producer barriers count128.
Compute synchronization, bitmap mapping, Q TMA, shared memory and all arithmetic
remain unchanged. This tests the producer issue concurrency needed after the
softmax-mask and stage-release optimizations. It also reduces resident warp count
at the shared-memory-limited one-CTA occupancy, which may hurt latency hiding.
Offline resources, bounded smoke, memcheck and full equivalence pending.

## Iteration 095 — all CTA threads copy the final output

Based on v091. After compute threads finish the final PV and fill the shared
BF16 output transpose, join completed producer and compute roles with a full
CTA barrier. All512 threads then copy eight128-bit vectors each to global
memory, replacing256 compute threads copying16 vectors each. A second full
barrier precedes TMEM deallocation. The arithmetic/normalization and shared
transpose remain unchanged; no output is computed by producer threads.

The output covers offset=(tid+v*512)*8 for tid0..511 and v0..7, exactly32768
BF16 values. This trades broader store concurrency against full-CTA epilogue
synchronization and altered register lifetimes. The final producer TMA has
already completed before compute's final PV, so no stage can still be written
when the shared output is read. Offline, bounded smoke, full bitwise and device
memory checks are pending. This version is independent of v094's CTA-size change.

### Iteration 094 initial result — near tie with fewer producer warps

Offline REG120/STACK0. Smoke/eight-row checks and qualified b512 memcheck pass;
all8192 seed1234 outputs match v091 bitwise in three repeats. Warm1816.99 us
versusTRT1691.71 us is only about4 us below v091. NCU base/stable:120 registers,
194904 shared bytes, occupancy17.846%, tensor28.744%, eligible0.398417,
long-scoreboard5.452995, zero local sectors, aggregate shared conflicts
4947972/4466142, diagnostic3.075072 ms. The expected occupancy reduction is
measured; a sustained gain has not yet been established. No promotion yet.

v095 preflight: REG118/STACK0. Runtime and equivalence tests are in progress.
After it finishes, collect a fresh v091 source-level profile with clock-control
none and pipeline-boost-state dynamic to guide subsequent optimization in the
clock regime that reproduced unprofiled latency ordering.

### Iteration 095 initial result — small output-copy gain

Smoke/eight-row checks and qualified b512 memcheck pass. All8192 seed1234 output
bits match v091 in three repeats. Warm1816.64 us versusTRT1691.84 us, only about
4 us below v091. NCU base/stable:118 registers,194904 shared bytes, occupancy
24.833%, tensor28.811%, eligible0.476426, long-scoreboard6.214066, zero local
sectors, aggregate shared conflicts4145626/5649830, diagnostic3.069664 ms.
Higher achieved occupancy partly reflects keeping producer warps alive through
the epilogue and does not itself prove more useful work. A four-order comparison
with v091/v094 is being inspected before any promotion.

### v091 source profile at unlocked clocks

With clock-control none / pipeline-boost-state dynamic, source counters report
28655616 shared wavefronts, exactly28655616 ideal and zero excessive. The top
three long-scoreboard sites account for41120/61115 samples (67.28%): producer
stage-reuse wait17172, QK completion14190, and PV completion9758. These samples
are attributed at consumer branches, not direct bandwidth or speedup estimates.

The next site has7895 samples at the ISETP consuming a loaded global index.
Global index fetch currently occurs only after the producer's stage-reuse wait.
This suggests a bounded scheduling experiment: issue that independent load
before waiting while delaying shared publication until the stage is free.
Raw CSV, a reusable summary script, and the derived context excerpts are saved.
Short-scoreboard hotspots also occur at accumulator-correction FMUL2 consumers
of TMEM loads; MIO samples are mainly on gather4 instructions. No new address-
bank-conflict hypothesis is supported by this profile.

## Iteration 096 — overlap index fetching with stage-reuse waiting

Based on v091. Load the current tile's global index into a private register
before the empty-stage mbarrier wait. Publish the index and its validity ballot
to shared memory only after the wait, preserving every stage-lifetime guarantee.
The input index array is read-only throughout a kernel launch. Arithmetic,
bitmaps, TMA issue count and compute synchronization are unchanged.

Inspect generated SASS to verify the load remains before the phase check;
the compiler may sink it back. Then run bounded smoke, memory/masked and full
bitwise checks before evaluating the performance hypothesis. Pending.

### Four-order comparison of v091/v094/v095

All512 sampled rows pass. Warm per-round medians (us):
TRT1867.98/1867.92/1870.10/1868.06;
v0911823.14/1824.75/1824.99/1825.09;
v0941819.04/1819.97/1819.52/1819.62;
v0951820.80/1820.98/1821.02/1821.02.
Cold v0911830.82–1831.01, v0941825.94–1826.90, v0951826.66–1826.86, versus
TRT1855.39–1857.50. Both changes have small repeatable gains in this audit;
v094 is slightly faster warm and uses the original compute-only epilogue.
Complete its second full seed, short/masked equivalence and extended eager/
Graph runs before promotion. Do not infer significance beyond these measurements.

v096 compiles REG118/STACK0. Inspect the index load's placement relative to the
producer phase check before interpreting the scheduling hypothesis as implemented.

### Iteration 094 promotion — additional validation and original-regime timing

All8192 seed5678 and all1024 short-case seed5678 output bits match v091 in
three repeats each, in addition to the seed1234 full audit. The masked input
also matches bitwise, retaining the known baseline-precision failures. Qualified
b512 memcheck already reported zero errors.

512-row20/100 eager events: warm1839.30 us versusTRT1874.00 us, cold1837.06 us
versusTRT1923.28 us. Graph: warm1839.20 us versusTRT1865.82 us, cold1827.62 us
versusTRT1927.14 us. Together with the four-order audit, the small gain over
v091 repeats in the recorded regimes. Promote v094 as the current fast path,
with all inherited FP8 precision limits explicit. v075 remains the higher-
precision option. v095's gain is slightly smaller and its further validation
is not needed for promotion while v094 is preferred.

v096 SASS confirms the producer LDG at0x08a0 precedes its stage-reuse phase
check at0x0920; the compiler did not sink it back. Offline REG118/STACK0.
Runtime and equivalence validation is in progress.

### Iteration 096 initial result — overlapping index latency improves throughput

Smoke/eight-row checks and qualified b512 memcheck pass. All8192 seed1234 output
bits match v091 in three repeats; the masked case also matches bitwise. Warm
1761.38 us versusTRT1691.74 us improves about3.3% from v091's1820.90 us short run.
NCU base/stable:118 registers,194904 shared bytes, occupancy23.133%, tensor29.315%,
eligible0.471978, long-scoreboard6.324691, zero local sectors, aggregate shared
conflicts3545270/648000, diagnostic3.019136 ms. The higher normalized stall ratio
does not negate lower total latency. Sustained original-regime/Graph/rotating-order
and second-seed/short equivalence validation are running.

## Iteration 097 — pre-wait index fetching with four producer warps

Compose v096's independent global-index fetch scheduling with v094's384-thread
CTA and four producer warps. Preserve all index publication, validity bitmap,
TMA transaction and compute-lifetime rules. Each producer warp handles eight
gather4 groups. This tests whether the small producer-count gain survives the
larger scheduling improvement. Offline resources, bounded smoke, memory checks
and full equivalence are pending; no timing or safety claim is inherited.

## Iteration 098 — pre-wait index fetching on the higher-precision path

Based on v075, move only the read-only global index load before the producer
stage-reuse wait, retaining shared publication after it. Keep the original
per-score masks, residual FP8 math, all compute synchronization and eight
producer warps. This isolates v096's producer scheduling change from bitmap
and compute-barrier changes that did not improve v075. Require generated-load
placement evidence and full output equivalence before inheriting any numerical
claim. Offline, memory and runtime checks are pending.

### Iteration 096 promotion — sustained gain and full equivalence

All 8192-row output bits match v091 on seeds 1234 and 5678, as do all 1024 rows
of the short chunk0/seed5678 case, with three repeats each. Masked output bits
also match, and qualified b512 memcheck reports zero errors. The fast path retains
the documented baseline FP8 precision failures; no tolerance was changed.

512-row 20/100 eager events: warm 1806.46 us versus TRT 1886.18 us, cold
1795.95 us versus TRT 1916.88 us. Graph: warm 1806.54 us versus TRT 1880.22 us,
cold 1796.08 us versus TRT 1920.86 us. Rotating orders give warm v096
1771.66/1771.74/1769.98 us, v094 1820.78/1820.96/1820.93 us, and TRT
1869.81/1878.22/1872.14 us. Cold v096 1767.50/1767.63/1769.44 us, v094
1826.99/1827.06/1826.93 us, and TRT 1857.25/1864.61/1857.49 us. The gain
survives every ordering and both extended execution regimes. Promote v096 as
the current baseline-precision path; v075 remains the higher-precision option.
Short five-event tuning still favors TRT, so retain the regime distinction.

v097/v098 offline compilation: REG120/STACK0 and REG118/STACK0 respectively.
v097 bounded smoke, memory and equivalence checks precede its paired event/NCU
run. v098 runtime validation follows sequentially on physical GPU1.

### Iteration 097 initial result — producer-count gain composes

Smoke/eight-row checks and qualified b512 memcheck pass. All 8192 seed1234
output bits and the masked case match v096; the full-target comparison uses
three repeats. Warm 1749.06 us versus TRT 1691.90 us is about 12 us below v096.
NCU base/stable: 120 registers, 194904 shared bytes, occupancy 17.764%, tensor
29.515%, eligible 0.362764, long-scoreboard 5.447457, zero local sectors,
aggregate shared conflicts 4063389/682505, diagnostic 2.997440 ms. Extended
validation and rotating-order comparisons are pending before promotion.

The v098 masked audit exits nonzero because it includes the known 78 TRT
failures. Preserve its complete report and inspect candidate-specific pass and
bitwise-equivalence fields before resuming; do not treat that wrapper exit as
a new kernel failure or ignore unexpected candidate errors.

## Iteration 099 — warp-local register exchange for TMA indices

Based on v097. Each of the four producer warps already owns 32 global indices
in lane-private registers. Broadcast the four indices for each gather4 group
with four warp shuffles before electing its one TMA issuer. All 32 lanes execute
each shuffle, with sources 0..31. Remove the now-unused shared index array and
its stores/loads, retaining shared validity bitmaps, all three producer barriers,
stage waiting and single-thread raw-PTX TMA issuance. Expected traffic reduction
may be offset by 32 shuffle instructions and repeated elect/reconvergence per
warp/tile; offline and runtime evidence are pending.

The installed DSL exposes shuffle_sync with full-warp mask and lane-index
semantics in cute/arch/nvvm_wrappers.py. The official TMA documentation describes
single-thread issuance and implicit election for its high-level copy API:
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html
This kernel uses raw PTX helpers and retains explicit election; it does not wrap
the high-level TMA copy API in an additional election.

## Iteration 100 — four producer warps on the higher-precision path

Based on v098. Retain all 256 compute threads, residual FP8 math, original
per-score masks and synchronization, but reduce producers from 256 to 128
threads. Each producer warp issues eight gather4 groups rather than four,
covering the same 128 rows. This tests whether v097's producer-count gain also
helps the higher-precision path. Offline, bounded smoke, qualified memcheck,
bitwise equivalence and paired timing are pending.

### Iteration 097 promotion — four producer warps remain beneficial

All 8192-row outputs on seeds1234/5678 and all1024 short-case outputs match v096
bitwise in three repeats each. Masked equality and qualified b512 memcheck also
pass. Eager20/100 warm/cold medians are1787.86/1784.90 us versusTRT1883.94/
1929.81 us; Graph1794.08/1783.62 us versusTRT1869.82/1918.70 us. Rotating
orders give warm v0971759.42/1757.28/1761.38 us, v0961771.58/1771.60/1771.63 us,
TRT1867.87/1876.13/1878.16 us. Cold v0971755.33/1759.06/1760.75 us improves
on paired v0961767.82/1769.26/1772.58 us; TRT1855.68/1863.76/1857.50 us.
Promote v097 as the current baseline-precision path, retaining every documented
FP8 accuracy limit and the distinction from short tuning timings.

### Iteration 098 initial result — index overlap also helps residual FP8

Bounded smoke/eight-row checks, all8192 seed1234 bitwise comparison with v075
(three repeats), masked exact equality and qualified b512 memcheck pass. Both
v075/v098 pass the masked FP32 tolerance; only the included TRT baseline fails
there. Warm1924.90 us versusTRT1691.71 us improves from v0751990.94 us.
NCU base/stable:118 registers,203064 shared bytes, occupancy23.172%, tensor
39.405%, eligible0.534069, long-scoreboard6.488646, zero local sectors,
aggregate shared conflicts3345307/644141, diagnostic3.299680 ms. Further
full-seed/short/extended/rotation evidence is being inspected before promotion.

### Iteration 098 promotion — full accuracy inheritance and sustained gain

All8192 output bits on seeds1234/5678 and all1024 short-case bits match v075
in three repeats each, with masked equality and qualified memcheck already
passing. This inherits v075's full-reference passes on those inputs. Eager
20/100 medians:2051.90/2043.68 us warm/cold versusTRT1887.10/1916.66 us;
Graph2056.35/2037.58 us versusTRT1871.82/1918.86 us. Rotating warm medians
v0982005.14/2007.23/2013.14 us versusv0752043.94/2046.02/2046.66 us and
TRT1868.77/1876.10/1878.14 us. Cold v0982009.15/2009.22/2009.02 us versus
v0752047.89/2047.81/2048.03 us andTRT1863.60/1861.68/1867.68 us. Promote
v098 as the current higher-precision path. It remains slower than TRT.

### v097 source profile at unlocked clocks

Clock-control none / pipeline-boost-state dynamic reports598327236 executed
instructions, with28655616 shared wavefronts equal to ideal and zero excessive.
Long-scoreboard samples total62518: QK completion19768, PV completion14429,
producer stage-reuse9447, and index-validity comparison9023. The last comparison
is now before the producer stage wait in generated SASS. The scheduling change
therefore overlaps index load/consumption with preceding compute work; the
source profile does not establish that the load remains outstanding during the
wait itself. Per-site sample counts are not time or speedup fractions, especially
when comparing different resident warp counts.

Correction FMUL2 instructions consuming TMEM loads account for several leading
short-scoreboard sites. Four producer warps leave register capacity for testing
64-column correction fragments that spilled in an older512-thread version.
All completion/thread fences must remain; only fragment shape is changed next.

## Iteration 101 — wider correction transfers with a 384-thread CTA

Based on v097. Change only accumulator-correction TMEM fragments from32 to64
physical columns: two Ld32x32b/St32x32b repetition64 transfers per compute
group replace four repetition32 transfers. Keep every load/store and thread
fence, per-head correction arithmetic, score transfer shape and output epilogue
unchanged. A previous512-thread experiment spilled with wider fragments;384
threads increase the per-thread register capacity at one CTA per SM. Inspect
actual resources and local traffic before evaluating this hypothesis. Pending.

v099 offline compilation uses REG80/STACK88, indicating nonzero stack storage;
v100 uses REG122/STACK0. Runtime validation and NCU local-sector counts will
distinguish the cost before either is considered for promotion.

### Iteration 099 result — shared exchange removal spills registers

Smoke/eight-row checks, full8192 seed1234 bitwise equivalence to v097 (three
repeats), masked equality and qualified b512 memcheck pass. Warm1968.35 us
versusTRT1691.71 us regresses fromv0971749.06 us. NCU base/stable:80 registers,
193880 shared bytes, occupancy17.689%, tensor26.181%, eligible0.371592,
long-scoreboard5.285820, local read/write sectors70130400/32550984, aggregate
shared conflicts57771/506731, diagnostic3.378144 ms. The1024-byte shared saving
and lower shared counters do not compensate for the generated local traffic.
No promotion or expanded timing validation.

## Iteration 102 — allow one-CTA register allocation for warp shuffles

Based on v099, set explicit min_blocks_per_mp=1 for its384-thread launch.
The approximately194KB shared footprint already limits this shape to one CTA
per SM. Test whether this launch bound avoids the compiler's80-register result
and local traffic. No explicit runtime register redistribution is requested.
Compilation requires the existing initialize-CUDA option for the DSL device
attribute query, on the selected GPU1 after idle preflight; fake inputs are
still used and compilation launches no kernel. Pending.

### Iteration 100 result — fewer producers do not help residual FP8

Smoke/eight-row checks, full8192 seed1234 bitwise equality to v098 in three
repeats, masked equality/FP32 pass and qualified b512 memcheck pass. Warm
1931.39 us versusTRT1691.58 us is slightly slower than v0981924.90 us. NCU
base/stable:122 registers,203064 shared bytes, occupancy17.767%, tensor39.258%,
eligible0.452134, long-scoreboard5.223805, zero local sectors, aggregate shared
conflicts4572493/747802, diagnostic3.311328 ms. No promotion or expanded
validation; retain v098's eight producer warps for the higher-precision path.
