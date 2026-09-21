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
