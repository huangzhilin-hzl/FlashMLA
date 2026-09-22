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

v101 offline resources are REG168/STACK56; wider fragments still spill under
the default allocation. v102's explicit one-CTA bound changes v099's allocation
to REG142/STACK0. Runtime timing/NCU must establish whether this fixes its cost.

## Iteration 103 — role register redistribution for wider correction

Based on v101, use min_blocks_per_mp=1 and request32 producer registers per
thread and224 compute registers per thread. This totals128*32+256*224=61440
registers, before any allocation granularity. Inspect the initial register
allocation and actual USETMAXREG instructions before runtime; do not launch
if the CTA allocation cannot supply those requests. The goal is to remove
v101's correction-fragment spills while preserving all its math and fences.
Producer decrease occurs after common Q setup; compute increase occurs before
the Q wait. Offline/SASS and bounded runtime validation pending.

## Iteration 104 — one producer rendezvous after index publication

Based on v097. Retain a single128-thread producer barrier after thread0
arrive/expect_tx and every producer's index/bitmap publication, before gather4
reads. Remove the earlier barrier and the post-issue barrier. The retained
rendezvous orders shared index reads and TMA issuance; each reused stage still
waits for its consumer empty barrier before any shared write or new transaction
expectation. A warp can start fetching the next tile's private indices sooner.
This revisits an earlier small barrier-count gain after the substantial index
scheduling and producer-count changes. All compute and phase-lifetime fences
remain. Offline, bounded smoke, memory and equivalence evidence are pending.

### Iteration 101 result — wider correction still spills

Smoke/eight-row checks, full8192 seed1234 bitwise equivalence to v097 in three
repeats, masked equality and qualified b512 memcheck pass. Warm1808.42 us versus
TRT1691.78 us is slower thanv0971749.06 us. NCU base/stable:168 registers,
194904 shared bytes, occupancy17.757%, tensor28.503%, eligible0.358347,
long-scoreboard5.494021, local read/write10485760/3714496 sectors, aggregate
shared conflicts4134840/847304, diagnostic3.104096 ms. No promotion.

### Iteration 102 result — removing shuffle spills is insufficient

Smoke/eight-row checks, full8192 seed1234 bitwise equivalence to v097 in three
repeats, masked equality and qualified b512 memcheck pass. Warm1814.56 us versus
TRT1692.10 us improves substantially fromv099 but remains slower thanv097.
NCU base/stable:142 registers,193880 shared bytes, occupancy17.725%, tensor
28.399%, eligible0.386531, long-scoreboard5.029150, zero local sectors, aggregate
shared conflicts2001/350518, diagnostic3.112480 ms. The remaining shuffle,
election and scheduling costs prevent promotion; lower shared counters alone
are not an optimization objective.

v103 preflight confirms REG168/STACK0 and actual USETMAXREG32/224 instructions.
Initial168*384=64512 registers covers the requested128*32+256*224=61440;
these per-warp allocations are multiples of256 registers. Bounded runtime
validation follows before full timing.

## Iteration 105 — dedicated MMA issue warp with PV-before-next-QK order

Based on v097. Retain256 compute threads and128 producers; add warp12 as a
32-thread dedicated MMA issuer (416 threads total). It waits for a full KV
stage, issues current QK, waits for compute P/correction readiness, issues
current PV, then proceeds to the next stage's QK without waiting for PV itself.
Separate QK-done, PV-done and P-ready mbarriers each alternate once per tile.

Compute still acquires the producer full barrier directly for index/bitmap
visibility, then waits QK before score reads. Its existing pre-PV256-thread
rendezvous drains every score read, P store and O correction before tid0 signals
P-ready. Every compute warp waits PV before the old KV stage is released and
before next P stores. Thus next QK writes only the disjoint score TMEM region,
reads the alternate KV stage, and cannot overwrite scores still being consumed.
Final PV completion precedes output transpose and TMEM deallocation.

Unlike v088, waiting for next KV cannot block compute-warp reconvergence at the
current PV wait. Unlike v074, current PV is queued before next QK, and the four
producer warps keep their existing128-row mapping. The extra issue warp and
readiness handshake may still cost more than the overlap saves. Offline/SASS,
bounded smoke, memory/masked and full bitwise checks are pending. No correctness
or performance claim is inferred solely from this synchronization argument.

### Iteration 103 result — redistribution removes wider-fragment spills

Smoke/eight-row checks, full8192 seed1234 bitwise equality to v097 (three
repeats), masked equality and qualified b512 memcheck pass. Warm1757.22 us versus
TRT1691.65 us recovers most ofv101's regression but is still slower thanv097.
NCU base/stable:168 registers,194904 shared bytes, occupancy17.737%, tensor
29.451%, eligible0.373962, long-scoreboard5.339017, zero local sectors,
aggregate shared conflicts4290784/943817, diagnostic3.016000 ms. No promotion.

### Iteration 104 result — fewer producer barriers do not help this schedule

Smoke/eight-row checks, full8192 seed1234 bitwise equality to v097 (three
repeats), masked equality and qualified b512 memcheck pass. Warm1755.26 us versus
TRT1691.68 us is slightly slower thanv0971749.06 us. NCU base/stable:120
registers,194904 shared bytes, occupancy17.777%, tensor29.356%, eligible0.362432,
long-scoreboard5.545522, zero local sectors, aggregate shared conflicts
3563570/676605, diagnostic3.013952 ms. No promotion or expanded timing claim.

### Iteration 105 initial result — dedicated issuing improves short timing

Offline REG85/STACK0. Bounded smoke, b2 qualified synccheck and b512 qualified
memcheck pass. Full8192 seed1234 outputs match v097 in three repeats, and the
masked output bits match. Warm1693.86 us versusTRT1691.84 us is near parity in
the short tuning regime and improves fromv0971749.06 us. NCU base/stable:
85 registers,194920 shared bytes, occupancy19.223%, tensor30.530%, eligible
0.391934, long-scoreboard5.931732, zero local sectors, aggregate shared
conflicts6238821/1956021, diagnostic2.896512 ms. Long-run and additional
full-seed/short equivalence checks are pending before promotion.

## Iteration 106 — dedicated issuing for residual FP8

Apply v105's416-thread producer/compute/MMA roles and three-barrier handshake
to v100's higher-precision math. Keep both FP8 probability terms, bounded
softmax anchor, per-score masks and the original post-PV256-thread barrier.
The dedicated issuer fills/lastuses all four B collectors for each N tile in
the same high-then-residual order, then queues next QK only after all current
PV instructions and their completion commit have been issued. Preserve direct
full-stage acquisition for shared index visibility and all TMEM fences.
The comparison baseline is validated higher-precision v098; v100's four-producer
mapping already matches v098 bitwise on the audited first seed/masked input.
Offline, bounded smoke/synchronization/memory and full equivalence are pending.

### Iteration 105 promotion — independent issue gains survive long runs

All8192 seed1234/5678 and all1024 short-case outputs match v097 bitwise in
three repeats each; masked equality, qualified b2 synccheck and b512 memcheck
also pass. Eager20/100 warm/cold medians1740.94/1768.85 us versusTRT1876.64/
1913.94 us; Graph1762.86/1742.77 us versusTRT1867.81/1916.77 us. Rotating
warm medians v1051728.06/1730.53/1726.42 us, v0971759.30/1757.30/1760.72 us,
TRT1851.44/1876.14/1880.13 us. Cold v1051714.19/1722.29/1712.43 us, v097
1761.70/1759.20/1761.26 us, TRT1859.60/1863.86/1865.79 us. The improvement
holds in every ordering. Promote v105 as the current fast path, retaining all
baseline FP8 accuracy limits. Short tuning is near TRT parity, not a proven win.

v106 compiles REG100/STACK0. Its bounded synchronization/memory, masked and
full-target equivalence checks precede paired timing on GPU1.

### Iteration 106 initial result — dedicated issuing helps residual FP8

Smoke/eight-row checks, qualified b2 synccheck/b512 memcheck, full8192 seed1234
bitwise equivalence to v098 in three repeats, and masked equality/FP32 tolerance
pass. Warm1851.68 us versusTRT1689.70 us improves fromv0981924.90 us. NCU
base/stable:100 registers,203080 shared bytes, occupancy19.263%, tensor41.229%,
eligible0.464376, long-scoreboard6.003787, zero local sectors, aggregate shared
conflicts6353209/1624138, diagnostic3.165440 ms. Extended evidence is pending.

### v105 source profile at unlocked clocks

Clock-control none / pipeline-boost-state dynamic:614231600 executed instructions,
28663808 shared wavefronts equal to ideal, zero excessive. Long-scoreboard
samples total64620, led by producer stage-reuse16884, compute PV completion
16869 and compute QK completion13859. The counts reflect both the extra resident
issue warp and different overlap; do not interpret differences as runtime
fractions. Further reduce the P-readiness handoff only with an explicit release
from every participating compute warp or thread.

## Iteration 107 — eight warp release arrivals for P readiness

Based on v105. Replace the pre-PV256-thread named barrier and tid0's single
P-ready arrival with a full-mask warp sync, retained before/after TMEM thread
fences, shared async-view fence, and one release arrival per compute warp.
Initialize P-ready with expected count8. Each warp's elected lane releases after
all32 lanes finish score reads, P writes and correction; the issuer waits for
all8 releases before any PV or next QK. Every compute warp still waits PV before
stage reuse and next P stores. Offline and bounded synchronization/memory/full
bitwise checks are pending.

## Iteration 108 — every compute thread releases P readiness

Based on v105. Initialize P-ready count256 and have every compute thread arrive
after its before-thread-sync TMEM fence and shared async-view fence. Remove
only the preceding named barrier and its after-thread-sync fence; the issuer
retains its acquire wait and after-thread-sync fence before MMA. Each thread
releases its own writes, so no representative lane needs a warp rendezvous.
This tests256 arrivals against v107's8 arrivals plus warp synchronization.
All later PV waits/fences remain. Offline and bounded validation pending.

PTX mbarrier.arrive defaults to release semantics at CTA scope, and mbarrier
wait defaults to acquire. The installed sync_warp lowers to bar.warp.sync:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
These ordering guarantees motivate the protocols; runtime tests do not replace
the required memory-order reasoning.

## Iteration 109 — control for the next-QK overlap hypothesis

Based on v105. Add a PV-completion wait and after-thread-sync fence in the
separate MMA issuer before it loops to next QK. Keep all roles, arithmetic,
P-readiness protocol and compute waits unchanged. This removes the intended
PV/next-QK issue overlap while keeping most structural/compiler changes from
dedicated issuing. A timing comparison helps separate overlap from those other
changes; an extra wait also has its own instruction cost, so the control is not
a cycle-exact decomposition. Offline, bounded smoke/equivalence and timing pending.

### Iteration 106 promotion — validated higher-precision improvement

All8192 seed1234/5678 output bits and all1024 short-case bits match v098 in
three repeats each, along with masked equality/FP32 tolerance and qualified
synccheck/memcheck passes. Eager20/100 medians1977.41/1981.01 us warm/cold
versusTRT1874.05/1918.05 us; Graph1998.90/1972.21 us versusTRT1867.82/1916.93 us.
Rotating warm v1061964.16/1964.16/1962.75 us versusv0982005.17/2007.23/2005.38 us
andTRT1867.87/1880.19/1881.60 us. Cold v1061957.58/1958.02/1955.70 us versus
v0982009.36/2008.46/2009.22 us andTRT1857.44/1857.46/1857.36 us. Promote v106
as the current higher-precision option; it remains slower thanTRT in all these
regimes while preserving the audited full-reference passes.

### Iteration 107 initial result — warp readiness saves a little latency

Offline REG85/STACK0. Smoke/eight-row checks, qualified b2 synccheck/b512
memcheck, full8192 seed1234 bitwise equality to v105 in three repeats and masked
equality pass. Warm1685.60 us versusTRT1691.90 us is about8 us belowv105.
NCU base/stable:85 registers,194920 shared bytes, occupancy19.237%, tensor
30.744%, eligible0.394402, long-scoreboard5.935145, zero local sectors,
aggregate shared conflicts6412906/1965066, diagnostic2.886592 ms. The small
short-run advantage is not yet a sustained-win claim. Expanded checks pending.

### Iteration 108 initial result — direct thread releases slightly improve timing

Offline REG85/STACK0. Smoke/eight-row checks, full8192 seed1234 bitwise equality
to v105 in three repeats, masked equality, qualified b2 synccheck and b512
memcheck pass. Warm1680.42 us versusTRT1689.89 us improves about13 us fromv105.
NCU base/stable:85 registers,194920 shared bytes, occupancy19.216%, tensor
30.749%, eligible0.394105, long-scoreboard5.970144, zero local sectors,
aggregate shared conflicts6388573/1998408, diagnostic2.877120 ms. Four-order
comparison and additional full-seed/short/extended checks are pending.

## Iteration 110 — early stage release for the dedicated residual path

Based on v106. Remove its post-PV256-thread named barrier and the redundant
following after-thread-sync fence, retaining each warp's PV wait, its
immediate after-thread-sync fence and the before-thread-sync fence before
tid0 releases the KV stage. P-ready already follows all index/score/P reads;
PV completion drains the asynchronous KV reads. The next QK uses the alternate
stage. This is the previously validated fast-path release protocol applied
alone to higher-precision v106. Bounded synchronization/memory and full
bitwise validation are pending before timing.

## Iteration 111 — direct per-thread readiness for both FP8 probability terms

Based on v106. Apply v108's count256 P-ready barrier, with every compute thread
releasing its writes after the before-thread-sync TMEM and shared async-view
fences. Both high/residual P stores and correction precede these releases.
Keep v106's post-PV256-thread barrier and all other behavior unchanged, so this
isolates readiness publication from v110's early release. Offline and bounded
synchronization/memory/full equivalence are pending.

### Iteration 109 control result — delaying next QK loses the gain

Offline REG85/STACK0. Bounded smoke/eight-row checks and full8192 seed1234
bitwise equality to v105 in three repeats pass. Warm1796.42 us versusTRT
1691.52 us is substantially slower thanv1051693.86 us. NCU base/stable:
85 registers,194920 shared bytes, occupancy19.236%, tensor28.700%, eligible
0.374448, long-scoreboard6.585509, zero local sectors, aggregate shared
conflicts4508767/856939, diagnostic3.081088 ms. This supports the value of
issuing next QK before waiting for current PV. The control also adds a wait,
so it does not isolate the saved cycles exactly. No promotion, expanded
accuracy or sanitizer claim for this control-only version.

## Iteration 112 — balanced per-thread probability sum

Based on v108. Replace the32-value probability ADD reduction with a five-level
balanced tree using the same pair order tested earlier on v084. Keep exp2,
FP8 conversion, P-ready and MMA behavior unchanged. The hypothesis is reduced
serial denominator dependencies in the dedicated issue pipeline. Summation
rounding changes, so full bitwise equivalence is not assumed; a promising
latency result requires independent full-reference audits at unchanged
atol0.01/rtol0.05 and comparison to TRT's documented FP8 limits. Offline and
bounded runtime checks are pending.

### Iteration 108 promotion — stable warm gain, mixed small cold differences

All8192 seed1234/5678 and all1024 short-case outputs match v105 bitwise in
three repeats each, retaining the known FP8 accuracy limits. Masked equality,
qualified b2 synccheck and b512 memcheck pass. Eager20/100 medians1753.09/
1766.93 us warm/cold versusTRT1876.54/1915.55 us; Graph1716.29/1734.90 us versus
TRT1872.02/1933.12 us. Separate-run eager medians do not consistently improve
on v105, so use the same-process rotation to assess the small incremental gain.

Four-order warm medians (us): v1081714.02/1720.51/1712.40/1716.26;
v1071718.40/1724.56/1718.40/1718.38;
v1051722.46/1723.71/1730.64/1730.75;
TRT1867.87/1875.50/1879.95/1876.14. v108 improves on both predecessors in each
warm ordering. Cold v1081708.18/1717.34/1719.26/1708.00 versusv1051726.64/
1714.32/1726.48/1728.46 andv1071715.30/1714.13/1715.63/1722.27 is mixed;
TRT1867.86/1860.69/1859.50/1865.79 remains slower in all four cold orders.
Promote v108 for its repeated warm gain and retained accuracy; do not claim
uniform cold improvement over v105/v107. No further v107 full-seed expansion
is needed while v108 is preferred.

### Iteration 110 initial result — small residual-path stage-release gain

Smoke/eight-row checks, qualified b2 synccheck/b512 memcheck, full8192 seed1234
bitwise equality to v106 in three repeats and masked equality/FP32 pass hold.
Warm1847.49 us versusTRT1692.26 us is about4 us belowv106. NCU base/stable:
100 registers,203080 shared bytes, occupancy19.248%, tensor41.198%, eligible
0.463342, long-scoreboard6.043707, zero local sectors, aggregate shared
conflicts7258109/1424805, diagnostic3.154688 ms. Sustained gain pending.

### Iteration 111 initial result — per-thread readiness nearly ties

The same smoke/eight-row, qualified synccheck/memcheck, full8192 seed1234 bitwise
and masked equality/FP32 checks pass against v106. Warm1850.37 us versusTRT
1693.79 us is only about1 us belowv106. NCU base/stable:100 registers,203080
shared bytes, occupancy19.253%, tensor41.142%, eligible0.463759, long-scoreboard
6.000689, zero local sectors, aggregate shared conflicts6360400/1427677,
diagnostic3.162816 ms. No promotion until same-process timing establishes gain.

## Iteration 113 — four smaller KV stages with N64 score tiles

Based on v108. Change blockK128/two stages to blockK64/four stages, keeping
416 threads, the dedicated MMA issuer and count256 P readiness. Total staged
KV bytes remain147456; P shrinks from8192 to4096 bytes. This permits four
independent gather tiles ahead of compute while doubling the number of tiles.
Extra QK instructions/synchronizations and smaller MMA N may outweigh improved
prefetch depth, so this is a measured hypothesis rather than an assumed gain.

CUTLASS tmem_frg_ws in mma_traits_sm100.hpp (M_MMA64 branch) maps each logical
N half onto a64-DP half, with N_MMA/2 physical columns. For N64, each of the
two compute groups reads16 columns; logical key=(DP//64)*32+group*16+j. Two
validity words per stage cover these64 keys. Each producer warp issues four
gather4 groups, and each PV N tile uses two K32 steps. Stage phases advance
every four blocks, independently of per-block QK/PV/P-ready phases.

Retain the full512-column TMEM allocation. After final PV all KV transactions
are drained; reuse the first two32KiB main-KV stages as the64KiB BF16 transpose
buffer. The alias bound now covers the full four-stage main allocation. This
changes reduction/quantization grouping, so bitwise equality is not assumed.
Require offline compile, bounded reference smoke, synchronization/memory checks
and independent precision audit if performance is promising. Use --block-k64.

### Iteration 112 initial result — balanced sum improves short timing

Offline REG85/STACK0; bounded smoke and eight-row FP32 checks pass. Warm
1669.31 us versusTRT1693.76 us improves fromv1081680.42 us. NCU base/stable:
85 registers,194920 shared bytes, occupancy19.218%, tensor30.989%, eligible
0.395207, long-scoreboard5.910932, zero local sectors, aggregate shared
conflicts6370234/1977881, diagnostic2.855360 ms. Independent full-reference
and masked audits have been collected and are being inspected before sustained
performance validation; numerical equivalence is not inferred from sampled rows.

### Iteration 112 precision audit — retain baseline FP8 limits

Independent full-reference audits check268435456 elements per target seed.
v112 andTRT have the same9 failures onseed1234 and6 onseed5678, with all
recorded failure coordinates/values matching (counts below the10-example cap).
Only2464/2601 BF16 outputs differ fromTRT, respectively. RelativeRMSE is
0.014979337847/0.014982757957; maximum absolute error0.019369811/0.018071592.
The full1024-row short case has7650 failures for both, with matching first10
examples (the cap prevents claiming all failing coordinates match), only275
unequal outputs versusTRT, and maxabs0.057887435. Masked outputs match v108
bitwise and retain86 failures versusTRT78. Qualified b512 memcheck reports
zero errors. Summation rounding changes are therefore not promoted as a strict
FP32 tolerance pass; extended performance validation is in progress.

### Four-order higher-precision comparison of v106/v110/v111

Warm per-round medians (us):v1061947.71/1949.22/1963.98/1964.19;
v1101943.73/1960.00/1959.79/1960.03;
v1111951.82/1962.94/1961.50/1964.10;
TRT1861.74/1876.02/1879.98/1880.13. Cold v1061959.39/1957.82/1954.83/1959.81,
v1101948.42/1949.76/1955.70/1949.70,v1111958.14/1957.78/1953.86/1959.71,
TRT1859.76/1861.04/1859.60/1857.60. v110 improves in3/4 orders for both caches,
with one warm regression; v111 has no consistent warm gain. Retain v106 while
further optimization is prioritized; neither tiny short-run gain alone warrants
promotion or inheriting unperformed second-seed/short full checks.

## Iteration 114 — balanced denominator reduction for residual FP8

Based on v106. Apply the five-level32-value sum tree from v112, keeping both
FP8 probability terms, bounded softmax anchor, four B collectors and the
original readiness/stage-release protocol. Earlier v084 did not improve the
older pipeline, but v112 improves the dedicated fast pipeline, motivating an
isolated retest here. Summation rounding changes; do not inherit bitwise or
full-reference correctness. A performance gain requires fresh full-reference
seeds, short and masked audits at unchanged tolerances. Offline/smoke pending.

v113 offline compilation succeeds with REG92/STACK0, and bounded b2 reference
smoke passes. Its N64 layout and four-stage phase protocol now proceed to
qualified synchronization/memory checks before full-target timing. v112's
extended512-row eager/Graph runs pass sampled tolerance; rotating-order evidence
is being inspected before promotion.

### Iteration 112 promotion — sustained balanced-sum improvement

Eager20/100 warm/cold medians1716.51/1745.12 us versusTRT1888.85/1915.10 us;
Graph1738.34/1722.18 us versusTRT1869.89/1919.95 us. Three-order warm medians
(us):v1121701.94/1701.89/1712.13,v1081713.34/1712.34/1712.82,
TRT1871.36/1874.05/1876.06. Cold v1121697.63/1697.66/1697.71 versus
v1081708.10/1719.46/1716.08 andTRT1859.58/1857.46/1859.60. Each recorded
ordering improves on v108, with one warm difference below1 us. Promote v112
at baseline FP8 precision, retaining the independently audited9/6 target failures,
short/mask limitations and qualified memcheck result documented above. Timing
ranges are observed round medians, not confidence intervals; clocks are unlocked.

### Iteration 113 result — four N64 stages regress

Qualified b2 synccheck and b512 memcheck report zero errors; smoke/eight-row
reference checks pass. Short warm2177.28 us versusTRT1691.65 us is substantially
slower than v1081680.42 us. NCU base/stable:92 registers,190856 shared bytes,
occupancy19.269%, tensor23.633%, eligible0.390847, long-scoreboard5.632356,
zero local sectors, aggregate shared conflicts3437457/3782797, diagnostic
3.736032 ms. Doubling the tile count also doubles QK MMA instruction count,
softmax handshakes and per-tile work; the observed deeper prefetch does not
compensate. Reject for this workload. No expanded full-reference or sustained
performance audit is claimed, and aggregate conflicts are not source-level
excessive-bank-conflict evidence.

### Iteration 114 initial result — balanced sum improves the residual path

Offline REG122/STACK0; bounded smoke/eight-row FP32 checks pass. Short warm
1828.93 us versusTRT1691.78 us improves about23 us from v1061851.68 us.
NCU base/stable:122 registers,203080 shared bytes, occupancy19.239%, tensor
41.673%, eligible0.459962, long-scoreboard5.944802, zero local sectors,
aggregate shared conflicts6474299/1451962, diagnostic3.121280 ms. Register
allocation increases from100, with no measured local traffic. Fresh full-reference
seeds/short/masked checks, qualified memcheck and sustained performance runs
are in progress; no correctness inheritance or promotion yet.

## Iterations 115/116 — balanced maximum reduction

Based respectively on fast v112 and higher-precision v114. Replace each thread's
32-score MAX reduction with a five-level fmax tree, then combine with the running
maximum. Keep the subsequent four-way cross-thread maximum, probability sum,
FP8 conversion and all pipeline synchronization unchanged. This tests the other
per-thread reduction after the measured balanced-sum gains. The source expresses
a shorter dependency tree; compiler reassociation may already remove the original
chain, so a gain is not assumed. Finite fmax reassociation has no rounding, but
full bitwise checks are still required on audited inputs before inheritance of
precision claims. Offline resource inspection and bounded smoke are pending.


### Iteration 114 promotion — independently validated higher precision

All268435456 output elements pass on each target seed1234/5678, maxabs
0.005918741226/0.005017399788, relativeRMSE0.001735969251/0.001735983890.
All33554432 short-case elements pass, maxabs0.008034229279. The masked
case passes with maxabs0.004001855850; one BF16 value differs fromv106,
confirming rounding changes rather than bitwise inheritance. Qualified b512
memcheck reports zero errors. No tolerance was changed.

Eager20/100 warm/cold1955.95/1957.86 us versusTRT1882.72/1915.01 us;
Graph1996.70/1968.24 us versusTRT1867.87/1918.93 us. Three rotated warm
medians (us):v1141939.41/1939.73/1939.68,v1061962.24/1964.11/1951.84,
TRT1869.86/1880.32/1880.13. Cold v1141927.17/1929.22/1927.15 versus
v1061958.70/1960.18/1953.76 andTRT1857.52/1859.47/1867.60. Promote v114
as the current higher-precision path for the repeated improvement in every
ordering, while explicitly retaining its remaining performance gap toTRT.

### Unlocked source profile after v112

NCU clock-control none/pipeline-boost-state dynamic: shared wavefronts
28663808 equal ideal, excessive0, instructions618706515. Long-scoreboard
sampling64353, with compute PV wait17396, producer empty wait16994, and
compute QK wait13562 (addresses and adjacent SASS retained in source CSV).
These sample counts are not runtime percentages. The observed wait sites
continue to motivate pipeline/granularity work rather than a bank-layout fix.
SASS already lowers the per-thread MAX reduction into an FMNMX3.NAN tree,
so v115/v116 are compiler-scheduling experiments; the original maximum is
not a fully serial32-operation chain. Their gain is therefore uncertain.

### Iterations 115/116 result — maximum trees do not improve

Both retain full8192 seed1234 output bits relative to v112/v114 in three repeats;
masked outputs also match (v115 retains86 failures, v116 passes), and qualified
b512 memcheck reports zero errors. Short warm v1151677.60 us versusTRT1691.84 us,
v1161835.04 us versusTRT1693.47 us regress by about8/6 us from their respective
predecessors. No promotion or second-seed/short/extended expansion.

NCU base/stable v115:85 registers,194920 shared bytes, occupancy19.224%,
tensor30.818%, eligible0.403568, long-scoreboard5.866403, zero local sectors,
aggregate shared conflicts6320404/1911987, diagnostic2.868320 ms. v116:
122 registers,203080 shared bytes, occupancy19.240%, tensor41.454%, eligible
0.473865, long-scoreboard5.909734, zero local sectors, aggregate shared
conflicts6462082/1328559, diagnostic3.140160 ms. The compiler's original
three-input MAX tree already offers parallelism; explicit binary reassociation
has no measured latency benefit in these short runs.

## Iteration 117 — N256 tiles with one KV stage in the dedicated pipeline

Based on v112. Double the score/PV reduction tile to256 keys, with one147456-byte
KV stage so shared memory stays within the SM limit. Retain416 threads and the
dedicated MMA warp. Each producer loads two indices and publishes two validity
ballots; each compute thread reads64 score columns and two mask words, using
Rep64 for scores and retaining Rep32 correction/output loads. P becomes16KiB,
QK uses M64N256, PV uses eight K32 steps per N256 output tile, and the probability
sum becomes a64-value balanced tree. Keep the512-column TMEM allocation.

This halves the number of QK instruction groups and softmax handshakes, but
single-stage KV removes copy/compute overlap and prevents next QK until PV
releases that stage. Older N256 experiments spilled in different role/compiler
configurations; the dedicated issuer now permits a separate resource retest.
Changed grouping requires independent accuracy audits if performance improves.
Offline compilation, bounded reference/synchronization/memory checks precede
full timing. Use --block-k256; no reduction in benchmark work or tolerances.

## Iteration 118 — one compute warp group in the dedicated fast path

Based on v112. Keep N128/two KV stages, but use128 compute threads,128 producer
threads and32 MMA threads (288 total). Each compute thread owns64 scores and
eight Rep32 output/correction chunks. P-ready and compute named barriers now
count128 threads, with two partial maxima/sums per head. Producer barriers
remain separate128-thread barriers. Score loads use Rep64 and output stores
still cover every64x512 BF16 value, with the physical TMEM N256 tile split
explicitly preserved in the channel mapping.

This trades less warp-level instruction/synchronization work for more registers
and work per compute thread. The two-way final denominator reduction changes
rounding, so precision is not inherited. No dynamic register redistribution or
undersized TMEM allocation is used. Offline resources and bounded validation
are pending before full performance/precision claims.

## Iteration 119 — persistent grid with the dedicated issuer

Based on v112 and the phase-carry mechanism already tested in v062. Launch at
most148 CTAs on the148-SM B300, striding across query rows while retaining the
512-column TMEM allocation and shared allocations. Initialize barriers once;
carry the absolute tile count across rows for full/empty/QK/PV/P-ready phase
selection and toggle Q's phase per query. Running maximum/sum and first-PV
accumulation still reset per query. A CTA-wide fenced barrier drains all roles
and the output transpose alias before the next query. Deallocate TMEM only
at CTA exit, with no dynamic register redistribution.

The hypothesis is lower per-query allocation/CTA setup cost in the newer
pipeline; v062's older compute/issuer schedule did not demonstrate a gain.
Register pressure and the additional CTA rendezvous can outweigh savings.
Bounded b512/chunk0 validation must cover multiple queries per CTA and varying
stage phases, followed by full bitwise audits if promising. No input data or
cross-query scores/KV values are reused. Offline resources are pending.

NVIDIA CUTLASS describes pipelining as a way to hide memory latency when large
storage requirements constrain occupancy, and explicitly discusses the tradeoff
between tile size and available concurrency. This is background for the N64,
N256 and warp-count experiments, not evidence that any particular candidate
must improve: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html

### Iteration 117 result — single-stage N256 loses overlap

Offline REG101/STACK0; bounded reference/eight-row checks and qualified b2
synccheck/b512 memcheck pass. Short warm2068.80 us versusTRT1691.58 us is
slower thanv1121669.31 us. NCU base/stable:101 registers,203096 shared bytes,
occupancy19.306%, tensor25.847%, eligible0.255136, long-scoreboard10.581799,
zero local sectors, aggregate shared conflicts180/47868, diagnostic3.415968 ms.
This version avoids the spills of older N256 candidates but still regresses;
reduced issue/handshake count does not compensate for lost KV overlap. The
large drop in aggregate shared conflicts does not imply a faster kernel.
No expanded accuracy or sustained performance audit; reject for this workload.

### Iteration 118 result — halving compute warps reduces throughput

Offline REG112/STACK0; bounded reference/eight-row checks and qualified b2
synccheck/b512 memcheck pass. Short warm1898.53 us versusTRT1692.22 us is
slower thanv112. NCU base/stable:112 registers,194408 shared bytes, occupancy
13.006%, tensor27.147%, eligible0.261547, long-scoreboard5.136814, zero local
sectors, aggregate shared conflicts6602236/1063473, diagnostic3.258784 ms.
No spills occur, but fewer active/eligible warps accompany the regression.
Retain256 compute threads; no expanded accuracy/performance audit for v118.

### Iteration 119 validation invocation correction

Offline REG128/STACK8 and b2 smoke pass. An attempted b512/check-rows512 run
failed inside the unchanged benchmark's reference selector because it includes
row512, outside b512. This is a validation invocation error, not a kernel failure;
the original log is preserved as v119_smoke_reuse_b512_reference_index.log.
The corrected b1024 reuse smoke uses two rows, followed by full bitwise target/
short comparisons and qualified b512 sanitizers with check-rows2. No benchmark
source is edited and no failed precision result is discarded.

## Iteration 120 — combine readiness and stage-release changes on v114

Based on v114. Apply count256 per-thread P-ready publication and remove the
post-PV compute barrier, retaining the before/after TC fences, shared async
fence, issuer acquire and each compute warp's PV-completion wait. Both high and
residual P terms are published by their writing threads. This combines the
v110/v111 changes after the balanced sum altered register allocation/scheduling;
the earlier isolated gains were small and inconsistent. Arithmetic is unchanged.
Require bounded synccheck/memcheck, full bitwise checks and measured performance;
no gain or precision inheritance is assumed.

## Iteration 121 — wider correction in the lower-register dedicated pipeline

Based on v112. Use two Rep64 TMEM load/multiply/store correction chunks in
place of four Rep32 chunks, while preserving the independent Rep32 output
transpose from v101. Earlier v101 spilled and v103's register redistribution
removed spills without an improvement in its older pipeline. The dedicated
issuer reduced register pressure, motivating a fresh resource/performance test.
All arithmetic, probability sums, masks and synchronization stay unchanged.
No dynamic register redistribution is requested; inspect actual allocation and
local traffic before interpreting timing. Offline/bounded validation pending.

## Iteration 122 — register-budget control for Rep64 correction

Based on v121. Add an explicit one-block launch bound and request32 registers
for the five producer/issuer warps and192 for the eight compute warps. Required
registers are160*32+256*192=54272. Do not launch until the compiled initial
allocation (including hardware allocation granularity) can support this budget
and SASS confirms the redistribution instructions. This isolates the effect
of v121's measured spills; success is not assumed from the source request.
Offline compilation requires initializing only the authorized GPU1 context.

### Iteration 119 result — persistence does not improve this pipeline

Full8192 seed1234 and full1024 short/chunk0 seed5678 outputs match v112 bitwise
in three repeats each; qualified b512 synccheck/memcheck report zero errors.
Corrected reuse smoke and eight target rows pass. Short warm1714.18 us versus
TRT1691.65 us is about45 us slower thanv112. NCU base/stable:128 registers,
194920 shared bytes, occupancy20.310%, tensor30.281%, eligible0.392454,
long-scoreboard6.137702, local load/store sectors0/7696, aggregate shared
conflicts6502984/1390528, diagnostic2.917056 ms. The small local-store count
is retained as evidence, without assigning the full regression to it. No
promotion or second-seed/extended validation.

### Iteration 120 result — combined synchronization does not improve v114

Full8192 seed1234 output bits match v114 in three repeats; masked bits match
and retain a reference tolerance pass. Qualified b2 synccheck/b512 memcheck
report zero errors. Short warm1831.14 us versusTRT1691.71 us is slightly slower
thanv1141828.93 us. NCU base/stable:122 registers,203080 shared bytes,
occupancy19.235%, tensor41.546%, eligible0.456099, long-scoreboard6.086062,
zero local sectors, aggregate shared conflicts6884712/1376198, diagnostic
3.130240 ms. No promotion or additional full-seed/extended audit.

### Iteration 121 result — Rep64 correction still spills

Full8192 seed1234 output bits and masked bits match v112, preserving its known
FP8 limits. Qualified b2 synccheck/b512 memcheck report zero errors. Offline
REG128/STACK64. Short warm1784.00 us versusTRT1691.84 us regresses. NCU
base/stable:128 registers,194920 shared bytes, occupancy19.172%, tensor28.925%,
eligible0.389745, long-scoreboard5.870859, local sectors71670496 loads /
41190648 stores, aggregate shared conflicts6973218/1846090, diagnostic
3.056992 ms. Reject this unbounded-register variant; v122 explicitly tests
whether removing spills changes the outcome.

## Iteration 123 — correction warps cover16 heads instead of32

Based on v112. Keep the entire QK/softmax/P path and Rep32 output epilogue.
Change only correction's TMEM copy to16x64b.x32, with two lanes per head and
four64-column chunks. A sparse DP layout selects0..15 of each32-DP region
for group0 and16..31 for group1. Warp-local shuffle obtains the corresponding
head's already-computed correction factor. Each warp's identity vote now covers
16 heads, allowing unaffected subsets to skip correction more often without
changing arithmetic or skipping any required output update.

CUTLASS copy_traits_sm100.hpp defines the16dp64b32x destination mapping.
The proposed offline audit checks all load/store coordinates and complete
128-DP x256-column output coverage, plus every factor's source lane. Extra
shuffle/address work can outweigh fewer active correction warps. Retain the
full512-column allocation, all TC fences and the count256 P-ready handshake.
Require layout audit, bounded reference/sanitizers and full bitwise comparison
before precision inheritance; offline compilation is pending.

### Iteration 122 compile failure and warpgroup constraint

The NVVM backend fails while compiling v122; the log supplies no detailed
backend cause. No candidate kernel was launched. Separately, reviewing its
register protocol against PTX exposes an incomplete final warpgroup in the
416-thread block. PTX requires all warps in a warpgroup to execute the same
setmaxnreg instruction; the source's register-total calculation alone is
insufficient to establish a valid redistribution protocol. Reject v122 as a
launchable candidate and retain its compiler failure. The incomplete group
is a design concern, not a proven explanation for NVVM's generic error.

Source: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg
The16-DP correction coordinate/load-store/coverage/shuffle audit passed offline
before v123 compilation. This validates mapping only, not runtime correctness.

## Iteration 124 — complete warpgroups for the register-budget control

Based on v122. Pad the CTA to512 threads so all four warpgroups are complete.
Warps0..7 request192 registers; warps8..15 request32. Only8..11 produce TMA
and warp12 issues MMA, while13..15 exit after the common setup/register step.
Keep the416-thread version's effective compute/producer/issuer work unchanged.
The required budget is256*192+256*32=57344 registers; initial allocation and
SASS must still be inspected before any launch. Additional setup/idle warps
are a changed resource cost, so this is not a perfectly isolated spill control.
Offline compilation is pending; no safety or performance claim yet.

### Iteration 123 compile correction and measured result

The initial compile lacked a proven two-column alignment for the dynamic DP
base and failed IR verification before any device launch. Add explicit alignment
facts for cgroup*(16<<16)+tile*64, always a multiple of64 columns. Preserve the
initial compiler log. Corrected offline REG85/STACK0; full8192 seed1234 output
bits match v112 in three repeats and masked bits match. Qualified b2 synccheck/
b512 memcheck report zero errors. Short warm1718.46 us versusTRT1691.74 us
regresses from v1121669.31 us. NCU base/stable:85 registers,194920 shared bytes,
occupancy19.220%, tensor30.062%, eligible0.388394, long-scoreboard6.152233,
zero local sectors, aggregate shared conflicts6335643/1948370, diagnostic
2.940992 ms. A narrower head set per correction warp does not produce an
end-to-end gain; no expanded precision/performance audit or promotion.

### Iteration 124 compile result

The padded full-warpgroup variant also fails NVVM compilation with the same
generic backend error; no candidate device launch occurs. This further prevents
attributing v122's compile failure specifically to its partial warpgroup. Retain
both logs, stop the redistribution variant, and test a launch-bound-only control
as v126. No runtime or safety claim is made for v122/v124.

## Iteration 125 — direct256-bit output stores

Based on v112. Keep all attention arithmetic, denominator sharing and TMEM
reads unchanged. Replace the final shared64KiB transpose and cooperative128-bit
writeback with each thread directly writing its head's values using two naturally
aligned256-bit stores per32-BF16 fragment. Each store covers a full32-byte sector;
the runner verifies the newly allocated output's32-byte base alignment and every
computed offset is a multiple of16 BF16 elements. Retain the final compute barrier
before TMEM deallocation. The sparse warp-wide address pattern can still cost
throughput even with complete sectors, so a gain is not assumed.

PTX st.global.v8.b32 is supported on SM100+, and CUTLASS copy_sm100.hpp provides
a256-bit no-allocation store. Use equivalent st.global.L1::no_allocate.v8.b32
inline PTX, with eight packed32-bit BF16 registers and no omitted output values.
SASS must confirm STG256; bounded checks, full bitwise comparison and qualified
memcheck precede performance claims. Source:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st

## Iteration 126 — launch bound without register redistribution

Based on v121. Add only min_blocks_per_mp=1 to the416-thread launch, without
setmaxnreg. This tests whether the backend can allocate more registers for Rep64
correction while avoiding the compile failures of v122/v124. All code paths and
work are otherwise unchanged. Compile with an initialized GPU1 context for
attribute queries; inspect actual registers, stack and NCU local traffic. No
performance or spill-elimination claim until measured.

## Iteration 127 — bound each wide correction fragment's lifetime

Based on v121. Move the TMEM store wait into the two-chunk Rep64 correction
loop, draining each chunk before the next chunk reuses its registers. This
increases the number of store waits from one to two but may reduce simultaneous
live fragments and ptxas scheduling pressure responsible for spills. No launch
bounds or register redistribution are requested, so the experiment isolates the
wait placement. Existing TC thread-sync fences, P-ready and MMA waits remain.
Offline registers/SASS and NCU local traffic must determine whether live-range
pressure actually changes; arithmetic equivalence and any gain require tests.

## Iteration 128 — direct output stores on the higher-precision path

Based on v114. Apply only v125's direct256-bit output epilogue and verified
32-byte output alignment, retaining both probability terms, the balanced sum,
bounded anchor, collector reuse and v114's synchronization protocol. Short v125
shows an approximately20 us gain without spills, motivating this independent
higher-precision adaptation. Full bitwise equivalence and sanitizers must
validate the unchanged attention arithmetic; sustained timing is required
before any promotion.

## Iteration 129 — evict-first output cache priority

Based on v125. Add only .L2::evict_first to the aligned256-bit output stores,
keeping .L1::no_allocate and all arithmetic/synchronization unchanged. Output
writes stream through512MiB per target invocation, while the gathered KV backing
store is72MiB and reused across queries. The hypothesis is that lower output
retention priority reduces competition with KV in L2; the cache hint is not a
guarantee of retention or reduced DRAM traffic. PTX explicitly permits L2
priority on .v8.b32 stores on SM100+. Inspect SASS, full bitwise validation and
measured latency before claiming any effect; collect memory counters if promising.

### Iteration 125 result and promotion — direct sector stores improve writeback

Offline REG85/STACK0; SASS confirms eight STG.E.NA.ENL2.256 instructions in the
unrolled output path. Full8192 seeds1234/5678 and full1024 short/chunk0 seed5678
outputs match v112 bitwise in three repeats each. Masked bits match; qualified
b2 synccheck/b512 memcheck report zero errors. All inherited FP8 reference
limits remain explicit. Short warm1648.86 us versusTRT1690.75 us improves
about20 us fromv112. NCU base/stable:85 registers,194920 shared bytes,
occupancy19.306%, tensor31.390%, eligible0.389172, long-scoreboard6.192877,
zero local sectors, aggregate shared conflicts6377112/1977739, diagnostic
2.818880 ms.

Eager20/100 warm/cold1720.98/1733.12 us versusTRT1890.26/1911.52 us;
Graph1736.62/1720.40 us versusTRT1871.82/1914.78 us. Three rotated warm
medians (us):v1251689.74/1688.99/1691.26,v1121700.98/1712.13/1701.02,
TRT1871.82/1873.95/1878.10. Cold v1251690.90/1693.62/1687.54 versus
v1121695.87/1699.86/1697.65 andTRT1859.52/1858.26/1859.55. Promote v125
for improvement in every recorded ordering and complete bitwise validation.
Separate-run eager/Graph differences are smaller and variable; preserve those
results rather than presenting only the largest observed gain.

Unlocked source profiling: shared wavefronts20275200 equal ideal, excessive0,
down8388608 (29.27%) from v112's28663808 after removing transpose traffic.
Instructions605733244 versus618706515; long-scoreboard samples66604 with PV
wait17445, producer-empty17107 and QK13342. Sampling totals are not latency
percentages and do not isolate the cycle cost of the removed epilogue.

### Iteration 126 result — one-block bound increases spills

Offline REG128/STACK176; full8192 seed1234 and masked bits match v112, with
qualified b512 memcheck reporting zero errors. Short warm2152.67 us versus
TRT1691.78 us is worse thanv1211784.00 us. NCU base/stable:128 registers,
194920 shared bytes, occupancy19.245%, tensor24.007%, eligible0.370749,
long-scoreboard6.481138, local sectors193305800 loads /154471792 stores,
aggregate shared conflicts5670394/1317087, diagnostic3.684576 ms. The launch
bound did not increase allocated registers or eliminate spills. Reject; no
second-seed/short/extended expansion.

### Iteration 127 result — per-chunk store wait does not remove spills

Offline REG128/STACK64, unchanged fromv121. Full8192 seed1234 and masked bits
match v112; qualified b512 memcheck reports zero errors. Short warm1786.05 us
versusTRT1689.98 us does not improve v121. NCU base/stable:128 registers,
194920 shared bytes, occupancy19.189%, tensor28.946%, eligible0.390310,
long-scoreboard5.858970, local sectors71670504 loads /41198200 stores,
aggregate shared conflicts7045151/1854044, diagnostic3.057440 ms. Reject;
the proposed live-range benefit is not supported by allocated resources/traffic.

### Iteration 128 initial result — direct stores also help higher precision

Offline REG122/STACK0. Full8192 seed1234 output bits match v114 in three repeats;
masked bits match and retain a reference tolerance pass, and qualified b512
memcheck reports zero errors. Short warm1802.37 us versusTRT1691.71 us improves
about27 us fromv1141828.93 us. NCU base/stable:122 registers,203080 shared bytes,
occupancy19.327%, tensor42.186%, eligible0.455513, long-scoreboard6.168872,
zero local sectors, aggregate shared conflicts6480575/1448818, diagnostic
3.083040 ms. Second-seed/short/synccheck and sustained performance validation
are in progress before promotion.

## Iteration 130 — evict-last priority for gathered KV

Based on v125, keeping its normal output L2 priority to isolate the input policy.
Create a fractional evict-last policy with fraction1.0 for the producer role and
pass it through .L2::cache_hint on both KV gather4 copies. Query TMA descriptors,
addresses, data, completion barriers and arithmetic are unchanged. This tests
whether prioritizing the72MiB KV backing store reduces competition from streaming
Q/output. Any additional policy/address registers and issue cost must be measured;
a hint alone does not prove improved cache residency. The policy is created once
per producer thread before the tile loop. Offline compilation and bounded/full
bitwise tests precede any performance claim.

PTX cp.async.bulk.tensor supports an optional64-bit cache policy following the
completion barrier, with .L2::cache_hint. Source:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

### Profiling caveat for v125's256-bit output stores

SourceCounters reports theoretical global STG sectors524288 for v125 versus
16777216 for v112, despite both writing the same512MiB output. The former equals
the number of warp-level STG256 instructions and is inconsistent with the full
32 active lanes'32-byte stores. Do not interpret this32x difference as actual
memory-traffic reduction; it appears to be a theoretical-source-counter limitation
for this opcode/instrumentation path. Hardware memory counters are needed to
assess cache-policy experiments. The shared-wavefront reduction comes from the
removed shared transpose and is a separate metric.

### Iteration 128 promotion — direct writeback retains higher precision

Both full8192 seeds1234/5678 and full1024 short/chunk0 seed5678 match v114
bitwise in three repeats, along with masked equality/reference pass and qualified
b2 synccheck/b512 memcheck. Eager20/100 warm/cold1941.71/1945.70 us versusTRT
1880.22/1916.98 us; Graph1970.22/1945.54 us versusTRT1871.52/1916.35 us.
Three-order warm medians (us):v1281922.99/1925.22/1923.31,
v1141939.60/1939.54/1939.66,TRT1869.34/1878.03/1882.32. Cold v128
1916.94/1916.85/1917.02 versusv1141929.12/1927.20/1927.39 andTRT1857.57/
1865.76/1857.36. Promote v128 for improvement in every recorded ordering,
retaining its remaining performance gap toTRT and the audited higher precision.

### Iteration 129 initial result — output cache priority nearly ties

Offline REG85/STACK0; SASS shows STG.E.NA.EFL2.256 in place of v125's ENL2.
Full8192 seed1234 output bits and masked bits match v125, with qualified b512
memcheck reporting zero errors. Short warm1646.72 us versusTRT1693.63 us is
only about2 us belowv125. NCU base/stable:85 registers,194920 shared bytes,
occupancy19.311%, tensor31.398%, eligible0.388990, long-scoreboard6.191804,
zero local sectors, aggregate shared conflicts6372695/1916474, diagnostic
2.815552 ms. No sustained-gain claim until rotating orders and hardware memory
counters assess the cache policy. Extra seed/short audits are not yet performed.

### Iteration 130 result — KV retention hint does not provide a consistent gain

Offline REG85/STACK0. Full8192 seed1234 matches v125 bitwise in three repeats;
masked bits also match, preserving the86 FP32-tolerance failures versus TRT's78.
Qualified b512 memcheck reports zero errors. Short warm1646.94 us versusTRT
1689.95 us nearly tiesv1251648.86 us. NCU base/stable:85 registers,194920 shared
bytes, occupancy19.344%, tensor31.448%, eligible0.411345, long-scoreboard5.988450,
zero local sectors, aggregate shared conflicts5819901/4345382, diagnostic
2.811104 ms. No second-seed/short/synccheck expansion yet.

Four rotating warm medians (us):v1251690.18/1678.74/1689.74/1689.76,
v1291687.68/1681.30/1679.41/1688.54,v1301687.57/1688.78/1689.86/1687.76,
TRT1874.05/1869.79/1871.98/1869.79. Cold:v1251687.41/1688.66/1680.96/1687.58,
v1291677.09/1679.54/1676.22/1675.33,v1301691.57/1685.50/1691.60/1681.30,
TRT1851.55/1853.41/1853.36/1853.34. KV evict-last has mixed changes in both
cache modes and is not promoted. Output evict-first wins all four cold orderings
by4.74–12.26 us, but one warm ordering loses2.56 us; keepv125 as current fast
path while extendingv129 validation. These are observed round medians, not CIs.

Hardware memory profiling uses clock-control none, pipeline-boost dynamic and
cache-control none, after ten warmups. Each version is a separate invocation;
NCU reports three replay passes. Diagnostic readings:

| Version | DRAM read GB | DRAM write MB | Global store sectors | L2 hit % | L2 read sectors | L2 write sectors | GPC GHz | Profile ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v125 | 1.567498 | 526.662656 | 16777216 | 75.269081 | 378242863 | 25169201 | 1.897222 | 1.647648 |
| v129 | 1.429414 | 526.148608 | 16777216 | 76.827832 | 372228707 | 25169083 | 1.900051 | 1.644608 |
| v130 | 1.563670 | 527.449344 | 16777216 | 75.269356 | 378339460 | 25180534 | 1.899039 | 1.642880 |

The hardware store count equals512MiB/32B for every version, resolving the
SourceCounters theoretical-sector anomaly without claiming less output traffic.
v129 lowers observed DRAM reads about8.81% and raises aggregate L2 hit rate
1.56 percentage points; this supports the cache-policy hypothesis but is not
alone proof of a stable latency gain. v130 scarcely changes either quantity.
Counter replay and separate runs limit causal interpretation of small differences.

## Iteration 131 — early score-consumption publication for QK lookahead

Based on v125. Add a count256 score-consumed mbarrier. Every compute thread drains
its score TMEM load and emits a before-thread-sync fence before arriving. The
independent MMA warp acquires this barrier before reusing the same score buffer
for the next QK. It waits for that next KV stage and submits QK before waiting for
current P/correction readiness and submitting current PV. Bootstrap QK0 and
handle the final tile without a future QK. Keep current KV release after PV and
retain all original arithmetic, masks and256-bit output stores.

This tests whether next QK overlaps softmax/correction enough to repay the extra
barrier. It can instead delay PV while waiting for future KV. Unlike older v074,
it uses a single score buffer with explicit read completion, the current four
producer warps, bitmap masks and optimized arithmetic/output; v074's failure is
a warning, not evidence that the new schedule wins. Compilation, bounded smoke,
qualified sync/memory checks and full equivalence must precede promotion.

## Iteration 132 — opportunistic QK lookahead

Based on v131, isolate the cost of blocking future-KV readiness ahead of current
PV. After acquiring score-consumed, test the next full-stage mbarrier once with
CuTeDSL mbarrier_test_wait and make the result warp-uniform with vote_all_sync.
If ready, issue next QK before current PV; otherwise issue current PV first and
then wait/issue next QK, following v125's relative MMA ordering. Retain the new
score-consumed barrier in both paths to isolate the issue-order decision. Both
paths keep an acquiring full wait, TC/shared fences and one QK commit; no tile
is omitted. This adds control/instruction footprint and may still lose even if
it removes the mandatory future-copy stall. No performance claim before tests.

The installed cutlass-dsl4.6.2 implementation maps mbarrier_test_wait to NVVM
MBarrierWaitKind.TEST (versus TRY for mbarrier_try_wait). Official semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait

### Iteration 129 expanded validation — cache hint retains numerical behavior

Full8192 seeds1234/5678 and full1024 short/chunk0 seed5678 now all matchv125
bitwise in three repeats. Masked bits match, and qualified b2 synccheck/b512
memcheck report zero errors. All inherited fast-path tolerance limitations remain.
Eager20/100 warm/cold1718.29/1720.53 us versusTRT1877.76/1911.38 us; Graph
1714.40/1697.74 us versusTRT1865.87/1915.04 us. These separately run medians
improve over recordedv125 results, but the first rotation included a warm loss;
a new six-roundv125/v129 rotation tests reproducibility before changing defaults.

### Iteration 131 result — lookahead transfers stalls to current PV

Offline REG84/STACK0; full8192 seed1234 and masked outputs matchv125 bitwise,
qualified b2 synccheck/b512 memcheck report zero errors. Short warm2037.98 us
versusTRT1691.87 us is389 us slower thanv125; reject this mandatory-lookahead
schedule. NCU base/stable:84 registers,194928 shared bytes, occupancy19.467%,
tensor27.329%, eligible0.412530, long-scoreboard7.175538, zero local sectors,
aggregate shared conflicts65339/7385278, diagnostic3.233280 ms. No expanded
second-seed/short audit or sustained timing is warranted for this slower version.

Unlocked SourceCounters:615714951 instructions, shared wavefronts20275200 equal
ideal with zero excessive. Long-scoreboard samples86701, including45441 at PV
wait (barrier offset0x2f928),22256 at producer-empty (0x2f960),3321 at issuer
future-full wait (0x2f950) and1142 at score-consumed (0x2f938). v125 PV-wait
samples were17445. This is consistent with delayed PV in the reordered schedule;
sampling alone does not isolate the exact causal latency of future-copy waits
versus extra synchronization or TC arbitration. v132 is the conditional-order
control. The sampled totals are not percentages of execution time.

## Iteration 133 — output evict-first on the higher-precision path

Based on v128, change only the32-byte output store's L2 eviction priority to
evict-first, as in v129. Preserve residual FP8 arithmetic, barriers, layouts,
vector width and alignment. The fast-path memory counters suggest less KV
competition, but the higher-precision kernel may have a different bottleneck;
measure it independently and compare output bits to v128 before any promotion.

## Iteration 134 — reuse max-reduction synchronization for score publication

Based on v132's opportunistic schedule. Replace the256-arrival score-consumed
barrier with count1, published by tid0 after the existing256-thread max-reduction
barrier and its TC fences. All score loads were drained and fenced before that
barrier, so its representative can publish their collective completion before
issuer reuse. Remove the earlier extra per-thread arrival/fence. Arithmetic and
QK/PV readiness decisions remain unchanged, but publication happens later;
it may reduce synchronization overhead at the expense of a shorter overlap
window. Validate phase reuse and all-row equivalence before ranking.

### Iteration 132 result — avoid future-copy blocking, still no net gain

Offline REG84/STACK0. Full8192 seed1234 and masked output bits matchv125;
qualified b2 synccheck/b512 memcheck report zero errors. Short warm1661.15 us
versusTRT1689.86 us recovers376.83 us fromv131 but remains12.29 us slower than
recordedv125. NCU base/stable:84 registers,194928 shared bytes, occupancy19.316%,
tensor31.127%, eligible0.397529, long-scoreboard6.005708, zero local sectors,
aggregate shared conflicts6271507/1832194, diagnostic2.838240 ms. Retain as an
issue-order diagnostic; no promotion or second-seed/short expansion. v134 tests
whether cheaper score publication removes the remaining overhead.

### Iteration 129 repeat rotation — small average benefit, not every ordering

A second same-process six-round rotation (20warm/100Graph repeats) gives warm
medians (us):v1251689.74/1690.53/1689.82/1690.82/1691.74/1681.02,
v1291683.17/1681.39/1686.72/1688.30/1687.71/1689.60,
TRT1867.76/1876.11/1876.14/1880.34/1878.18/1880.22. Cold:v1251686.64/1683.52/
1683.38/1684.51/1693.65/1683.46,v1291681.34/1677.20/1677.09/1675.39/1683.34/
1683.58,TRT1857.50/1865.71/1865.70/1859.52/1856.96/1859.70. v129 wins five
of six rounds in each cache mode, with a warm loss8.58 us and a cold near-tie
loss0.13 us. Combined with the first rotation, it wins8/10 warm and9/10 cold
round medians. This is a small empirical/cache-counter-supported improvement,
not a universal guarantee; keepv125 as the established default and retainv129 as
a fully bitwise-validated cache-policy alternative. No further repeated timing
solely to obtain a favorable ordering is needed.

### Iteration 132 source control — no early QK observed in this profile

Unlocked SourceCounters records604473721 instructions and64895 long-scoreboard
samples, with PV17425, producer-empty16944 and QK13576. Shared wavefronts20275200
remain equal to ideal, excessive0. The early-QK UTCQMMA/UTCBAR branch executes0
times; the fallback QK branch executes122880 times (8192queries *15future tiles),
bootstrap8192 and PV131072. This is direct evidence that the conditional schedule
reverts to PV-before-QK in this instrumented invocation. Profiling can perturb
readiness; do not assert that every unprofiled execution also has zero early QK.

## Iteration 135 — two-CTA resource experiment with overlapping TMEM storage

Starting from v125, use64-key KV tiles, one128-thread compute group, two producer
warps and one issuer warp (224threads). Two KV stages require73728 bytes; Q still
requires36864 bytes, P4096 bytes. Exchange producer indices by warp shuffles and
reuse the128-float partial reduction array for final sums and denominator. This
removes enough shared storage to target two CTAs per SM; actual residency depends
on compiler registers, shared rounding and TMEM allocation and must be measured.

Allocate256 TMEM columns instead of512. All512 output channels occupy columns
0..255. QK uses columns224..255, intentionally overlapping the last physical
output chunk. Before every QK after tile0, compute threads read that chunk into a
32-FP32 register shadow, drain the load and publish count128 output-saved. The
issuer acquires it before overwriting with QK. After score loads/softmax, compute
restores the shadow with the same per-head correction, even for identity
correction; the other seven output chunks retain the usual conditional rescale.
PV waits for every compute thread's P/correction publication. On tile0, PV's first
K instruction overwrites its accumulators, so no old-output restoration is needed.

Risks: the shadow extends live ranges and may spill; QK now waits for output
preservation after PV, reducing existing overlap; N64 doubles QK issue count and
online-softmax rounds. Numerical rounding differs, so v125 equivalence cannot be
assumed. Static ownership audit checks all4096 scores,32768 outputs,4096 shadow
cells and64 gathered rows per stage; it does not establish device semantics.
Offline resources, bounded smoke, qualified sanitizers, original-reference checks
and NCU occupancy precede any performance claim. No launch-bound or dynamic
register redistribution is added in this first candidate.

### Iteration 133 initial result — higher-precision cache hint nearly ties

Offline REG122/STACK0. Full8192 seed1234 output bits and masks matchv128;
qualified b2 synccheck/b512 memcheck report zero errors, and masked FP32 checks
pass. Short warm1802.05 us versusTRT1689.86 us is indistinguishable from recorded
v1281802.37 us. NCU base/stable:122 registers,203080 shared bytes, occupancy
19.337%, tensor42.265%, eligible0.455848, long-scoreboard6.156068, zero local
sectors, aggregate shared conflicts6466215/1416810, diagnostic3.077344 ms.
A three-order comparison is pending; no second-seed/short expansion or promotion.

### Iteration 134 result — fewer publications recover some overhead

Offline REG85/STACK0. Full8192 seed1234 and masked bits matchv125, qualified b2
synccheck/b512 memcheck report zero errors. Short warm1652.93 us versusTRT
1691.84 us improves8.22 us fromv132 but still trailsv1251648.86 us. NCU base/stable:
85 registers,194928 shared bytes, occupancy19.311%, tensor31.284%, eligible
0.395118, long-scoreboard6.048755, zero local sectors, aggregate shared conflicts
6293935/1866058, diagnostic2.832256 ms. No demonstrated net gain; retain as a
synchronization control. No second-seed/short/sustained expansion is claimed.

### v135 instrumentation policy

The runner supports MLA_TMEM_GUARDRAILS=1 solely for qualified correctness runs,
passing --ptxas-options=-g-tmem-access-check to CuTe compilation. Performance runs
leave it unset. This explicitly instruments the new TMEM overlap/allocation with
NVIDIA's Tensor Core access checks; instrumented latency is not benchmark data.
Reference: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#tensor-core-mma-guardrails

### Iteration 133 rotation — small consistent gain in the recorded orderings

Three-order warm medians (us):v1331920.22/1913.10/1919.65 versusv1281925.25/
1923.10/1927.25 andTRT1869.84/1878.03/1876.10. Cold:v1331914.75/1898.53/1912.80
versusv1281916.94/1916.98/1917.18 andTRT1859.54/1859.58/1859.60. v133 wins all
three recorded orders in both cache modes but still trailsTRT. Extend bitwise
validation and eager/Graph20/100 before deciding promotion; cache hints have
shown small, regime-dependent gains in the fast path.

### Iteration 135 guardrail failure — no performance result

Offline REG114/STACK0 and the assumed logical-coordinate audit pass, but the
first b2 Tensor Core guardrail run fails: both CTAs report tcgen05.mma accessing
unallocated column256 with columns0..255 reserved. The process exits; the b512,
synccheck, mask and performance commands never run. GPU1 subsequently reports
0% utilization/0MiB memory twice; no reset or other-GPU action was performed.
Retain source and complete failure log. Do not rank this version or infer safety
from the static logical-coordinate audit.

The C++ CUTLASS tmem_frg_ws M64 layout uses N/2 logical columns across128 datapaths
(mma_traits_sm100.hpp around935), matching the initial element mapping, while
this instrumented program rejects the256-column reservation. This discrepancy
requires further isolation; the present evidence does not establish whether it
is an actual MMA reservation requirement, an error in our descriptor/address
calculation, or a guardrail limitation. Do not bypass the failure to claim a
working two-CTA optimization.

## Iteration 136 — conservative allocation control

Keep v135's N64 algorithm, register shadow and all addresses, but allocate512
TMEM columns. This covers the full N-wide range from both QK base224/N64 and
PV base128/N256, even under the more conservative interpretation. It tests the
functional overlap protocol independently of the failed256-column reservation;
it cannot establish two simultaneously usable TMEM allocations per SM. Keep
Tensor Core guardrails enabled for initial checks; uninstrumented timings, if
correctness passes, are only a control for the resource experiment.

### External compiler evidence related to v122/v124

NVIDIA/CUTLASS issue3420 reports the same generic NVVM-backend failure for a
consumed tcgen05.ld.x64 combined with warpgroup register deallocation, including
SM103 observations. Its reproducer reports that replacing x64 with x32 or
removing deallocation compiles. This resembles our v122/v124 failure, but is not
proof that all cases share one root cause or that this installed4.6.2 wheel is
fixed. The public page is closed without visible resolution details. Preserve
our failed compiler logs and avoid attributing the error solely to partial
warpgroups (v124 used complete warpgroups).
Source: https://github.com/NVIDIA/cutlass/issues/3420

## Iteration 137 — paired x32 correction loads with complete register donors

Based on v125, group each pair of32-column correction chunks into64 live FP32
values: issue two native x32 TMEM loads before one load wait, apply packed
multiplication, then issue two x32 stores. Keep two such groups and the final
store drain. Unlike v121/v124, no x64 load appears in compiler IR, addressing the
specific x64/deallocation combination reported in CUTLASS issue3420. This is an
experimentally motivated workaround, not a claim of a proven vendor fix.

Use512threads and complete warpgroups:8 donor warps decrease to32 registers;
8 compute warps increase to192. Only warps8..11 produce KV and warp12 issues
MMA; warps13..15 exit after the common redistribution. Retain v125 direct256-bit
output. The role budget is57344 registers; initial allocation must cover it
before launch. Offline compilation with CUDA context queries precedes resource
and USETMAXREG inspection. This tests fewer TMEM load waits without x64 IR or
spill traffic; arithmetic and tile geometry remain unchanged.

### Iteration 133 expanded result — validated alternative, keep current default

Full8192 seeds1234/5678 and full1024 short/chunk0 seed5678 matchv128 bitwise in
three repeats, in addition to masked equality and qualified sanitizers. Eager
20/100 warm/cold1947.57/1939.18 us versusTRT1878.99/1918.99 us; Graph1966.26/
1936.42 us versusTRT1867.84/1917.06 us. Compared with the separately recordedv128
runs, eager warm is slightly worse, while cold and Graph improve slightly. The
three rotated orders favorv133, but the overall change remains small and regime
dependent; retainv128 as the established higher-precision default and exposev133
as a fully validated cache-policy alternative. No claim of surpassingTRT.

### Iteration 136 result — safe allocation control is much slower

Offline REG114/STACK0. Qualified b2 and b512 memcheck with explicit Tensor Core
guardrails report zero errors; b2 synccheck and the original eight-row full-target
check pass. Masked FP32 comparison has66 mismatches/max_abs0.02237046 versus
TRT78 mismatches, with59700 unequal outputs; this is not a full-tolerance pass or
bitwise agreement. No full-row/second-seed audit is claimed.

Uninstrumented short warm2960.13 us versusTRT1691.74 us rejects this schedule.
NCU base/stable:114 registers,115296 dynamic shared bytes, occupancy21.356%,
tensor17.274%, eligible0.242501, long-scoreboard4.365522, zero local sectors,
aggregate shared conflicts12996/723591, diagnostic5.106496 ms. Increased resident
warps can include a CTA waiting for its512-column TMEM reservation and do not
establish simultaneous useful MMA work. Reducing shared storage alone did not
repay the smaller tiles, extra output preservation and lost QK/PV overlap.

### Iteration 137 compile failure — paired x32 alone does not resolve the bug

Compilation with CUDA attribute queries again fails in the NVVM backend with
no additional diagnostic. No CUBIN, register-budget verification or GPU launch
exists. Replacing the x64 copy atom with two x32 atoms was insufficient in this
kernel; do not present the external issue's x32 bisection result as a general
workaround. Preserve the exact source SHA and compiler log.

## Iteration 138 — paired x32 loads without redistribution

Remove v137's dynamic register increase/decrease and restore416threads without
an explicit minimum-block launch bound. Keep the paired x32 loads/stores and
one wait per pair. This control separates paired-load behavior from register
redistribution and checks whether default allocation spills as v121 did.

## Iteration 139 — inline-PTX register directive control

Keep v137's512threads, complete warpgroup roles and32/192 register targets,
but emit side-effecting inline PTX setmaxnreg directives instead of the CuTe/NVVM
setmaxregister operations. This probes the failing compiler-lowering path, not
an assumed performance gain. Compilation must succeed and SASS must retain
both USETMAXREG directives, with a sufficient initial CTA register budget,
before any bounded GPU run. The same57344-register role budget applies.

### Iteration 139 compile failure — inline PTX is not a workaround here

Replacing NVVM setmaxregister operations with inline PTX still triggers the same
generic NVVM-backend failure. No CUBIN or device launch exists. The installed
CuTe functions were verified to use nvvm.setmaxregister, so this was a distinct
lowering-path control, not an identical wrapper. Stop register-directive retries
on this variant; preserving the failure is more useful than repeating them.

### Iteration 138 compile result — paired x32 avoids the old x64 spills

Without redistribution, compilation succeeds at REG123/STACK0. This differs
fromv121's native x64 correction at REG128/STACK64. Runtime local-memory traffic,
all-row equivalence and latency remain to be measured; zero static stack alone
is not yet a performance result or a complete spill audit.

### Iteration 138 initial runtime result — no spills and a small tuning gain

Full8192 seed1234 output bits matchv125 in three repeats. The qualified b2
synccheck/b512 memcheck commands complete successfully. Short warm1639.46 us
versusTRT1691.49 us improves about9.40 us from recordedv125. NCU base/stable:
123 registers,194920 shared bytes, occupancy19.294%, tensor31.571%, eligible
0.398541, long-scoreboard6.192859, zero local sectors, aggregate shared conflicts
6219608/1958982, diagnostic2.800736 ms. Expanded bitwise/latency validation and
unlocked source sampling are in progress. No promotion yet.

## Iteration 140 — paired x32 correction on higher precision

Based on v128, transplant v138's paired x32 loads followed by one load wait,
64-value packed correction and paired x32 stores. Keep residual FP8 probability
arithmetic, collector reuse, strict synchronization and output cache policy.
No launch bounds or dynamic register directives. This tests whether the fast
path's absence of x64-induced spills and lower wait count transfers to the
higher-precision register lifetimes; inspect resources and measure independently.

## Iteration 141 — combine paired correction with output evict-first

Based on v138, change only the output L2 eviction priority to evict-first, as in
v129. The cache policy's modest improvement may or may not survive the paired
correction schedule. Preserve all numerical operations and assess same-process
ordering alongside v138 after its full validation is available.

### Iteration 138 expanded result — promote paired x32 correction

All8192 output rows matchv125 bitwise for seeds1234/5678, three repeats each;
all1024 short/chunk0 seed5678 rows and masked inputs also match. Qualified
synccheck/memcheck report zero errors. These are equivalence results, preserving
all documented FP8 tolerance failures rather than eliminating them.

Eager20/100 warm/cold1708.69/1715.17 us versusTRT1880.22/1914.93 us. Graph20/100
1724.70/1713.23 us versusTRT1869.02/1920.93 us. Four same-process rotated orders:

| Cache | v125 us, rounds0–3 | v129 us, rounds0–3 | v138 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1688.54 / 1689.47 / 1689.74 / 1690.40 | 1677.54 / 1678.88 / 1688.72 / 1689.39 | 1681.52 / 1681.57 / 1682.54 / 1681.74 | 1872.50 / 1877.97 / 1878.11 / 1882.02 |
| cold | 1691.42 / 1685.52 / 1693.68 / 1687.68 | 1681.36 / 1683.31 / 1688.61 / 1689.50 | 1677.36 / 1684.62 / 1675.25 / 1679.09 | 1855.97 / 1857.63 / 1857.70 / 1869.63 |

v138 beats establishedv125 in all4 warm and all4 cold orders; v129 is mixed
againstv138. Promotev138 as the fast path, retainv128 as higher precision.
Unlocked source profiling records602477910 instructions versus605733244 inv125
(-0.54%),20275200 actual/ideal shared wavefronts and zero excessive. Long-scoreboard
samples65996 includePV17413,producer empty17172,QK13624. SASS confirms adjacent
LDTM.x32 pairs before correction FMUL2; waits are encoded as scheduling dependencies,
so do not count NOP instructions as explicit wait instructions.

### Softmax packed-FMA inspection — no redundant experiment

Before creating a packed-FMA variant, inspectedv138 SASS. Its existing vector
expression already lowers to FFMA2 pairs interleaved with32 MUFU.EX2 instructions.
Explicitly calling fma_packed_f32x2 cannot claim a32-to16 instruction reduction
that the compiler already performs. Do not create that proposed variant merely
to restate the same operation. API signatures were checked in installed CuTeDSL4.6.2;
[PTX fma semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma)
and [CuTe arch API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html)
are the primary references. Seek a distinct instruction or dependency change.

## Iteration 142 — pair final output loads

Based onv138, also pair two final-output LDTM.x32 operations before one load
wait, reusing the existing64-float register fragment. Normalize/convert/store
one32-value half at a time to avoid keeping all64 BF16 results live. Preserve
physical/logical output mapping,32-byte alignment, STG256 policy and arithmetic.
This targets four-to-two output load waits per compute thread; the epilogue runs
only once per16 key tiles, so expected benefit is modest. Compile and inspect
spills before bounded smoke and exact-equivalence validation.

### Iteration 140 result — higher-precision pairing regresses

REG123/STACK0; all8192 seed1234 outputs matchv128 bitwise in three repeats and
masked outputs match with zero tolerance failures. Qualified b2 synccheck/b512
memcheck report zero errors. Short1822.75 us versusTRT1690.88 us is slower than
v128's1802.37 us; do not promote or extrapolatev138's fast-path gain.
NCU base/stable:123 registers,203080 shared bytes, occupancy19.343%, tensor41.792%,
eligible0.454520,long-scoreboard6.163935, zero local sectors, aggregate shared
conflicts6284051/1403235,diagnostic3.114080 ms. No second-seed/short full audit.

### Iteration 141 initial result — small combined cache-policy gain

REG123/STACK0; all8192 seed1234 outputs matchv138 bitwise in three repeats,
masked output bits match, qualified b2 synccheck/b512 memcheck report zero errors.
Short1636.54 us versusTRT1691.94 us is about2.91 us belowv138. NCU base/stable:
123 registers,194920 shared bytes, occupancy19.298%, tensor31.570%,eligible0.398663,
long-scoreboard6.196022,zero local sectors,aggregate shared conflicts6212999/
1914229,diagnostic2.800256 ms. This small tuning change needs expanded validation
and rotation before deciding whether to replace the normal-L2-policy default.

### Iteration 141 expanded result — retain as a cache-policy alternative

Both full seeds and the short case match v138 bitwise in three repeats each;
masked equality and qualified sanitizers also pass. Eager 20/100 warm/cold:
1706.26/1721.17 us versus TRT 1884.32/1917.01 us; Graph 1725.07/1689.62 us versus
TRT 1871.66/1925.84 us. Three rotated warm orders favor v141 (1679.49/1679.46/
1679.46 us) over v138 (1681.54/1684.13/1681.81 us). Cold orders are mixed:
v141 1679.54/1681.12/1672.27 us versus v138 1675.38/1683.70/1684.06 us.
Keep v138 as default and v141 as a fully validated cache-policy alternative;
the 2–5 us warm advantage is small and cold/eager effects vary.

### Iteration 142 result — paired epilogue loads do not improve latency

REG123/STACK0. Full seed1234 matches v138 bitwise in three repeats; masks match,
qualified b2 synccheck and b512 memcheck report zero errors. Short 1644.74 us
versus TRT 1690.78 us is about 5.28 us above v138. NCU base/stable: 123 registers,
194920 shared bytes, occupancy 19.283%, tensor 31.492%, eligible 0.396840,
long-scoreboard 6.166939, zero local sectors, aggregate shared conflicts
6214389/1963056, diagnostic 2.810336 ms. Do not promote; no expanded audit.

## Iterations 143–144 — consolidate producer barriers in the current pipeline

An older v063 control removed producer barriers before pre-wait index fetch,
four-producer specialization, the dedicated MMA issuer, and paired TMEM correction
were introduced. Revisit that dependency change on v138 with two bounded controls.
v143 removes only the barrier between writing indices/masks and recording expected
TMA bytes. The following 128-thread barrier already publishes both and prevents
any gather from preceding the transaction expectation. Retain the post-gather
barrier. v144 additionally removes the post-gather barrier: the next tile uses
separate stage storage, and each producer waits for its stage's empty notification
before reuse; the retained publication barrier still joins all producers before
each tile's gather. No shared layout or numerical operations change. Validate
synchronization and bitwise outputs before interpreting performance.

### External exp2 emulation inspection

The installed CuTe exp_packed_f32x2 is deprecated and lowers to packed multiply
plus two ordinary exp2 calls; it is not a packed SFU instruction. FlashAttention
[issue 2358](https://github.com/Dao-AILab/flash-attention/issues/2358) documents an
SM103 exclusion for polynomial emulation, without a maintainer explanation on the
retrieved page. The inspected FlashInfer checkout also enables its FMHA emulation
only for capability (10, 0). These are implementation choices, not proof of a
hardware limitation or universal performance result. No polynomial approximation
is introduced here, and no bitwise claim is inferred from another kernel's tests.

## Iteration 145 — validity bitmaps on the current higher-precision path

Source inspection shows v128 still loads 32 individual validity entries per
compute thread; the earlier v092 bitmap experiment was not promoted. Port only
the bitmap publication and lookup from the fast path to v128. Each producer warp
publishes its 32 validity predicates once; each compute thread reads its matching
32-bit segment. Keep scaled-score multiplication inside the validity predicate,
the bounded anchor, residual FP8 arithmetic, balanced sums, strict barriers and
single x32 correction unchanged. This revisits bitmap cost after the dedicated
issuer and newer register allocation; the older v092 result does not establish
its effect in this schedule. Require bitwise v128 equivalence and mask accuracy.

## Iteration 146 — acquire the compute bitmap earlier

Based on v138, move the single shared bitmap load to immediately after acquiring
the stage-full barrier and before waiting for QK completion. Retain both waits and
all fences. The producer has published the bitmap by full acquisition; QK does
not write it. This tests whether its shared-load dependency can overlap the QK
wait. Verify SASS placement: source motion alone does not establish an executed
instruction change. Arithmetic, masks, and the pipeline protocol are unchanged.

## Iteration 147 — prefetch correction registers during probability math

Based on v138, issue the first pair of x32 output-correction loads just after
computing the correction factor, before probability exponentials, reduction and
FP8 conversion. Keep the original warp-uniform identity test and block>0 guard;
wait before consuming the prefetched registers at the original correction point.
The second pair remains demand-loaded. Previous PV is already complete before
this iteration starts. This trades a longer 64-register live range for potential
load/math overlap; compile resources and actual spills may reject it. No numerical
operation is changed, and a successful compile is not a runtime result.

### Iterations 143–144 result — small synchronization effects, retain default

Both compile with REG123/STACK0. Full seeds1234/5678 and short/chunk0 seed5678
match v138 bitwise in three repeats each; masks match, qualified b2 synccheck and
b512 memcheck report zero errors. Short medians: v143 1636.42 us versus TRT
1691.78 us; v144 1636.48 us versus TRT 1689.89 us. Four rotated Graph orders:

| Cache | v138 us, rounds0–3 | v143 us, rounds0–3 | v144 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1682.51 / 1681.68 / 1681.79 / 1681.57 | 1679.49 / 1683.46 / 1679.74 / 1679.55 | 1679.46 / 1679.42 / 1681.65 / 1679.34 | 1868.32 / 1879.98 / 1877.84 / 1876.13 |
| cold | 1677.20 / 1681.44 / 1682.32 / 1677.31 | 1681.55 / 1675.49 / 1672.99 / 1675.09 | 1673.12 / 1675.01 / 1682.56 / 1673.18 | 1859.49 / 1859.81 / 1864.32 / 1859.70 |

v143 has mixed warm/cold wins. v144 wins all four warm orders by 0.14–3.05 us,
but one cold order is 0.24 us slower. These small effects do not warrant another
default change yet; retain both as validated scheduling alternatives.

NCU base/stable (v143 / v144): 194920 shared bytes; occupancy 19.281% / 19.303%;
tensor 31.603% / 31.654%; eligible 0.399988 / 0.401802; long-scoreboard 6.185871 /
6.229347; zero local sectors in both; shared-conflict aggregates 6836608/1896738
and 6618725/1928820; diagnostic durations 2.797216 / 2.796064 ms. These aggregate
counts are not evidence of a changed shared-memory layout or excessive conflicts.

### Iteration 145 initial result — bitmap benefit returns in the newer schedule

REG98/STACK0, versus v128 REG122. Full seed1234 matches v128 bitwise in three
repeats; masked original-tolerance checks pass and masked output bits match.
Qualified b2 synccheck and b512 memcheck report zero errors. Short 1781.79 us
versus TRT 1691.90 us improves about 20.58 us from v128. NCU base/stable:
98 registers, 203112 shared bytes, occupancy 19.337%, tensor 42.787%, eligible
0.449606, long-scoreboard 6.364198, zero local sectors, aggregate shared conflicts
6443889/2243835, diagnostic 3.040896 ms. Expanded validation is pending; the
register reduction does not raise CTA residency with this shared/TMEM footprint.

## Iteration 148 — consume residual PV collectors immediately

Based on v145, change each PV output tile from hi(K0..K3), lo(K0..K3) to
hi(K0), lo(K0), hi(K1), lo(K1), and so on. Each high operation fills its original
B collector and the following residual operation last-uses it; both still share
exactly the same V tile. This shortens collector lifetimes while retaining the
same number of FP8 MMA operations and shared loads. It may instead serialize
issue or lose useful scheduling freedom, so performance is empirical.

The probability values and mathematical expression are unchanged, but FP32
accumulation order changes. Do not inherit bitwise equivalence or full-tolerance
results from v145. Begin with qualified sanitizers, masked FP32 checks and the
original sampled benchmark; if the timing merits further work, run independent
full-reference audits before promotion. No change to the benchmark or tolerance.

### Iteration 145 expanded result — promote the higher-precision bitmap path

Both full seeds and the full short case match v128 bitwise in three repeats;
masked FP32 checks pass and masks match v128 bits. Thus the documented original
full-tolerance passes are retained on these audited inputs. Eager 20/100 warm/cold
1921.14/1922.64 us versus TRT 1880.14/1914.94 us; Graph 1955.74/1922.10 us versus
TRT 1867.89/1915.04 us. Four same-process rotated Graph orders:

| Cache | v128 us, rounds0–3 | v133 us, rounds0–3 | v145 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1923.07 / 1925.25 / 1923.30 / 1923.12 | 1925.15 / 1925.78 / 1924.94 / 1923.06 | 1898.85 / 1899.25 / 1900.48 / 1900.75 | 1813.65 / 1879.89 / 1877.92 / 1876.10 |
| cold | 1918.86 / 1916.77 / 1918.99 / 1916.94 | 1912.77 / 1915.12 / 1914.67 / 1914.93 | 1896.50 / 1896.43 / 1896.66 / 1906.70 | 1865.63 / 1867.76 / 1863.73 / 1863.76 |

v145 beats both higher-precision predecessors in every recorded order. Promote
v145, while explicitly retaining TRT's lead. The unusual first TRT warm median
is retained as measured, not dropped. Unlocked source counters: 673569020 dynamic
instructions, 28663808 actual/ideal shared wavefronts, zero excessive. Long-scoreboard
samples73515 include PV26090, producer-empty17173 and QK14184. No same-mode
v128 source profile has yet been collected, so these totals alone do not quantify
the bitmap's instruction reduction relative to v128.

## Iteration 149 — early bitmap load on the higher-precision path

Move v145's bitmap read after full-stage acquisition and before QK completion
wait, matching v146's dependency experiment. Keep scaled-score math, residual P,
strict barriers and all validity semantics unchanged. The first fast-path tuning
result motivates this separate test, but does not establish a high-precision gain.

### Iteration 146 initial result — early bitmap load improves short timing

REG123/STACK0. Full seed1234 and masks match v138 bitwise; qualified b2 synccheck
and b512 memcheck report zero errors. Short 1628.10 us versus TRT 1693.86 us is
about 11.36 us below v138. NCU base/stable: 123 registers, 194920 shared bytes,
occupancy 19.287%, tensor 31.782%, eligible 0.389984, long-scoreboard 6.146301,
zero local sectors, aggregate shared conflicts 6319524/2098529, diagnostic
2.781248 ms. Expanded validation, rotation and source placement inspection are
pending; do not attribute the whole difference to shared-load latency yet.

### Iteration 147 result — correction prefetch hurts without spilling

REG126/STACK0. Full seed1234 output and mask bits match v138; qualified b2
synccheck and b512 memcheck report zero errors. Short 1690.85 us versus TRT
1693.54 us is 51.39 us slower than v138. NCU base/stable: 126 registers,
194920 shared bytes, occupancy 19.327%, tensor 30.640%, eligible 0.414074,
long-scoreboard 5.930648, zero local sectors, aggregate shared conflicts
6334422/1762472, diagnostic 2.896416 ms. Reduced aggregate long-scoreboard ratio
is not a speedup: the changed register lifetimes and schedule regress overall.
No expanded audit or promotion for this prefetch placement.

### Iteration 146 expanded result — promote earlier bitmap acquisition

Full seeds1234/5678 and full short/chunk0 seed5678 match v138 bitwise in three
repeats each; masks and qualified sanitizers pass equivalence/safety checks.
All previous fast-path FP8 tolerance failures remain. Eager warm/cold1714.32/
1714.99 us versus TRT1881.14/1914.91 us; Graph1714.21/1709.50 us versus
TRT1867.89/1919.25 us. The separate eager warm median is slightly worse than v138,
so the promotion rests on the full paired evidence, not an all-regime claim.
Four same-process rotated Graph orders:

| Cache | v138 us, rounds0–3 | v144 us, rounds0–3 | v146 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1682.62 / 1681.57 / 1681.58 / 1682.29 | 1677.70 / 1677.50 / 1679.30 / 1679.50 | 1673.17 / 1672.37 / 1673.34 / 1671.20 | 1872.05 / 1879.46 / 1878.13 / 1878.19 |
| cold | 1683.36 / 1684.61 / 1683.41 / 1679.47 | 1673.55 / 1681.62 / 1672.82 / 1673.15 | 1675.41 / 1666.53 / 1675.12 / 1667.42 | 1859.44 / 1861.70 / 1861.70 / 1859.58 |

v146 beats the established v138 default in all warm/cold orders and v144 in all
warm orders; cold effects versus v144 are mixed. Promote v146 as the fast path.
Unlocked source:600988722 instructions versus v138602477910; shared wavefronts
remain20275200 actual/ideal, zero excessive. Long-scoreboard samples65833 include
producer-empty17818, PV16908 and QK13664. SASS places the bitmap LDS immediately
before the QK phase check; v138 loads it after the score LDTM. Source motion is
therefore reflected in executed code, while the small sample differences do not
uniquely explain every microsecond of the timing gain.

## Iteration 150 — combine early bitmap acquisition with consolidated producers

Based on v146, transplant v144's one-barrier producer protocol. Preserve the
full-stage acquisition, early bitmap read, QK acquisition and all arithmetic.
The prior changes were validated independently; their performance interaction
requires a new paired run and exact-equivalence checks.

## Iteration 151 — remove a redundant TMEM code-motion fence

Based on v146, remove only the compute-side after-thread-sync TMEM fence between
full acquisition and bitmap LDS. Retain full acquisition and the fence after QK
acquisition, which still precedes every subsequent tcgen05 operation. The intervening
bitmap LDS is an ordinary shared load. This is an inference from the documented
[tcgen05 fence semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-special-sync-operations):
the remaining after fence orders subsequent tcgen05 operations after both prior
waits. First inspect generated code; if the instruction sequence is identical,
do not invent a performance gain or run redundant benchmark trials.

### v128 versus v145 unlocked source comparison

A subsequent v128 profile uses the same unlocked SourceCounters/WarpStateStats
recipe as v145. Dynamic instructions fall from713767944 to673569020 (-5.63%);
actual/ideal shared wavefronts fall from43868160 to28663808 (-34.66%), with zero
excessive wavefronts in both. This supports reduced validity-load work, rather
than removal of a bank-conflict layout problem. v128 long-scoreboard samples75429
includePV26556, producer-empty18065 andQK14202, versus v14573515/26090/17173/14184.
Sample counts fluctuate and are not direct elapsed-time savings.

### Iteration 148 initial result — faster residual PV order, new rounding

REG98/STACK0. Seed1234 has7817 BF16 outputs different from v145, maximum absolute
output difference0.001953125, identically across three repeats. This is expected
from the changed accumulation order and invalidates inheritance of bitwise results.
Masks have one different BF16 output but both versions pass the original FP32
tolerance with max_abs0.00400185585. Qualified b2 synccheck/b512 memcheck report
zero errors. Short1749.12 us versus TRT1691.84 us improves32.67 us from v145.
NCU base/stable:98 registers,203112 shared bytes,occupancy19.341%,tensor43.647%,
eligible0.461838,long-scoreboard6.168047,zero local sectors,aggregate shared
conflicts6297198/2319846,diagnostic2.978976 ms. Independent full-reference
checks on two target seeds and the short case are running before promotion.

### Iteration 149 initial result — early bitmap also helps higher precision

REG100/STACK0. Full seed1234 and masks match v145 bitwise; masked FP32 tolerance
passes. Qualified b2 synccheck/b512 memcheck report zero errors. Short1771.74 us
versus TRT1691.42 us improves10.05 us from v145. NCU base/stable:100 registers,
203112 shared bytes,occupancy19.327%,tensor42.957%,eligible0.462870,long-scoreboard
6.258764,zero local sectors,aggregate shared conflicts6484087/2356232,diagnostic
3.023744 ms. Expanded equivalence and rotated timing are pending.

### Iteration 151 compile inspection — generated code changes

Both v150 and v151 compile at REG123/STACK0. The v146/v151 disassembly contains
2880 encoding words each, but the words differ. Inspection shows a NOP removed
between full acquisition and bitmap address calculation, plus changed control
bits and subsequent instruction/branch offsets. Equal padded code size does not
mean identical executed code, so bounded numerical/synchronization and timing
checks are warranted. This is not yet a measured speedup.

## Iteration 152 — combine immediate residual collector use and early bitmap

Based on v148, move only the bitmap read to before QK acquisition, as tested in
v149. Retain v148's interleaved high/residual accumulation order. The appropriate
bitwise baseline is v148, not v145; v148's independent FP32 checks are separate.
This tests whether the two observed high-precision improvements compose without
register spills or a scheduling regression.

### Iteration 148 expanded result — independently validated higher-precision improvement

Independent FP32-reference audits pass the unchanged atol0.01/rtol0.05 tolerance:
all268435456 elements on each target seed1234/5678, and all33554432 short-case
seed5678/chunk0 elements. Maximum absolute errors are0.005918741226,
0.005017399788 and0.008034229279; relative RMSE0.001735969640,
0.001735983959 and0.001678571923. Masks also pass; no bitwise equivalence to
v145 is claimed. Qualified sanitizers already pass.

Eager20/100 warm/cold1883.06/1892.50 us versus TRT1880.13/1902.13 us; Graph
1924.24/1894.53 us versus TRT1867.82/1894.32 us. Four rotated Graph orders:

| Cache | v145 us, rounds0–3 | v148 us, rounds0–3 | v149 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1898.66 / 1898.54 / 1898.77 / 1898.59 | 1876.62 / 1877.95 / 1880.00 / 1878.06 | 1888.26 / 1889.76 / 1888.42 / 1900.70 | 1863.78 / 1872.24 / 1871.89 / 1872.02 |
| cold | 1892.38 / 1891.30 / 1892.14 / 1892.46 | 1874.00 / 1869.84 / 1876.03 / 1871.92 | 1882.27 / 1882.27 / 1884.19 / 1884.16 | 1855.68 / 1857.06 / 1855.84 / 1855.55 |

v148 beats v145 and v149 in every recorded order. Promote v148 as the higher-
precision default; TRT remains ahead in these rotated tests, by roughly0.3–0.7%
warm and0.7–1.1% cold. The eager cold win does not establish a universal lead.
Unlocked source:679841906 instructions (more than v145673569020), shared wavefronts
28663808 actual/ideal, zero excessive; long-scoreboard71642, includingPV24945,
producer-empty16415 andQK14190. Faster latency despite more total instructions is
consistent with a scheduling gain; counters alone do not prove the exact internal
collector mechanism.

### Iteration 149 expanded result — validated, superseded by the PV-order change

Full seed5678 and short/chunk0 seed5678 match v145 bitwise in three repeats,
in addition to seed1234, masks and qualified sanitizers. Eager warm/cold1914.56/
1912.61 us versus TRT1869.23/1916.59 us; Graph1943.74/1914.74 us versus
TRT1865.90/1899.58 us. Rotation improves on v145 in three of four warm orders
and all four cold orders, but v148 is faster throughout. Retain as an independently
validated input to the v152 combination, without making it the default.

## Iterations 153–154 — isolate high-precision synchronization after the PV change

Based on v152's interleaved residual PV and early bitmap pipeline, revisit the
independent synchronization controls previously tested as v110/v111 on v106.
The older results were not consistently beneficial; changed producer/compute
work and collector ordering motivate two new bounded controls, not an assumed gain.

v153 removes only the trailing 256-thread barrier after PV completion. The
retained pre-P-ready barrier joins all compute reads of indices/scores/P; the PV
completion then drains asynchronous KV reads before thread0 releases the stage.
All threads still wait for PV and keep the TMEM fences. v154 instead retains
that trailing barrier, replaces pre-P-ready CTA synchronization plus one arrival
with256 per-thread arrivals, and retains per-thread TMEM/shared fences. The
issuer's acquire waits for every compute thread's P-high/P-low/correction writes.

Each variant preserves arithmetic and uses v152 as the exact-equivalence baseline.
Qualification requires bounded smoke, synccheck, memcheck, full-seed output bits
and masked FP32 checks before paired performance interpretation.


### Iteration 150 result — combined producer changes have mixed sustained effects

REG123/STACK0. Both full seeds1234/5678 and short/chunk0 seed5678 match v146
bitwise in three repeats each; mask bits match, and qualified b2 synccheck/b512
memcheck report zero errors. All fast-path FP8 tolerance limitations remain.
Short1624.38 us versus TRT1691.87 us is3.72 us below v146. Eager20/100 warm/cold
1694.83/1719.23 us versus TRT1866.38/1921.20 us; Graph1724.46/1700.02 us versus
TRT1867.81/1914.94 us. Three rotated Graph orders:

| Cache | v146 us, rounds0–2 | v150 us, rounds0–2 | TRT us, rounds0–2 |
|---|---|---|---|
| warm | 1693.89 / 1671.42 / 1673.42 | 1670.80 / 1669.10 / 1690.75 | 1878.11 / 1880.11 / 1882.02 |
| cold | 1665.33 / 1675.26 / 1665.14 | 1673.22 / 1662.98 / 1673.30 | 1867.92 / 1859.65 / 1859.49 |

Warm wins two of three orders; cold wins one. Retain all order outliers and
keep v146 as default; the combination is a validated alternative, not a stable
improvement. NCU base/stable:123 registers,194920 shared bytes,occupancy19.284%,
tensor31.837%,eligible0.394969,long-scoreboard6.189487,zero local sectors,
aggregate shared conflicts6687675/2065833,diagnostic2.777184 ms.

### Iteration 151 result — removing the extra fence has no short-run gain

Full seed1234 output matches v146 in three repeats; masks match and qualified
b2 synccheck/b512 memcheck report zero errors. Short1628.03 us versus TRT1691.78 us
is effectively unchanged from v1461628.10 us. No expanded audit or promotion.
NCU base/stable:123 registers,194920 shared bytes,occupancy19.313%,tensor31.922%,
eligible0.389618,long-scoreboard6.160951,zero local sectors,aggregate shared
conflicts6361361/2099771,diagnostic2.781344 ms. A removed code-motion NOP and
changed scheduling encodings do not establish a measurable latency improvement.

### Iteration 152 result — validated warm improvement, mixed cold result

REG100/STACK0. Full seeds1234/5678 and short/chunk0 seed5678 match v148 bitwise
in three repeats each. Mask bits match and masked FP32 tolerance passes; qualified
b2 synccheck/b512 memcheck report zero errors. The full-tolerance conclusions on
these inputs therefore carry over from the independent v148 FP32 audits.
Short1739.04 us versus TRT1691.84 us improves10.08 us from v148. Eager20/100
warm/cold1896.66/1884.37 us versus TRT1888.08/1916.90 us; Graph1927.20/1900.50 us
versus TRT1876.02/1917.01 us. Three rotated Graph orders:

| Cache | v148 us, rounds0–2 | v152 us, rounds0–2 | TRT us, rounds0–2 |
|---|---|---|---|
| warm | 1880.06 / 1882.24 / 1882.14 | 1872.91 / 1873.79 / 1873.87 | 1867.92 / 1877.86 / 1880.69 |
| cold | 1871.73 / 1870.64 / 1871.87 | 1859.79 / 1872.13 / 1869.86 | 1863.66 / 1857.71 / 1859.49 |

v152 wins all warm orders by7.15–8.45 us and two of three cold orders; cold round1
is1.49 us slower. Separate eager/Graph warm medians do not improve on v148.
Retain v152 as a validated scheduling candidate and the basis for synchronization
controls; keep the established v148 default pending further implementation evidence.
TRT comparisons also vary with cache/order, so no universal high-precision lead.
NCU base/stable:100 registers,203112 shared bytes,occupancy19.341%,tensor43.850%,
eligible0.477152,long-scoreboard6.080971,zero local sectors,aggregate shared
conflicts6368821/2289596,diagnostic2.970944 ms.

### Iterations 153–154 compile qualification

Both compile offline without a CUDA context at REG100/STACK0. Their source hashes
are4d6ce7a0d9201fc52b99342a6b3083374545c8c618404bec980a95d4a9049a93 and
27c33221ea1eabdc7edd3e199fadfb85786eb1c96a644701a0787a8635cbc819.
No register redistribution or launch-footprint change is involved. Runtime
qualification is separate from this compile result.


### Iteration 153 initial result — a small stage-release improvement

Full seed1234 matches v152 in three repeats, masked FP32 checks pass and mask
bits match. Qualified b2 synccheck/b512 memcheck report zero errors. Short1734.72 us
versus TRT1691.90 us is4.32 us below v152. NCU base/stable:100 registers,203112
shared bytes,occupancy19.333%,tensor44.001%,eligible0.474625,long-scoreboard6.137062,
zero local sectors,aggregate shared conflicts6425253/2135511,diagnostic2.955936 ms.
Expanded equivalence and same-process rotated comparisons are pending.

### Iteration 154 result — per-thread P release alone does not improve timing

Full seed1234 and mask bits match v152; masked FP32 checks and qualified
synccheck/memcheck pass. Short1743.71 us versus TRT1691.90 us is4.67 us slower
than v152. NCU base/stable:100 registers,203112 shared bytes,occupancy19.343%,
tensor43.822%,eligible0.470472,long-scoreboard6.163579,zero local sectors,
aggregate shared conflicts6347528/2311928,diagnostic2.969344 ms.
No expanded audit or promotion for this standalone synchronization change.


## Iterations 155–156 — alternate independent PV output accumulators

Based on v152, isolate issuer scheduling from the v153/v154 synchronization
controls. v152 completes all four high/low K pairs for output tile0 before tile1.
v155 makes K outermost and alternates output tiles after each adjacent high/low
pair. v156 issues high0,high1,low0,low1 for each K, retaining two distinct live
B collectors and alternating collector pairs between K slices. Every fill is
last-used with the same V descriptor before that collector is reused.

Both schedules retain the exact high(K0),low(K0),high(K1),low(K1),... arithmetic
order for each output element. They keep all16 PV MMAs and the existing commit,
wait, P conversion, stage-release and mask protocols. The hypothesis is that
alternating independent accumulator regions may shorten back-to-back dependencies;
this is not a claim about undocumented tensor-core microarchitecture. Bounded
compile/safety/equivalence qualification precedes timing; v152 is the bitwise
baseline. Resource or performance regressions will be recorded without promotion.


## Iteration 157 — pack the pre-softmax score scaling

The v148 unlocked SASS shows32 scalar FMUL score-scaling instructions per compute
thread/tile followed by validity SEL instructions, whereas subsequent exponential
shifts already use FADD2. This differs from the previously rejected explicit-FMA
proposal, where the fast path was already packed. Based on v152, explicitly use
16 mul_packed_f32x2 operations, then retain the exact per-element bitmap selection
and -1e30 masked sentinel. Valid elements keep the same FP32 scaling operation;
invalid elements are overwritten before maximum/exponential/reduction operations.

[PTX mul documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul)
supports packed FP32 multiplication on sm100 and newer; this does not guarantee
fewer executed instructions after register allocation. Inspect SASS and resource
usage, then check full output bits/masks and qualified sanitizers. No probability,
anchor, denominator or PV order change is intended. Baseline:v152.


### Iteration 153 expanded result — promote the stage-release control

Both full seeds and the full short/chunk0 case match v152 bitwise in three
repeats each, preserving v148's independent FP32-reference passes on these
inputs. Masks and qualified sanitizers pass. Eager20/100 warm/cold1873.97/1900.54 us
versus TRT1880.13/1914.83 us; Graph1925.15/1898.61 us versus TRT1873.78/1917.02 us.
The separate Graph warm median is effectively unchanged from v1481924.24 us;
this is not an all-regime improvement. Four rotated Graph orders:

| Cache | v148 us, rounds0–3 | v152 us, rounds0–3 | v153 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1878.69 / 1881.33 / 1880.77 / 1882.10 | 1871.97 / 1871.86 / 1873.01 / 1876.05 | 1869.70 / 1869.82 / 1870.02 / 1871.78 | 1869.87 / 1876.06 / 1880.30 / 1879.52 |
| cold | 1869.70 / 1872.03 / 1869.90 / 1873.94 | 1861.06 / 1859.98 / 1871.73 / 1873.84 | 1865.89 / 1867.52 / 1867.68 / 1866.74 | 1866.11 / 1867.62 / 1862.70 / 1865.95 |

v153 beats the established v148 default in all eight recorded orders, warm by
8.99–11.51 us and cold by2.22–7.20 us. It also beats v152 in every warm order,
while cold results versus v152 are mixed. Promote v153 as higher-precision default.
The rotated warm TRT advantage ranges from near zero to0.55%; cold has two wins
and two losses, and the separate Graph warm comparison still favors TRT. Do not
claim a universal high-precision lead or merge these distinct timing regimes.


## Iterations 158–159 — streaming Q cache priority

At the fixed target, Q occupies288 MiB, KV72 MiB and output512 MiB. Each query
CTA loads its Q tile once and reuses it from shared memory, while different CTAs
reuse gathered KV from the common cache. Based on v146 (v158) and v153 (v159),
attach a fractional L2 evict-first policy with fraction1.0 only to the five Q TMA
loads. KV policy, output policy, tensor-map shapes, shared layout, arithmetic and
all synchronization stay unchanged. This complements the prior output/KV policy
controls; query TMA priority itself has not previously been changed.

The documented cp.async.bulk.tensor L2 cache hint is advisory, not cache bypass
or a correctness dependency. A smaller DRAM count would not by itself prove a
latency improvement. Compile/resource inspection, exact output bits, masks and
qualified sanitizers precede short and, if warranted, extended paired timing.
Use v146 and v153 respectively as equivalence baselines; retain the fast-path
precision limits and the distinction between warm/cold execution regimes.


### Iterations 155–156 result — alternating output tiles does not help

Both REG100/STACK0. Each matches v152 on full seed1234 in three repeats and on
mask bits; masked FP32 checks and qualified b2 synccheck/b512 memcheck pass.
Short v1551804.42 us versus TRT1691.90 us; v1561753.18 us versus TRT1689.89 us,
respectively65.38/14.14 us slower than v152. No expanded audit or promotion.
v155 also reuses a given B collector sooner between the two output tiles, so
this experiment does not isolate accumulator-dependency cost from collector reuse.
v156 separates the two live collectors but still fails to improve overall timing.

NCU base/stable(v155/v156):203112 shared bytes,occupancy19.357%/19.344%,tensor
42.189%/43.619%,eligible0.455346/0.474114,long-scoreboard6.417673/6.145557,
zero local sectors,aggregate shared conflicts6563620/2385966 and6596272/2321443,
diagnostic3.080480/2.992128 ms. Same register count and zero spills do not make
a changed issue order beneficial.

## Iteration 160 — combine packed score scaling and stage release

Based on v153, transplant only v157's16 packed score multiplies before unchanged
bitmap masking. Retain v153's stage-release protocol and all remaining arithmetic.
v157's initial short gain over v152 motivates this interaction test; it does not
guarantee an additive gain. Exact-equivalence baseline:v153. Inspect resources,
then bounded smoke/sanitizers/full seed and masks before interpreting timing.


### Iteration 153 unlocked source comparison

Same unlocked SourceCounters/WarpStateStats recipe:675737258 dynamic instructions
versus v148679841906; shared28663808 actual/ideal wavefronts with zero excessive.
Executed BAR.SYNC count falls from5021696 to3973120, exactly1048576 fewer warp
instructions (8192 CTAs ×16 stages ×8 compute warps), consistent with removing
one per-stage compute barrier. Other schedule/address differences also contribute
to the total instruction delta. Long-scoreboard samples71132 includePV24796,
producer-empty16116 andQK14293. Sample counts are not duration shares.

### Iteration 157 initial result — packed score scaling survives lowering

REG100/STACK0. SASS audit of the first compute LDTM-to-BAR region finds16 FMUL2
and zero scalar FMUL, versus v153's32 scalar FMUL and zero FMUL2. This is an
actual instruction reduction in that region, not merely packed source notation.
Full seed1234 matches v152 bitwise in three repeats; mask bits match, masked FP32
checks and qualified b2 synccheck/b512 memcheck pass. Short1732.61 us versus
TRT1691.62 us improves6.43 us from v152. Expanded validation is pending.
NCU base/stable:100 registers,203112 shared bytes,occupancy19.330%,tensor44.079%,
eligible0.433828,long-scoreboard6.405203,zero local sectors,aggregate shared
conflicts6344405/2093303,diagnostic2.950208 ms. The higher long-scoreboard ratio
coexists with faster timing; do not treat the ratio as a direct latency fraction.


## Iterations 161–162 — SM103 fused TMEM load/max reduction

The [CUDA13.0 PTX specification](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-ld)
introduces tcgen05.ld.red in PTX8.8 and supports the sm103 family. The installed
CuTeDSL exposes LdRed32x32bOp and tuple destinations for loaded values plus the
reduction. The inspected FlashInfer implementation uses this API in
flashinfer/cute_dsl/attention/roles/mla_compute.py:273 and
attention/monolithic/mla_decode_fp8.py:3073. This avoids inventing a manual TMEM
address mapping or an inline-assembly return convention.

v161 replaces v146's score Ld32x32b.x32 with LdRed32x32b.x32 MAX, retaining the
same partitioned score tensor, .NaN propagation and explicit load wait. A full
validity word uses the loaded maximum combined with the running maximum. Any
holes or partial tile use the original masked-register max reduction. All masking,
probabilities, denominator and PV operations remain as before. The all-valid
branch applies only to max reduction, not a new unmasked probability path.

v162 applies the same experiment to v160. Its hardware maximum is over raw
scores, so scale that scalar before combining with the scaled running maximum;
positive FP32 multiplication is monotonic on these finite inputs. Validity masks
still trigger the original scaled-score reduction when needed. Exact equivalence
must be measured, not assumed from this algebraic argument.

Keep512 TMEM columns and the existing launch footprint. Compile resources and
inspect actual LDTM lowering, then run guarded smoke/memcheck with
MLA_TMEM_GUARDRAILS=1, plus ordinary synccheck and full/masked output equivalence.
Clear the guardrail flag for uninstrumented timing. Baselines:v146/v160.


### Iteration 157 expanded result — packed scaling has a small sustained gain

Full seeds1234/5678 and the full short/chunk0 case match v152 bitwise in three
repeats each; masks and qualified sanitizers pass. Eager20/100 warm/cold1872.02/
1873.92 us versus TRT1884.03/1914.85 us; Graph1923.17/1875.76 us versus
TRT1869.97/1900.58 us. Four rotated Graph orders:

| Cache | v152 us, rounds0–3 | v153 us, rounds0–3 | v157 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1869.82 / 1868.00 / 1869.95 / 1873.82 | 1865.81 / 1865.52 / 1865.89 / 1867.74 | 1863.65 / 1864.27 / 1863.90 / 1865.65 | 1866.40 / 1871.82 / 1873.42 / 1870.54 |
| cold | 1859.65 / 1861.66 / 1860.91 / 1861.60 | 1853.98 / 1855.58 / 1863.70 / 1855.49 | 1852.94 / 1859.50 / 1863.10 / 1851.25 | 1857.57 / 1856.50 / 1860.54 / 1857.34 |

v157 beats both candidates in all four warm orders, but cold comparisons are
mixed. Retain as a fully validated arithmetic-scheduling alternative pending
v160's combination, rather than claiming a cache-independent speedup.
Unlocked source:656335596 instructions, shared28663808 actual/ideal, zero excessive;
long-scoreboard72508, includingPV24738, producer-empty17692 andQK13833.
The total instruction count is below both v148679841906 and v153675737258,
consistent with the confirmed scalar-to-packed scaling reduction, though those
predecessors also differ in bitmap placement/stage release.

### Iterations 158–159 result — Q eviction priority improves aggregate hits, not timing

v158 REG123/STACK0, v159 REG100/STACK0. Each matches its baseline on full seed1234
in three repeats and on masked output bits; v159 masked FP32 tolerance passes,
while v158 retains the fast-path failures. Qualified b2 synccheck/b512 memcheck
report zero errors. Short v1581627.23 us versus TRT1691.65 us is effectively
unchanged from v1461628.10 us; v1591740.86 us versus TRT1689.98 us regresses
from v1531734.72 us. Neither is promoted or expanded.

NCU base/stable(v158/v159):shared194920/203112 bytes,occupancy19.299%/19.331%,
tensor31.804%/43.789%,eligible0.390557/0.469985,long-scoreboard6.146275/6.161828,
zero local sectors,shared-conflict aggregates6311676/2010310 and6432791/1889520,
diagnostic2.782656/2.968992 ms. Aggregate L2 hit rate increases from v14675.044%
to v15877.781%, and from v15375.088% to v15977.841%. These whole-kernel hit rates
do not isolate KV traffic, nor establish a latency benefit. Streaming-Q priority
alone is not a useful performance change in the observed regime.

### Iteration 160 initial result — packed scaling composes with stage release

REG100/STACK0. Full seed1234 matches v153 in three repeats; masks match and
masked FP32 checks plus qualified b2 synccheck/b512 memcheck pass. Short1724.54 us
versus TRT1689.70 us improves10.18 us from v153 and8.07 us from v157. NCU
base/stable:100 registers,203112 shared bytes,occupancy19.334%,tensor44.358%,
eligible0.429091,long-scoreboard6.458982,zero local sectors,aggregate shared
conflicts6420251/2086491,diagnostic2.939168 ms. Expanded validation is pending.


### Iteration 160 expanded result — promote the packed higher-precision path

Both full seeds and the full short/chunk0 case match v153 bitwise in three
repeats each, retaining the documented v148 FP32-reference passes on these
inputs. Eager20/100 warm/cold1872.77/1886.27 us versus TRT1884.06/1914.90 us;
Graph1923.22/1890.34 us versus TRT1873.89/1914.67 us. Four rotated Graph orders:

| Cache | v153 us, rounds0–3 | v157 us, rounds0–3 | v160 us, rounds0–3 | TRT us, rounds0–3 |
|---|---|---|---|---|
| warm | 1867.79 / 1869.68 / 1867.94 / 1871.90 | 1865.66 / 1867.87 / 1867.87 / 1867.92 | 1861.78 / 1861.50 / 1861.76 / 1861.74 | 1866.22 / 1875.90 / 1876.05 / 1878.11 |
| cold | 1857.74 / 1867.78 / 1867.65 / 1868.05 | 1865.49 / 1863.63 / 1863.81 / 1865.47 | 1861.58 / 1857.86 / 1859.73 / 1859.57 | 1861.68 / 1859.87 / 1862.64 / 1865.63 |

v160 beats v153 in all warm orders and three of four cold orders; cold round0
is3.84 us slower. It beats v157 in all eight orders. Promote v160 based on the
combined evidence, retaining that cold regression explicitly. Every recorded
rotated TRT comparison favors v160, but the first cold margin is only0.10 us;
this is parity rather than a convincing standalone win. Separate Graph warm
still favors TRT by about2.6%, so no universal high-precision performance lead.

### Iterations 161–162 compile inspection

v161 REG123/STACK0; v162 REG101/STACK0. Both lower to
LDTM.STAT.x32.MAX.F32.NAN, with the expected PTX tcgen05.ld.red and retained
wait::ld. The installed compiler therefore supports the requested SM103 operation.
Compile-only success is not a runtime, guardrail or output-equivalence result.


## Iterations 163–164 — fold mask work into the fused-load fallback

Based on v161/v162, move the existing per-score mask loop inside the already
required non-full-bitmap branch. All-valid words use unchanged loaded/scaled
scores and the hardware maximum; words containing any hole execute the original
mask loop and software reduction. The bitmap word is uniform across each warp
under the retained mapping. No unconditional masked fast path is introduced.

The old v093 all-valid specialization regressed. The new fused-load variants
already require a max fallback branch, so merging mask work into that same branch
has a different instruction/control-flow tradeoff. Inspect code and registers;
only proceed after the parent load/max versions pass guarded qualification.
Exact baselines:v161/v162, with inherited precision limits checked explicitly.


## Iterations 165–166 — isolate the all-valid branch from hardware max

v165/v166 start from v146/v160 and bypass only per-element masking when the
acquired bitmap word is all ones. Always retain the ordinary Ld32x32b and
software MAX reduction. These are controls for v163/v164: a win from combining
an all-valid branch with hardware max cannot be attributed to the load reduction
without checking the branch alone. The old v093 branch result remains a negative
precedent, not proof of performance on today's packed/scoring pipeline.

They preserve every finite input's selected score values and the original MAX
operation; masked words follow the unmodified loop. Compile/resource, safety,
bitwise and paired timing qualification remain required. Baselines:v146/v160.


### Iterations 161–162 result — fused loading alone is slower

Both guarded b2 smoke and qualified guarded b512 memcheck pass with zero reported
device errors; ordinary b2 synccheck also reports zero. Full seed1234 matches
v146/v160 respectively in three repeats; masks match. v162's masked FP32 check
passes, while v161 retains the documented fast-path tolerance failures.
Short v1611634.46 us versus TRT1691.84 us is6.36 us slower than v146;
v1621730.69 us versus TRT1691.94 us is6.15 us slower than v160.
No expanded audit or promotion for the standalone load/max replacement.

NCU base/stable(v161/v162):registers123/101,shared194920/203112 bytes,
occupancy19.293%/19.312%,tensor31.635%/44.089%,eligible0.395876/0.424030,
long-scoreboard6.255671/6.531636,zero local sectors,aggregate shared conflicts
6315071/2039868 and6285841/1996486,diagnostic2.793152/2.951360 ms.
Hardware support and fewer explicit MAX operations do not by themselves reduce
latency; the new load and fallback control flow must be evaluated together.

### Iterations 163–164 compile qualification

The combined mask fallback compiles at REG109/STACK0 for v163, versus v161123;
v164 uses REG107/STACK0, versus v162101. This lowers fast-path register pressure
but raises it on higher precision. Both retain512 TMEM columns, so no increased
CTA residency is inferred. Parent guarded qualification passed; the children now
require their own guarded and bitwise checks.


## Iterations 167–168 — expose the uniform bitmap branch to the compiler

Based on v163/v164, add only make_warp_uniform to the acquired bitmap value before
QK waiting/masking. The installed CuTeDSL documents this API as a compiler hint,
not a runtime check. Its precondition holds: stage and cgroup are warp invariant,
ctid//64 is constant within every32-thread warp, and all lanes load the same word
while the stage remains owned by the consumers. No producer can rewrite it before
empty release. Do not generalize the hint to arbitrary per-lane validity values.

The combined mask branch unexpectedly regresses despite zero spills. This tests
whether explicitly communicating uniform control can improve its lowering; it is
not yet evidence that divergence caused the regression. Preserve guarded checks,
full/masked bit equivalence and the same profiling/timing protocol. Baselines:
v163/v164. Compare SASS before claiming a uniform-branch mechanism.


### Iterations 163–164 result — combined mask fallback regresses strongly

Guarded b2 smoke and qualified guarded b512 memcheck pass, ordinary b2 synccheck
reports zero errors. Full seed1234 matches v161/v162 respectively in three repeats;
masks match, v164 masked FP32 checks pass and v163 retains fast-path failures.
Short v1631855.87 us versus TRT1691.78 us; v1641957.82 us versus TRT1691.87 us.
Both are much slower than their parents despite zero spills. No promotion or
expanded accuracy claim. The planned software-max and uniform-hint controls
investigate this regression; do not assume that skipped source operations imply
fewer cycles.

NCU base/stable(v163/v164):registers109/107,shared194920/203112 bytes,
occupancy19.343%/19.370%,tensor27.741%/38.716%,eligible0.469447/0.473395,
long-scoreboard5.424880/5.715607,zero local sectors,aggregate shared conflicts
5068054/592415 and4974000/516172,diagnostic3.185216/3.358528 ms.
The lower long-scoreboard ratio and lower aggregate conflict counts accompany
worse performance, not improvement.

### Static code inspection after v163

v161/v163 both retain16 FFMA2 and128 FMUL2 static instructions, so this inspection
does not support lost packed arithmetic as the cause. v163 instead has more
reconvergence/control instructions, including28 BRA.U.ANY entries versus zero in
v161. These static counts do not say how often a branch executes; unlocked source
counters are needed before attributing runtime cost. v165/v166 compile at
REG123/STACK0 andREG105/STACK0; their runtime qualification remains pending.


### Iterations 167–168 compile result — uniform hint is a no-op here

v167 matches v163's3680 instruction encoding words exactly; v168 matches
v164's4176 words exactly. Resource counts also remain109/107 registers and zero
stack. Do not launch redundant runtime/performance tests or infer a speedup from
the source hint. The hint alone does not change this compiler's lowering.

### v161/v163 unlocked source comparison — extra control work is executed

v161596817394 instructions versus v163928480677; both have20275200 actual/ideal
shared wavefronts and zero excessive. FFMA2 count16777216 and FMUL2 count64597504
are unchanged. BSSY grows647168→4382720, BSYNC1179648→4915200, and BRA.U.ANY
0→3670016. The first issuer MMA in v163 is surrounded by ELECT/PLOP/branch work;
v161 issues the corresponding MMA without that per-operation sequence.
This supports an executed control/code-generation regression, not a loss of
packed arithmetic or a shared-bank-conflict explanation. It does not yet identify
the compiler transformation responsible.

v161 long-scoreboard66217 includesPV17369,producer-empty17183,QK13711. v163
70504 includesPV19377,QK18878,producer-empty15958. Sample counts are not latency
shares. Scalar software MAX issue counts are unchanged; predicated instructions
still occupy issue slots, so count totals alone do not imply actual unmasked
reductions occurred on every input.

## Iterations 169–170 — explicit single issuer lane

Based on v163/v164, replace only the two issuer-warp elect_one regions with
if tid ==384. This selects lane0 of warp12 for the same QK/PV sequences and their
completion commits. Q loading and producer gathering retain their original
election. No CTA has more than one issuer thread, and that thread remains the
same throughout the loop. TMEM allocation, masking, arithmetic and synchronization
are unchanged.

The source profile motivates a bounded control for the extra per-MMA election
loops, not a claim that fixed-lane selection is generally better. Compare compiler
output before runtime work. If it changes, repeat guarded smoke/memcheck,
synccheck and full/masked exact equivalence to v163/v164 before timing.


### Iterations 165–166 result — the branch alone also regresses

Full seed1234 matches v146/v160 respectively in three repeats; mask bits match,
v166 masked FP32 checks pass, and qualified b2 synccheck/b512 memcheck report
zero errors. Short v1651672.03 us versus TRT1689.82 us is43.93 us slower than
v146; v1661988.77 us versus TRT1689.89 us is264.23 us slower than v160.
No expanded audit or promotion. The mask-bypass branch alone is therefore not
a useful optimization in this generated code, independently of load/max support.

NCU base/stable(v165/v166):registers123/105,shared194920/203112 bytes,
occupancy19.313%/19.345%,tensor30.953%/38.055%,eligible0.439866/0.462501,
long-scoreboard5.556309/5.876755,zero local sectors,aggregate shared conflicts
6411003/1892005 and5026567/327096,diagnostic2.858496/3.412960 ms.

### Predication audit of the load/max fallback

On the profiled target, v161's per-thread software MAX instructions issue but
have zero predicated-on thread executions. v163's conditional mask SEL and
software MAX instructions likewise have zero predicated-on thread executions.
The full-word path is being selected as intended; source-level branch bypass
has lowered to predication and retained issue work. This further separates
correct branch selection from the additional issuer/control overhead visible
in the dynamic counts.


### Iterations 169–170 compile control — outer lane selection is insufficient

REG109/STACK0 andREG107/STACK0 remain. Explicit lane0 removes only two outer
ELECT instructions (34→32,42→40); BRA.U.ANY counts remain28/36 and the per-MMA
control sequences persist. The proposal fails its intended compiler mechanism,
so reject before runtime benchmarking. No timing or output-equivalence result
is claimed for these compile-only variants.

## Iterations 171–172 — expose invariant MMA operands

Based on v163/v164, mark the MMA destination, accumulator predicate, and both
64-bit shared descriptors warp invariant. Preserve descriptors by applying the
32-bit make_warp_uniform hint separately to their low/high words and reassembling
them; passing64-bit descriptors directly to this32-bit API would truncate them.
The invariant follows from each issuer warp's shared query/stage pointers and
uniform loop indices; operands do not depend on lane IDs or which lane is elected.
Only the elected thread executes the same MMA, as before.

This directly tests operand uniformity behind the observed per-MMA election
loops, after the bitmap hint and explicit issuer lane controls failed. Compile
and compare actual opcode/control lowering first. If unchanged, stop before
redundant GPU trials; if changed, use the existing512-column guarded protocol,
full/masked equivalence and paired timing. Baselines:v163/v164. The hint supplies
a proven precondition, not a runtime repair for nonuniform data.


### Iterations 171–172 — rejected: full-warp shuffle inside elected region

Offline compilation changes the intended control sequences: BRA.U.ANY falls to
zero and ELECT to8, at REG109/STACK0 andREG103/STACK0. However v171's guarded
b2 smoke reaches the configuration print (after compilation), then times out
after90 seconds without correctness output. GPU1 returns to0 utilization and
0 MiB after the timeout; no reset or other process operation was used.

PTX inspection identifies a correctness flaw in this proposal: make_warp_uniform
lowers to shfl.sync.idx.b32 with membermask -1, placed after elect.sync's
non-elected-thread branch. Only the elected lane reaches the shuffle while all
32 lanes are named. The operand being mathematically invariant does not satisfy
the collective's participation contract. The hypothesis text above incorrectly
treated this operation as a non-executing hint. Reject v171; v172 shares this
structural flaw and is not launched. No timing or equivalence is claimed.

Reference: [PTX shfl.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)
specifies that named non-exited lanes must participate.

## Iterations 173–174 — broadcast invariant roots with all issuer lanes active

Start from v163/v164, removing the invalid per-MMA helper approach. Broadcast
only the retrieved TMEM address and nvalid once, in the full32-lane issuer
region before any elect_one. The pointer is reconstructed in TMEM address
space with its original16-byte alignment. All issuer lanes load the same
allocation slot and per-query sequence length. Descriptors, math and completion
commits remain unchanged; only the elected lane issues each MMA.

This tests whether marking the common input roots can avoid the per-MMA
control sequence without putting collectives in single-lane code. Inspect
PTX placement/SASS first, then guarded smoke, qualified sanitizer and equivalence
only if the compile result is useful. No performance claim before measurements.


### Iterations 173–174 result — root broadcasts compile away

Instruction encoding words are exactly identical to v163/v164, respectively.
REG109/107 and zero stack remain, as do the28/36 BRA.U.ANY instructions.
Skip runtime because this adds no executable change and cannot resolve the
observed regression.

## Iterations 175–176 — construct uniform operands before per-MMA election

Based on v171/v172's operand broadcasts, fix collective participation by moving
the elected region into mma_ws, after descriptor/destination/accumulator
broadcasts. All32 issuer lanes execute each broadcast; one elected lane issues
each MMA. The two completion commits retain separate elect_one regions. The
QK/PV operand order, collector policy, barriers and masks are unchanged.

This is a bounded compiler-control experiment: it may add shuffle or election
overhead even if it removes the more expensive per-MMA fallback loops. Inspect
PTX to ensure each full-mask shuffle precedes election, then guarded qualification
and exact equivalence before paired timing. It is not a proposed default yet.


### Iterations 175–176 result — per-operation collectives do not fix lowering

REG109/98, zero stack. The two variants retain34/42 ELECT and28/36 BRA.U.ANY,
with6/11 SHFL.IDX added. They fail the targeted removal of per-MMA control loops.
No GPU launch, correctness or performance claim; skip runtime. The lower
register count of v176 alone does not justify promotion.

## Iterations 177–178 — express producer/issuer/compute roles by uniform warp ID

Based on v163/v164. Replace role comparisons on thread ID with equivalent
comparisons on warp ID: producer warps8–11, issuer warp12, compute warps0–7.
The launch has exactly416 threads, so no partial warp or extra role exists.
Make the warp ID uniform before any role/election branch, with all32 lanes
participating. Intra-role thread indexing and all math/memory operations remain
unchanged.

Earlier operand hints did not resolve the per-MMA election fallback. This tests
control uniformity at the common role split instead of only data uniformity
inside an already divergent region. Compare SASS and PTX before qualification.


### Iterations 177–178 compile result — targeted issuer loops removed

REG113/96, zero stack. ELECT falls34/42→4, BRA.U.ANY falls28/36→0, and
UTCQMMA.WS remains26/34. No SHFL.IDX survives in SASS. PTX inspection confirms
there is no newly inserted full-warp collective inside the elected region.
This meets the intended compiler mechanism and advances to guarded qualification.
It is not yet numerical or runtime-performance evidence.

## Iterations 179–180 — uniform role control on the current defaults

Apply the same warp-ID role split to v146/v160, retaining their original
ordinary score loads and unconditional masking. These controls distinguish a
general role-control effect from recovering v163/v164's mask-branch regression.
No change to arithmetic, TMEM layout, synchronization or output storage.
Compile first and skip runtime if encoding-identical; otherwise qualify with
full/masked output comparisons, sanitizer checks and paired timing.


## Iterations 181–182 — explicit uniform branch around the mask fallback

Based on v177/v178. Replace the DSL mask/fallback region with a register-only
PTX helper using bra.uni. The bitmap is warp invariant because its shared
index depends only on stage, ctid//64 and the128-thread compute group. No
collective is used. If the word is all-valid, preserve all32 score registers
and the hardware-derived maximum through tied assembly operands. Otherwise
set invalid scores to the same -1e30 bit pattern and reduce all32 scores plus
rowmax with16 ternary max.NaN operations. The rest of softmax/PV is unchanged.

The maximum reduction tree changes, but max does not introduce rounding for
finite inputs; use exact full/masked output comparisons to verify generated
behavior. This is motivated by v177 still issuing predicated mask/maximum work
after eliminating the issuer loops. Compile and inspect actual branching and
register/stack costs before guarded qualification and timing.


### Iterations 177–178 measured result — recovery without a new default

Guarded b2 smoke/b512 memcheck and b2 synccheck report zero errors. Full8192-row
seed1234 matches v146/v160 in three repeats, and all masked-input bits match.
v178 masked FP32 checks pass; v177 retains v146's86 masked tolerance failures.

Short paired v1771700.00 us versus TRT1691.90 us improves v1631855.87 us, but
is71.90 us slower than v146. v1781790.21 us versus TRT1693.70 us improves
v1641957.82 us, but is65.67 us slower than v160. The role split recovers a
large code-generation regression; it does not make the load/max+mask combination
a better default. No expanded audit or promotion.

NCU base/stable(v177/v178):registers113/96,shared194920/203112 bytes,
occupancy19.310084%/19.337507%,tensor30.377359%/42.488447%,
eligible0.476185/0.502616,long-scoreboard5.317375/5.496871,zero local sectors,
aggregate shared conflicts6458869/1828663 and6460286/1805829,
diagnostic2.912096/3.059264 ms.

Unlocked source v177 has703517916 instructions versus v163928480677, with
20275200 actual/ideal shared wavefronts and zero excessive. Long-scoreboard
65148 includesPV17828,producer-empty16546,QK13350. These are sampled stall
counts, not latency shares. Static inspection still shows predicated mask and
software-max instructions occupying issue slots, motivating the explicit
uniform fallback branch in v181/v182.

### Iterations 179–180 result — role change alone has no short-run gain

REG123/101,zero stack. Both retain4 ELECT and no BRA.U.ANY. Full seed1234
outputs match v146/v160 in three repeats; masks match exactly, v180 masked FP32
checks pass, and qualified b512 memcheck/b2 synccheck have zero errors.

Short v1791631.49 us versus TRT1691.78 us is3.39 us slower than v146.
v1801728.96 us versus TRT1691.58 us is4.42 us slower than v160. No expanded
audit or promotion. This control supports targeting the mask-branch lowering,
not a blanket claim that warp-ID roles are faster for every kernel.

NCU base/stable(v179/v180):registers123/101,shared194920/203112 bytes,
occupancy19.298517%/19.324486%,tensor31.736470%/44.165733%,
eligible0.388634/0.426351,long-scoreboard6.168906/6.469596,zero local sectors,
aggregate shared conflicts6455352/2110299 and6426750/2131341,
diagnostic2.786112/2.945408 ms.


### Iterations 181–182 compile result — direct branch still if-converted

Both compile successfully atREG123/103 and zero stack. However SASS still
predicates the entire mask/software-max region instead of branching over it.
The bra.uni source annotation does not force a native branch. Skip runtime: the
specific intended issue-slot bypass was not achieved. No correctness/performance
claim for these compile-only variants.

## Iterations 183–184 — indexed uniform branch to preserve fallback skipping

Based on v181/v182, change only the direct conditional PTX branch to brx.idx.uni
with a local two-entry .branchtargets table. A selp produces index0 for the
all-valid word and index1 otherwise, so the index is always in bounds. All
active lanes within each compute warp share the bitmap and index. Both labels
are within the same inline-assembly scope/function; no external target exists.

This is a compiler-lowering control for the observed if-conversion, not an
assumption that indirect branches are generally faster. Inspect generated SASS
for an actual skip before guarded qualification and exact full/masked checks.
Reference: [PTX brx.idx](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-brx-idx).


Before any v183/v184 launch, strengthen output constraints to early-clobber
(=&f with tied score/fullmax inputs). The assembly modifies scores before its
last reads of bitmap and rowmax; those independent late-read inputs must not
share output registers. This completes the assembly operand contract. Recompile
and inspect resources/SASS after this correction; earlier compile files are
retained with an initial prefix. Neither initial candidate was launched.


### Iterations 183–184 corrected compile result

The corrected assembly constraints compile atREG123/122, zero stack and an
8-byte constant branch table. Both contain one native BRX between load/max and
the mask fallback. Mask SEL and16 FMNMX3.NAN instructions now sit after that
branch rather than being individually predicated by the all-valid condition.
MMA counts remain26/34 with4 ELECT and no per-MMA BRA.U.ANY loops.

Corrected source SHA256:v183
eaab6acfcb3d0039a4d3a25d458b47e75897320a4e89bb2f56e0484046367e79;
v184 a033188d7b0685e0edc3980d0fc5f24ef3b2be2564b59f2593face2e553e415d.
These are the sources entering guarded qualification; no initial-constraint
variant is used for runtime tests.


### Iterations 183–184 initial runtime qualification

Guarded b2 smoke/b512 memcheck and b2 synccheck pass with zero errors. Full
seed1234 matches v146/v160 in three repeats; masked-input bits also match and
v184 masked FP32 checks pass. v183 retains the same86 masked FP32-tolerance
failures as v146; bitwise equivalence is not a full FP32 pass.

Short paired v1831550.50 us versus TRT1691.81 us improves the recorded v146
short median1628.10 us by77.60 us. v1841693.89 us versus TRT1689.76 us improves
v1601724.54 us by30.65 us. These initial results justify expanded seed/short
equivalence,100-repeat eager/Graph measurements, and rotating-order comparisons
to the current defaults. No promotion from the short medians alone.


## Iterations 185–186 — retain indexed fallback, restore original role predicates

Based on corrected v183/v184, restore v146/v160's thread-ID role comparisons
and ordinary warp_idx call. The explicit branch now lives entirely inside the
register-only assembly helper, so it may no longer trigger the DSL control
interaction that required warp-ID role comparisons in v177/v178. The earlier
role-only control v179/v180 did not improve default timings.

This separates the role-predicate workaround from the useful native fallback
skip. All assembly constraints, masks, arithmetic and synchronization stay
unchanged. Compile after current extended timing finishes; inspect whether the
per-MMA loops return, then qualify only if the proposed simplification survives
code generation. No runtime result yet.


### Iterations 183–184 extended result — promote both experimental defaults

Both full seeds1234/5678, all1024 short-case rows at chunk0/seed5678 and
masked inputs match v146/v160 bitwise in the recorded audits. Qualified
sanitisers pass. v183 inherits all fast-path tolerance failures; v184 retains
v148's independently audited full-reference passes through exact equivalence.

Extended20/100 event warm/cold:v1831655.50/1646.51 us versusTRT1880.22/1916.94;
v1841829.20/1855.50 versus1871.90/1916.22. Graph:v1831663.04/1644.51 versus
1869.81/1914.62;v1841901.55/1849.60 versus1867.89/1916.85. The separate Graph
warm high-precision case still trailsTRT by about1.8%; do not replace it with
the faster rotating-order measurements or claim universal superiority.

Four rotating-order Graph medians, in round order:


v183 warm: candidate1614.02/1614.30/1629.39/1632.34 us; v1461671.46/1673.54/1673.57/1673.57 us; TRT1868.93/1877.73/1878.30/1880.30 us.

v183 cold: candidate1599.28/1619.09/1617.06/1618.05 us; v1461665.28/1675.26/1674.43/1675.02 us; TRT1857.54/1857.73/1857.65/1865.73 us.

v184 warm: candidate1847.42/1847.42/1847.36/1849.23 us; v1601861.86/1861.76/1861.76/1880.06 us; TRT1874.16/1884.53/1883.60/1887.41 us.

v184 cold: candidate1826.66/1824.72/1826.77/1826.62 us; v1601859.65/1857.58/1861.57/1859.66 us; TRT1859.55/1859.63/1865.78/1867.68 us.

Both candidates beat their previous defaults in all eight same-run warm/cold
comparisons, so promote fast v183 and higher-precision v184. The absolute
speedup varies with execution regime; original scripts/tolerances are unchanged
and SHA256 matches the user's source. Continue tuning from these versions.

NCU base/stable(v183/v184):registers123/122,shared194920/203112 bytes,
occupancy19.293211%/19.323837%,tensor33.523619%/45.188299%,
eligible0.367788/0.377673,long-scoreboard6.431440/6.972028,zero local sectors,
aggregate shared conflicts6473487/3937124 and6537068/2735684,
diagnostic2.639232/2.878304 ms.

Unlocked source:v183541013498 instructions versusv146600988722 andv177703517916;
v184596787377 instructions. Shared actual equals ideal at20275200/28663808,
with zero excessive. v183 long-scoreboard62208 includesPV16824,producer-empty
15589,QK12951;v18472229 includesPV24358,producer-empty17207,QK13805. These
sample counts are not latency shares. The fallback MAX issue audit is retained
in v183_mask_issue_audit.json to distinguish actual skipping from predication.


### Iterations 185–186 compile result

REG123/122 and zero stack; one BRX remains, with4 ELECT and no per-MMA
BRA.U.ANY. Encoding words differ from v183/v184, so qualify rather than
assuming equality. The original thread-ID role predicates no longer recreate
the prior issuer loop when fallback control is inside the PTX helper.

## Iterations 187–188 — reduce temporary registers in the masked maximum

Based on corrected v183/v184, retain the same indexed branch and exact masking,
but reduce33 values with a single dependent accumulator rather than a
16-temporary tree. Both use16 ternary max.NaN instructions. This does not
change finite maximum values; full/masked bitwise checks remain required.

The goal is to lower peak register pressure from the fallback even when the
all-valid workload skips it, particularly v184's122 registers. The tradeoff is
a longer dependency chain when a partial word actually executes the fallback.
Compile first; if register pressure is unchanged, do not infer an improvement
from fewer source-level temporaries. No promotion without actual measurements.


### Iterations 187–188 compile result — fewer declared temporaries, same registers

Both compile with the same123/122 registers and zero stack as v183/v184.
The16-maximum instruction count is unchanged. The proposed register-pressure
benefit did not materialize, while the fallback dependency chain became longer.
Reject before GPU launch; no correctness or runtime result is claimed.

## Iteration 189 — in-place packed score scaling in the higher-precision helper

Based on v184. Move the16 packed score multiplies into the existing assembly
helper before the indexed branch, using tied raw-score inputs and scaled-score
outputs. Explicit mul.rn.f32x2 uses the same FP32 constant bit pattern
0x3db8aa3b in both halves, with no FTZ modifier. Early-clobber constraints keep
late-read bitmap and rowmax inputs separate. Hardware maximum scaling, masking,
reduction and subsequent softmax/PV arithmetic stay unchanged.

SASS in v184 writes scaled scores to a separate32-register range from the
TMEM-loaded scores. This tests in-place register reuse without changing packed
arithmetic count or rounding. Inspect resource usage and emitted FMUL2/BRX
before guarded qualification and full/masked bitwise comparisons to v184.


### Iteration 189 compile result — no in-place allocation benefit

REG122/STACK0 and the separate raw/scaled32-register ranges remain. Of3200
encoding words, only32 differ:16 later FADD2 instructions exchange commutative
operands and corresponding reuse flags. The intended score-allocation change
did not occur. Reject before runtime; no GPU launch or correctness/performance
claim. Detailed comparison is retained in v189_sass_difference.json.

## Iterations 190–191 — output cache policy after removing mask issue work

Based on v183/v184, add L2 evict-first only to the existing256-bit output stores,
as in earlier validated v141/v133. Output layout/alignment, arithmetic and
synchronization are unchanged. The objective is to reduce competition from
512 MiB of output stores with gathered KV reuse. Earlier cache-policy results
were small and mixed; they are not assumed to transfer to this implementation.

The new fast path removes about10% of v146's issued instructions. Retest whether
this makes the previously observed DRAM-traffic reduction useful on its shorter
critical path. Use paired latency and compare hardware DRAM byte metrics, not
just aggregate L2 hit rate. Compile, qualify, then measure.


### Iterations 185–186 result — validated alternatives, defaults unchanged

Guarded b2 smoke/b512 memcheck and b2 synccheck report zero errors. Both full
seeds and the short case match v183/v184 bitwise in three repeats, and masks
also match; v186 retains the higher-precision masked FP32 pass.

v185 warm: candidate1611.70/1616.08/1620.61/1612.16 us; v1831614.00/1613.92/1632.37/1621.14 us; TRT1858.27/1874.08/1876.10/1874.59 us.

v185 cold: candidate1615.82/1597.46/1615.89/1616.00 us; v1831618.06/1617.82/1618.90/1618.51 us; TRT1857.74/1859.55/1859.55/1866.96 us.

v185 wins7/8 same-run comparisons against v183.

v185 cuda-event: trtllm/native warm 1876.00 us, cute-v185/native warm 1642.66 us, trtllm/native cold 1918.11 us, cute-v185/native cold 1651.09 us.

v185 cuda-graph: trtllm/native warm 1870.08 us, cute-v185/native warm 1660.54 us, trtllm/native cold 1909.81 us, cute-v185/native cold 1646.61 us.

v185 NCU base/stable: launch__registers_per_thread=123 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.279593 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.543216 %, smsp__warps_eligible.avg.per_cycle_active=0.367543 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.442222 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,358,042 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,580,654 , gpu__time_duration.sum=2.636736 ms.

v186 warm: candidate1843.42/1844.46/1845.30/1844.00 us; v1841824.91/1847.47/1849.41/1847.47 us; TRT1870.80/1878.19/1880.30/1878.11 us.

v186 cold: candidate1824.14/1824.66/1824.67/1826.67 us; v1841828.83/1825.73/1828.30/1825.01 us; TRT1865.62/1867.87/1865.86/1866.67 us.

v186 wins6/8 same-run comparisons against v184.

v186 cuda-event: trtllm/native warm 1872.48 us, cute-v186/native warm 1840.64 us, trtllm/native cold 1914.90 us, cute-v186/native cold 1849.36 us.

v186 cuda-graph: trtllm/native warm 1869.90 us, cute-v186/native warm 1888.46 us, trtllm/native cold 1914.94 us, cute-v186/native cold 1851.14 us.

v186 NCU base/stable: launch__registers_per_thread=122 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.314710 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.240755 %, smsp__warps_eligible.avg.per_cycle_active=0.375969 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.946855 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,336,957 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,699,212 , gpu__time_duration.sum=2.872736 ms.

Short paired v1851549.28 us versus TRT1691.74 us;v1861689.79 versus1691.81.
These improve the corresponding v183/v184 short records by1.22/4.10 us.
Rotated comparisons have regressions, and separate event/Graph comparisons
against previous extended records are mixed. Keep these as validated
alternatives instead of replacing v183/v184 with a small regime-dependent gain.


## Compiler control — isolated CuTeDSL 4.6.3

Official release notes for4.6.3 and4.7.1 report the setmaxnreg/warp-specialized
compilation fix, and the maintainer confirms issue3420 was fixed in both:
https://github.com/NVIDIA/cutlass/releases/tag/v4.6.3
https://github.com/NVIDIA/cutlass/issues/3420#issuecomment-5433964668

Test4.6.3 as the smallest patch over installed4.6.2. Install matching DSL,base,
core,cu12 andcu13 wheels into a task-local compiler_envs directory, selected
only by the current process PYTHONPATH. Leave /opt/sglang untouched. Record all
package versions and loaded paths. First compile unchanged v183/v184 and the
previously failing v124/v137; inspect register, stack and SASS differences.
Compilation success is not a runtime validation or a speedup.

The Pod's configured mirror timed out; its external network is unavailable and
its global pip constraint pins4.6.2. Download wheels locally from official PyPI,
verify published SHA256, transfer into the task directory and install offline
with process-local constraints disabled. Preserve download metadata and logs.

## Iterations 192–193 — retry complete-warpgroup register redistribution

Based on v183/v184. Launch512 threads so both donor warpgroups are complete;
warps8–15 decrease to32 registers and compute warps0–7 increase to192.
Producer warps8–11 and issuer warp12 keep their existing work; warps13–15
only donate registers. v193 explicitly restricts its compute branch towarp<8.
Require min_blocks_per_mp=1 and inspect the compiler's initial allocation
before launch. Final requested budget is57344 registers, below65536/SM.

This reopens v137's failed lowering experiment after an official compiler fix,
on the current load/max/indexed-mask implementation. No TMEM extent, arithmetic,
barrier or pipeline change is intended. More registers could improve scheduling
even without baseline spills; no gain is assumed. Compile first, verify complete
warpgroup control and register budget, then guarded smoke/memcheck, synccheck,
full/masked equivalence and paired measurements under the same compiler.


### Iterations 190–191 result — small cache-policy gain with mixed execution regimes

REG123/122 and STACK0; output SASS changes to STG.E.NA.EFL2.256. Ordinary b2 smoke, b512 memcheck and b2 synccheck pass with zero errors. Full seeds1234/5678 and the1024-row short case match v183/v184 bitwise for three repeats, as do masked inputs. v190 inherits fast-path tolerance failures; v191 retains the audited higher-precision passes.

Short v190: trtllm/native 1691.94 us, cute-v190/native 1549.06 us.

v190 warm: cute-v190/native 1609.73/1611.47/1610.88/1611.78 us; cute-v183/native 1611.98/1612.06/1612.86/1613.86 us; trtllm/native 1866.80/1867.92/1869.84/1867.87 us.

v190 cold: cute-v190/native 1595.39/1595.57/1593.52/1595.42 us; cute-v183/native 1616.19/1601.47/1616.10/1607.73 us; trtllm/native 1853.26/1855.31/1857.30/1857.36 us.

v190 cuda-event20/100: trtllm/native warm 1863.14 us, cute-v190/native warm 1630.27 us, trtllm/native cold 1896.74 us, cute-v190/native cold 1646.05 us.

v190 cuda-graph20/100: trtllm/native warm 1861.62 us, cute-v190/native warm 1642.51 us, trtllm/native cold 1896.26 us, cute-v190/native cold 1632.22 us.

v190 NCU base/stable: launch__registers_per_thread=123 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.282596 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.524179 %, smsp__warps_eligible.avg.per_cycle_active=0.368032 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.424515 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,474,622 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,831,716 , gpu__time_duration.sum=2.637696 ms.

Short v191: trtllm/native 1691.71 us, cute-v191/native 1691.84 us.

v191 warm: cute-v191/native 1820.82/1820.77/1822.32/1822.96 us; cute-v184/native 1825.20/1826.05/1828.99/1825.65 us; trtllm/native 1862.38/1867.87/1867.86/1869.58 us.

v191 cold: cute-v191/native 1807.89/1808.30/1810.43/1812.46 us; cute-v184/native 1822.66/1822.62/1822.90/1824.83 us; trtllm/native 1850.88/1855.60/1855.74/1857.47 us.

v191 cuda-event20/100: trtllm/native warm 1869.94 us, cute-v191/native warm 1826.70 us, trtllm/native cold 1901.58 us, cute-v191/native cold 1853.47 us.

v191 cuda-graph20/100: trtllm/native warm 1859.55 us, cute-v191/native warm 1884.24 us, trtllm/native cold 1895.54 us, cute-v191/native cold 1851.68 us.

v191 NCU base/stable: launch__registers_per_thread=122 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.320824 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.208293 %, smsp__warps_eligible.avg.per_cycle_active=0.377666 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.964201 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,526,396 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,618,748 , gpu__time_duration.sum=2.878272 ms.

Unlocked/dynamic NCU with cache-control none, after10 warmups: v183/v190 DRAM reads1.561872/1.427949 GB, writes527.141632/525.975552 MB, L2 hit75.304519%/76.874009%, diagnostic1.545024/1.546144 ms. v184/v191 reads1.562872/1.427605 GB, writes526.805248/526.225152 MB, hit75.296663%/76.890635%, diagnostic1.688832/1.689056 ms. Read traffic falls8.57%/8.65%, with no improvement in these profiled durations.

Both versions win all eight same-run rotating-order comparisons, with the larger benefit under cold cache. This supports a small cache-policy improvement, not a proportional8.6% speedup. v191 standalone Graph warm remains1884.24 us versusTRT1859.55 us; its cold1851.68 us also does not beat the previous independently measured v1841849.60 us. Preserve regime dependence and unlocked-clock distributions.

Promote v190/v191 as experimental defaults based on the eight paired wins and complete recorded equivalence audits. Retain all standalone-regime limitations above. v192/v193 deliberately compare to their unchanged v183/v184 parents to isolate compiler/register effects from this cache hint.


### Compiler 4.6.3 / iterations 192–193 compile result

Isolated imports and native compiler library resolve inside compiler_envs/cutlass463;
all five packages report4.6.3. Unchanged v183/v184 compile to exactly the same
SASS encoding words as4.6.2, with REG123/122 and zero stack. No runtime control
is inferred beyond native code identity. Old v124/v137 and new v192/v193 still
fail with ptxas C7600: register allocation failed with target192. None launched.
The official fix does not resolve these specific candidates.

## Iterations 194–195 — keep register increase inside the compute branch

Based on v192/v193. Keep the32-register decrease for both complete donor
warpgroups, but move compute's192-register request past Q setup into the
warp<8 role branch. This tests whether the common control-flow merge after
inc/dec creates constraints before the true role split. Q setup remains on
warp0 before its increase, under the initial allocation. All math and tile
operations are unchanged. Compile-only until resource/budget inspection passes.

## Iterations 196–197 — increase producer/issuer register allowance

Based on v194/v195. Use64 registers for both donor warpgroups and176 for
compute. This separates insufficient donor allowance from compute pressure;
the total request is61440 and requires initial allocation of at least120
registers per thread for512 threads. Check the actual initial allocation and
complete-warpgroup control before any runtime test. More total SM capacity
does not substitute for sufficient registers in this CTA's own pool.
Primary semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg


### Iterations 194–197 compile controls

v194/v195 now compile on4.6.3 with REG128, STACK0 and two native USETMAXREG
instructions. Both use initial65536-register CTA allocation and final57344,
with eight complete donor and eight complete compute warps. v196 compiles
but has STACK24 and static5STL/6LDL; reject it before runtime. v197 uses the
64/176 budget with REG128, STACK0 and final61440 registers.

Cross-compile v194/v195/v197 with installed4.6.2: all also compile with
REG128/STACK0, but SASS encodings differ from4.6.3. Therefore the successful
control-flow change does not require the patch compiler; qualify and time
these candidates under installed4.6.2 first. Inherited source comments
recommending4.6.3 predate this control; they are not a dependency requirement.
No newer-compiler runtime/performance result is claimed yet.

Old4.6.2 v124 logs report NVVM compilation failure;4.6.3 reaches PTX assembly
and reports register-allocation C7600. These are distinct failure stages. The
new patch alone is insufficient, rather than evidence its advertised fix has
no effect. Moving the increase into the compute branch resolves compilation
for the current native x32 implementation on both compilers.

## Iterations 198–199 — native x64 correction with successful role-local allocation

Based on v194/v195. Replace only output-correction copies with two native
64-column loads/stores per compute thread. Final BF16 output keeps32-column
copies, and score load/max/indexed masking remains unchanged. Fast v198
replaces pairedx32 loads/stores; higher-precision v199 replaces four serial
x32 copies. Preserve packed FP32 correction, TMEM fences and P/PV readiness.

This revisits x64 after resolving the allocation control-flow failure; compiler
versions remain separate controls. Inspect static registers/spills and actual
x64 instructions, then require guarded TMEM/memory, sync and full/masked
bitwise checks before judging speed. No result yet.


### Iterations 194,195,197 — installed-compiler qualification and short measurements

All three use installed CuTeDSL4.6.2. Guarded b2 smoke/b512 memcheck and b2 synccheck pass with zero errors. Full seed1234 matches the relevant v183/v184 parent bitwise in three repeats, and masked comparisons also match; v195/v197 pass the masked FP32 reference. No other compiler runtime result is claimed.

v194 short: trtllm/native 1690.82 us, cute-v194/native 1560.77 us.

v194 NCU base/stable: gpu__time_duration.sum=2.656800 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.389328 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.286282 %, smsp__warps_eligible.avg.per_cycle_active=0.352976 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.532359 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,568,689 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=4,282,564 .

v195 short: trtllm/native 1691.71 us, cute-v195/native 1673.41 us.

v195 NCU base/stable: gpu__time_duration.sum=2.838752 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.405082 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.820621 %, smsp__warps_eligible.avg.per_cycle_active=0.386270 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.766837 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,913,938 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,989,857 .

v197 short: trtllm/native 1691.94 us, cute-v197/native 1662.30 us.

v197 NCU base/stable: gpu__time_duration.sum=2.817632 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.392097 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=46.154693 %, smsp__warps_eligible.avg.per_cycle_active=0.381841 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.816036 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,954,856 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,989,558 .

Launch registers128 describe the initial allocation; actual role limits are32/192 forv194/v195 and64/176 forv197. The extra three donor warps exit after donation, so512 launch threads do not imply512 continuously active useful threads. Hardware local read/write sectors are zero in all three.

v1941560.77 us is slower than v1831550.50, despite more compute registers. Do not extend it solely for speed; retain it as a qualified x64 parent. v1951673.41 andv1971662.30 improve v1841693.89 by20.48/31.59 us and warrant full-seed/short equivalence and longer same-process comparison against currentv191. Defaults remainv190/v191 pending those comparisons.

### Iterations 198–199 compile result

Both native x64 candidates compile on4.6.2 and4.6.3 with REG128/STACK0, two LDTM.x64 and two STTM.x64 instructions, and no static local load/store instructions. The role-local allocation also resolves this x64 compilation pattern on the installed compiler; no toolchain upgrade is required for qualification. Keep the two compiler artifacts separate and use4.6.2 for runtime first.


## Iterations 200–201 — isolate donor allowance from compute allowance

Based on v194/v195, increase only donor registers32→64, retaining compute192.
This distinguishes v197's faster64/176 result from both changed role budgets.
The final budget is65536, exactly the512-thread CTA allocation at128 registers
per thread. Do not launch if the compiler/driver reports a smaller initial
allocation; the extra physical SM capacity alone would not avoid a pool stall.
Both donor warpgroups remain complete and compute allocation remains inside
its own branch. No arithmetic, cache hint, copy width or synchronization change.
Compile and inspect first; runtime only after budget and spill checks.


### Iterations 198–199 runtime result — native x64 does not improve these parents

Installed4.6.2: guarded b2 smoke/b512 memcheck and b2 synccheck all pass. Both candidates match v183/v184 on full seed1234 for three repeats and on masked inputs; v199 retains the masked FP32 pass. No other full seeds or extended performance claimed for these candidates.

v198 short: trtllm/native 1689.92 us, cute-v198/native 1573.95 us.

v198 NCU base/stable: gpu__time_duration.sum=2.674272 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.379767 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.076386 %, smsp__warps_eligible.avg.per_cycle_active=0.342405 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.627931 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,620,837 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=4,328,180 .

v199 short: trtllm/native 1691.78 us, cute-v199/native 1673.44 us.

v199 NCU base/stable: gpu__time_duration.sum=2.837824 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.411569 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.875816 %, smsp__warps_eligible.avg.per_cycle_active=0.385690 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.765491 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,939,983 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,020,157 .

v1981573.95 us regresses from v1941560.77 and v1831550.50. v1991673.44 is essentially unchanged from v1951673.41 and slower than v1971662.30. Reject native x64 as a speed optimization here, despite successfully removing the historical compile/spill barrier.

### Iterations 195/197 extended result — promote higher-precision v197

warm cute-v184/native: 1847.42/1847.36/1847.50/1847.36/1845.38 us.

warm cute-v191/native: 1824.86/1824.86/1823.66/1826.94/1824.80 us.

warm cute-v195/native: 1824.77/1824.64/1824.93/1824.82/1820.62 us.

warm cute-v197/native: 1816.67/1810.59/1812.30/1811.50/1810.59 us.

warm trtllm/native: 1867.94/1878.21/1884.11/1885.36/1886.96 us.

cold cute-v184/native: 1825.44/1826.83/1825.02/1824.72/1822.72 us.

cold cute-v191/native: 1822.51/1822.70/1824.83/1826.51/1822.62 us.

cold cute-v195/native: 1808.51/1808.46/1808.59/1812.54/1814.29 us.

cold cute-v197/native: 1795.65/1798.11/1798.56/1799.47/1796.32 us.

cold trtllm/native: 1867.66/1868.27/1861.86/1865.62/1865.58 us.

v195 cuda-event20/100: trtllm/native warm 1871.28 us, cute-v195/native warm 1837.02 us, trtllm/native cold 1916.48 us, cute-v195/native cold 1845.52 us.

v195 cuda-graph20/100: trtllm/native warm 1867.74 us, cute-v195/native warm 1884.32 us, trtllm/native cold 1906.80 us, cute-v195/native cold 1847.44 us.

v197 cuda-event20/100: trtllm/native warm 1873.04 us, cute-v197/native warm 1821.74 us, trtllm/native cold 1917.09 us, cute-v197/native cold 1833.15 us.

v197 cuda-graph20/100: trtllm/native warm 1884.29 us, cute-v197/native warm 1811.97 us, trtllm/native cold 1918.93 us, cute-v197/native cold 1836.06 us.

Both candidates match v184 in the second full seed5678 and1024-row chunk0/seed5678 short case, three repeats each; earlier full seed1234/mask and guarded checks remain valid. v197 beats current v191, its v184 parent and v195 in all ten warm/cold rotating-order comparisons. Its standalone Graph warm/cold1811.97/1836.06 us also beat pairedTRT1884.29/1918.93. Promote v197 as the higher-precision experimental default; fast default remainsv190. This changes the evidence relative to earlier standalone warm regressions, without promising all workloads or clock regimes.

Unlocked source:v195629565014 issued instructions versusv197599875543, a4.72% reduction. The latter is still slightly above v184596787377, so total instruction count is not by itself the performance explanation. Both have shared actual=ideal28663808 and zero excessive. Long-scoreboard sample totals70468/69999:PV24586/24069, producer-empty16205/16516,QK13863/13000; these are sample counts, not latency shares.

The opcode audit records18 static MOV.SPILL and18 R2UR.FILL in v195, each with2359296 issued warp instructions; none remain in v197. The operands visibly move values between UR and R registers, not local memory, consistent with both kernels reporting zero local sectors. The two types account for4.72 million of the29.69 million total instruction reduction; do not attribute the entire gain to these pairs. UTCQMMA.WS and indexed BRX counts are preserved.

### Iterations 200–201 compile result

Both compile on4.6.2 with REG128/STACK0, satisfying the initial65536-register pool required by final64/192 allocation. v200 retains6 MOV.SPILL/R2UR.FILL pairs; v201 has none. Native code differs from v197, so qualify and measure instead of treating the changed budget as a no-op.


## Iteration 202 — combine output eviction hint with the higher-precision register path

Based on validated v197. Add only output L2 evict-first, as in v191, retaining
64/176 role budgets, x32 TMEM copies and all arithmetic/synchronization. The
hint reduced DRAM reads and helped rotated cold latency on v184/v191, but
did not improve profiled warm duration. Re-evaluate the actual latency on this
new parent; do not assume the earlier benefit is additive. Compile and inspect
SASS/resources, then memory/sync and equivalence checks before paired timing.


### Iterations 200–201 result — maximum 64/192 budget adds no default-path gain

Guarded b2 smoke/b512 memcheck, b2 synccheck and full seed1234 three-repeat exact equivalence pass. Both mask comparisons match v183/v184; v201 passes masked FP32. These runtime results confirm the recorded initial65536-register CTA pool supports the exact65536 requested budget. No extended equivalence/timing is claimed for these candidates.

v200 short: trtllm/native 1691.62 us, cute-v200/native 1553.50 us.

v200 NCU base/stable: gpu__time_duration.sum=2.644160 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.381442 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.426260 %, smsp__warps_eligible.avg.per_cycle_active=0.351607 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.548784 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,551,573 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=4,205,063 .

v201 short: trtllm/native 1692.22 us, cute-v201/native 1665.22 us.

v201 NCU base/stable: gpu__time_duration.sum=2.826272 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.403410 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=46.039772 %, smsp__warps_eligible.avg.per_cycle_active=0.380690 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.910416 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=7,253,624 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,016,140 .

v2001553.50 us improves its32/192 parent v1941560.77 but remains above v1831550.50 and current v1901549.06. v2011665.22 improves v1951673.41 but is slower than v1971662.30. No default change. The donor64 result supports keeping more registers for producer/issuer roles; compute192 over176 does not improve the observed high-precision short case.


## Isolated compiler runtime control — v197 under4.6.2 and4.6.3

The successful register-role kernels compile to different SASS under the two
compilers. Test the same v197 source under isolated4.6.3 only after guarded
smoke/memory and sync qualification. Compare complete byte hashes of Q,KV,
indices,lens and BF16 output across two separate compiler processes, for both
full seeds and the1024-row short case, three output repeats each. This avoids
loading conflicting compiler libraries into one Python interpreter. Hashing
and host copies are validation-only and excluded from benchmark timing.

The comparison helper records source hashes, loaded compiler path, package and
torch versions. Exact output equality inherits v197's audited precision scope;
it is not a separate FP32-reference proof. Keep compiler-labelled artifacts
separate from the default4.6.2 benchmark records. Compare paired TRT timings
under the same original benchmark method if qualification succeeds.

The compiler hash control also includes the exact two-row hole/partial-tile
fixture from audit_mask_precision.py, with input bytes hashed after mutation.


### Iteration 202 result — validated cache-policy alternative, not promoted

REG128/STACK0 and STG.E.NA.EFL2.256. Ordinary b2 smoke, b512 memcheck and b2 synccheck pass. Both full seeds, short case and masked inputs match v197 bitwise; all original higher-precision audit limits are inherited.

full_event: trtllm/native warm 1691.46 us, cute-v202/native warm 1661.12 us.

cuda-event_event100: trtllm/native warm 1881.30 us, cute-v202/native warm 1817.22 us, trtllm/native cold 1913.89 us, cute-v202/native cold 1814.46 us.

cuda-graph_event100: trtllm/native warm 1873.74 us, cute-v202/native warm 1869.94 us, trtllm/native cold 1916.90 us, cute-v202/native cold 1826.11 us.

warm cute-v197/native: 1812.78/1811.09/1808.58/1818.64 us.

warm cute-v202/native: 1810.02/1810.05/1810.43/1809.84 us.

warm trtllm/native: 1865.97/1874.02/1878.16/1882.13 us.

cold cute-v197/native: 1798.99/1796.21/1812.45/1798.13 us.

cold cute-v202/native: 1792.03/1793.97/1794.54/1793.84 us.

cold trtllm/native: 1859.62/1859.70/1858.51/1863.57 us.

NCU base/stable: gpu__time_duration.sum=2.816832 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.397456 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=46.170841 %, smsp__warps_eligible.avg.per_cycle_active=0.382082 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.816106 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,954,999 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,849,184 .

v202 wins7/8 same-run comparisons but loses one warm round by1.86 us. Its separate Graph warm1869.94 us is substantially above v1971811.97 us in the preceding independent record; the pairedTRT differs too. Cold performance improves, but a uniform additive gain is not established. Keep v202 as a fully validated alternative; default remains v197.


### Isolated v197 compiler control — identical audited bytes, no4.6.3 speedup

New4.6.3 passes guarded b2 smoke/b512 memcheck and b2 synccheck. All12 input/output records match4.6.2 exactly: both full seeds,1024-row short case and the two-row hole/partial-tile fixture, each repeated three times. Kernel SHA256 ab5db755204cc204a45fbb7b94c7e9d9a8e5c53d0c50ae183c5a191a4c0dfaa7. Input hashes include Q,KV,indices and lengths; output hash covers every BF16 byte. Hashing and copies are outside performance measurements.

462 warm cute-v197/native: 1842.22/1888.35/1869.90/1863.70 us.

462 warm trtllm/native: 1865.20/1869.02/1868.83/1872.02 us.

462 cold cute-v197/native: 1831.17/1832.82/1832.78/1835.14 us.

462 cold trtllm/native: 1917.09/1910.96/1916.93/1916.99 us.

463 warm cute-v197/native: 1874.02/1888.34/1892.54/1878.27 us.

463 warm trtllm/native: 1869.28/1867.94/1864.82/1869.70 us.

463 cold cute-v197/native: 1838.98/1843.07/1839.01/1846.37 us.

463 cold trtllm/native: 1916.93/1914.86/1915.10/1916.99 us.

Each round launches independent processes in alternating compiler order, using original Graph20/100 and the pairedTRT baseline.4.6.3 loses all four cold comparisons by6.22–11.23 us; warm is slower in three rounds and essentially tied in one(0.016 us difference). Retain installed4.6.2. Static instruction totals are both1600, but4.6.3 changes control/address instructions and adds WARPSYNC instructions; dynamic source sampling below also changes total issued instructions; do not attribute the entire latency difference to one opcode.

The installed4.6.2 standalone warm measurements here span1842.22–1888.35 us. Two of four exceed their pairedTRT, including one near tie. This supplements the previously faster standalone1811.97 us record: a warm win is not uniformly reproduced. The same-process five-round audit still supports v197 over v191, but it does not justify a universal standalone warm advantage overTRT. Preserve both distributions and execution contexts.


The unlocked source control reports 603280446 issued instructions on4.6.3 versus599875543 on4.6.2 (+0.568%). Shared wavefronts remain actual=ideal28663808, excessive0. Long-scoreboard samples total69446: PV24255, producer-empty16400, QK12765. Default base/stable NCU reports128 initial registers,203112B dynamic shared memory,19.411364% active warps,46.113538% tensor activity,0.380506 eligible warps,6.758706 long-scoreboard ratio and zero local-memory sectors. Its2.827520ms diagnostic duration is above4.6.2's2.817632ms. These samples corroborate unchanged broad wait locations, not a causal decomposition of the small runtime regression.


## Iterations 203–204 — explicit uniform length across every pipeline role

Based on current defaults v190/v197. Apply make_warp_uniform to the query length in producer, issuer and compute branches, before their loops and outside every elect_one region. All active lanes in each selected warp read the same lens[qi], and role predicates are warp-uniform. This preserves variable lengths and masks; no full-length specialization is introduced. The earlier v173/v174 issuer-only control was binary-identical on a different parent. This control includes producer and compute loop bounds, motivated by compiler-sensitive scalar control/address operations. Compile first on installed4.6.2 and compare native instruction encodings/resources. If identical, record a no-op without GPU execution. If changed, require guarded qualification, full equivalence and masked audit before timing. v204 also corrects an obsolete compiler-version comment without changing allocation.


### Iterations 203–204 result — complete native-code no-op

Installed4.6.2 emits identical native instruction encodings for v190/v203 (2896 encoding words) and v197/v204 (3200 words), excluding function-name text. Resources remain123/128 registers respectively, STACK0 and LOCAL0. The compiler already infers enough uniformity for this expression in all roles. No candidate GPU execution, numerical claim or performance measurement is added. Records: artifacts/uniform_lengths.


## Iterations 205–206 — remove the final shared-reciprocal handoff

Based on defaults v190/v197. Retain the final partial-sum store, its256-thread barrier and all TMEM completion/deallocation synchronization. Each output thread then reads the four partial denominators for its own head and computes one reciprocal with the identical parenthesization and division expression. Remove the64-thread reciprocal producer, shared denominator write/read and second256-thread barrier. Keep the shared-memory allocation/layout unchanged to isolate this control.

This exchanges192 extra per-CTA reciprocals and additional shared reads for one barrier and a branch/handoff per query. It is not the earlier per-element division removed by v044: each new reciprocal is reused across all128 output elements owned by that thread. The operation is outside the16-tile attention loop, so any benefit is expected to be small. Compile/resources, bounded guarded smoke/memcheck, synccheck, full bitwise and hole/partial-tile audit precede timing; no numerical equivalence is assumed from expression similarity.


### Iterations 205–206 qualification

Both compile with unchanged123/128 registers and STACK0/LOCAL0. Guarded b2 smoke/b512 chunk0 memcheck, b2 synccheck and8192-row seed1234 exact three-repeat comparison pass. The mask fixture matches each parent bitwise; higher-precisionv206 passes the original masked tolerance. Additional full seed/short qualification is pending performance evidence.

v205 completed short paired timing1548.54us versusTRT1691.87us and NCU before an artifact-copy failure: the snapshot glob also matched a new compile-work directory. The original benchmark and NCU results are intact. Fix run_iteration.sh to copy only regular files, preserving existing snapshots and avoiding a redundant v205 rerun. v206's performance step had not started when the driver stopped.


## Warm-regime telemetry diagnostic

Add bench_with_telemetry.py, which wraps only the whole source.measure_case invocation and calls its original implementation unchanged. A separate CPU process reads only the authorized GPU1 UUID through NVML, with no CUDA context or setting changes. Cases are bracketed with monotonic host timestamps; those windows include warmup, graph capture, event setup and timed invocations, not individual GPU kernel intervals. Sampling on/off controls are required before interpreting correlations.

NVML documents PowerUsage as a one-second average on newer non-GA100 architectures; utilization likewise has an internal sampling interval. Polling every10ms does not yield10ms instantaneous power/utilization. SM-clock readings do not identify Tensor Core boost state. Retain these limitations and the uninstrumented timing records. Primary references: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html and https://docs.nvidia.com/deploy/nvml-api/api/structnvmlUtilization__t.html .


### Iterations 205–206 result — final reciprocal handoff removal is mixed

v205 short trtllm/native warm 1691.87us.

v205 short cute-v205/native warm 1548.54us.

v205 NCU base/stable: gpu__time_duration.sum=2.638368 ms, launch__registers_per_thread=123 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.294755 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=33.542172 %, smsp__warps_eligible.avg.per_cycle_active=0.368444 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.423142 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,474,239 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,822,712 .

v206 short trtllm/native warm 1691.94us.

v206 short cute-v206/native warm 1661.12us.

v206 NCU base/stable: gpu__time_duration.sum=2.816064 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.406962 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=46.179907 %, smsp__warps_eligible.avg.per_cycle_active=0.382516 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.807788 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,939,871 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,985,393 .

warm trtllm/native: 1867.81 / 1874.62 / 1878.13 / 1879.97us.

warm cute-v190/native: 1607.90 / 1608.78 / 1611.89 / 1612.00us.

warm cute-v205/native: 1610.91 / 1610.18 / 1611.62 / 1611.82us.

cold trtllm/native: 1857.60 / 1854.10 / 1859.84 / 1863.70us.

cold cute-v190/native: 1616.02 / 1615.90 / 1597.52 / 1597.60us.

cold cute-v205/native: 1595.42 / 1613.95 / 1616.18 / 1615.97us.

warm trtllm/native: 1871.97 / 1882.16 / 1880.91 / 1886.26us.

warm cute-v197/native: 1808.53 / 1812.64 / 1810.53 / 1809.66us.

warm cute-v206/native: 1807.87 / 1810.46 / 1807.71 / 1810.37us.

cold trtllm/native: 1859.39 / 1867.94 / 1865.89 / 1867.71us.

cold cute-v197/native: 1797.12 / 1801.23 / 1798.10 / 1798.43us.

cold cute-v206/native: 1810.13 / 1795.95 / 1808.42 / 1796.22us.

v205 wins two warm and two cold rounds, losing the others; the cold differences change sign and reach about20us. v206 wins three warm and two cold rounds, but its cold regressions10.32/13.01us exceed its observed cold gains2.21/5.28us. Neither establishes a uniform gain. Retain v190/v197. Qualification is limited to the full1234 and mask fixtures already recorded; no additional seed/short audit or standalone long-test promotion is claimed.


Static opcode comparison confirms one fewer BAR.SYNC.DEFER_BLOCKING in each reciprocal candidate (8→7 fast,9→8 higher precision), one fewer STS and LDS instruction, and the same four ELECT sites. Static total1448→1440 for fast and1600→1600 for higher precision. Repeating reciprocal/reduction work across more lanes changes dynamic counts; static instruction removal alone does not establish a gain.

### Warm-regime telemetry result at the original20/100 settings

Four independent sampling-on/off pairs alternate order. All512-row checks pass and the original benchmark source hash is unchanged. On/off timing is supplementary; no kernel or device settings changed. GPU1's separately queried power-management limit was1100000mW, and the installed NVML constant for SwPowerCap is4. This metadata query is read-only.

sampling=on warm trtllm/native: 1869.82 / 1874.14 / 1869.41 / 1866.00us.

sampling=on warm cute-v197/native: 1892.34 / 1886.34 / 1884.24 / 1888.29us.

sampling=on cold trtllm/native: 1912.54 / 1914.78 / 1917.25 / 1916.78us.

sampling=on cold cute-v197/native: 1834.70 / 1834.96 / 1832.98 / 1845.22us.

sampling=off warm trtllm/native: 1865.58 / 1867.78 / 1864.75 / 1861.73us.

sampling=off warm cute-v197/native: 1886.26 / 1885.81 / 1886.34 / 1873.98us.

sampling=off cold trtllm/native: 1902.64 / 1914.69 / 1918.80 / 1917.01us.

sampling=off cold cute-v197/native: 1835.02 / 1841.18 / 1835.12 / 1843.30us.

Sampling-on v197 warm1884.24–1892.34us and off1873.98–1886.34us both lose their pairedTRT in all four rounds; cold beatsTRT in every round. Sampling perturbs some numbers, so retain the off controls. During whole warm case windows (including warmup/capture/setup), TRT SM-clock medians are1912–2032MHz versusv1971710–1717MHz. All v197 warm and both cold windows report event reason4; TRT warm includes0 and4. Memory clock stays3996MHz. No thermal reason was observed in these samples. This supports differing operating states during back-to-back cases, not a per-kernel frequency or causal latency correction.

PowerUsage increases across the sequence toward the1100W configured limit; its one-second average includes preceding cases and must not be attributed to the current kernel alone. The meaning of event reason4 is verified against installedpynvml and NVIDIA's SwPowerCap documentation: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlClocksEventReasons.html .

### Follow-up methodology controls

Run original measure_case with500 warmups and100 measured calls, with alternating NVML sampling on/off, to test whether longer preconditioning changes the warm/cold result. This is a changed warmup diagnostic, not a replacement for20/100 baseline results.

The existing round-robin helper calls nvidia-smi before and after every case. Those subprocesses create inter-case idle gaps outside the GPU event interval, potentially changing subsequent power/clock state. Add an explicit endpoint-telemetry on/off option (default on preserves prior behavior), and allow the separate-process telemetry wrapper to run that helper. Compare otherwise identical rotations before attributing standalone/rotation differences to the kernel or memory placement.


### Warmup500 diagnostic result

sampling=on warm trtllm/native: 1952.61 / 1927.38 / 1925.20us.

sampling=on warm cute-v197/native: 1842.19 / 1851.49 / 1851.50us.

sampling=on cold trtllm/native: 1914.70 / 1920.70 / 1919.01us.

sampling=on cold cute-v197/native: 1832.75 / 1837.34 / 1837.28us.

sampling=off warm trtllm/native: 1933.30 / 1925.22 / 1929.28us.

sampling=off warm cute-v197/native: 1849.52 / 1851.44 / 1851.58us.

sampling=off cold trtllm/native: 1918.83 / 1918.80 / 1919.02us.

sampling=off cold cute-v197/native: 1837.18 / 1839.10 / 1836.16us.

The sampling-off warm runs now give v1971849.52–1851.58us versusTRT1925.22–1933.30us, all three wins, and retain cold wins. With sampling on, whole warm-case SM-clock medians areTRT1627–1635MHz andv1971702–1717MHz; both are predominantly power limited. The one-second power readings settle near1.09kW in later windows. This demonstrates warmup-dependent comparisons under unchanged kernel code; it does not replace the original20-warmup results or prove that all differences are clock-caused. Original script and local source copy both still hashd843320fb5147a807135282a1b87bb4c247cdd7a6a16b9107d546c48c8a8fb66.


### Endpoint-query control — inter-case gaps materially change timings

Endpoint queries on: same-cache inter-case host gaps median208.759ms, range202.790–407.253ms over32 gaps. These gaps lie outside source.measure_case and GPU events.

endpoint=on warm trtllm/native: 1870.51 / 1874.05 / 1878.05 / 1874.06 / 1882.18 / 1884.46us.

endpoint=on warm cute-v191/native: 1823.04 / 1824.46 / 1824.85 / 1824.69 / 1823.01 / 1832.05us.

endpoint=on warm cute-v197/native: 1811.54 / 1810.58 / 1808.45 / 1808.29 / 1808.58 / 1824.88us.

endpoint=on cold trtllm/native: 1859.57 / 1857.47 / 1859.60 / 1861.47 / 1858.11 / 1865.66us.

endpoint=on cold cute-v191/native: 1822.98 / 1822.59 / 1822.70 / 1822.77 / 1822.40 / 1824.77us.

endpoint=on cold cute-v197/native: 1802.90 / 1797.44 / 1798.27 / 1798.02 / 1814.53 / 1812.64us.

Endpoint queries off: same-cache inter-case host gaps median0.213ms, range0.188–0.429ms over32 gaps. These gaps lie outside source.measure_case and GPU events.

endpoint=off warm trtllm/native: 1871.87 / 1939.39 / 1937.58 / 1871.95 / 1956.00 / 1941.39us.

endpoint=off warm cute-v191/native: 1878.06 / 1859.87 / 1859.62 / 1888.24 / 1861.09 / 1859.57us.

endpoint=off warm cute-v197/native: 1851.49 / 1847.25 / 1847.71 / 1853.57 / 1848.06 / 1845.36us.

endpoint=off cold trtllm/native: 1911.06 / 1916.94 / 1919.04 / 1914.75 / 1921.09 / 1908.72us.

endpoint=off cold cute-v191/native: 1860.24 / 1858.61 / 1863.62 / 1861.66 / 1861.84 / 1858.96us.

endpoint=off cold cute-v197/native: 1830.88 / 1832.18 / 1830.77 / 1829.01 / 1830.91 / 1832.50us.

Two independent outer pairs reverse endpoint-mode order; each inner rotation uses three rounds,20 warmups and100 measured Graph calls, with the NVML sampler on in both modes. Removing nvidia-smi queries shifts v197 warm from approximately1.81ms to1.85ms and reduces its sampled SM clocks. v197 still beats v191 in all twelve warm/cold comparisons with endpoint queries off. The earlier promotion relative to v191 is supported, while the earlier absolute1.81ms rotation timings must be labelled as including query-induced inter-case gaps.

Set bench_round_robin.py endpoint-telemetry default to off for future optimization comparisons; --endpoint-telemetry on reproduces the historical helper regime. This changes only the supplementary helper, not the user's benchmark or source.measure_case. Re-evaluate the small reciprocal candidates in a five-round, five-case rotation without endpoint queries and without the NVML sampler, as the former query gaps are larger than their measured candidate differences.


## Iterations 207–208 — packed FP32 nodes of the unchanged denominator tree

Based on current defaults v190/v197. Replace the first four levels of the32-value balanced denominator tree with add_packed_f32x2(rnd="rn",ftz=False), pairing two independent adjacent tree nodes per instruction. Keep the final scalar add and running-sum FMA unchanged. Each node still adds the same two operands in the same tree position; no reassociation, input specialization or tolerance change is intended.

This requests15 packed adds plus one scalar instead of31 scalar adds. Register pairing/repacking may offset the saved ALU issue or raise pressure, so inspect actual SASS and spills before launch. The installed4.6.2 helper exposes explicit rounding/FTZ arguments. PTX8.6 introduced same-type add.f32x2 forSM100+, includingSM103: https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#floating-point-instructions-add . Guarded access/sync and exact full/masked equivalence precede timing; arithmetic source similarity is not treated as proof.


### Reciprocal candidates without endpoint queries

warm trtllm/native: 1868.00 / 1955.97 / 1957.97 / 1950.96 / 1954.11us.

warm cute-v190/native: 1642.56 / 1642.98 / 1646.88 / 1644.83 / 1644.86us.

warm cute-v205/native: 1640.69 / 1644.93 / 1644.90 / 1646.98 / 1642.86us.

warm cute-v197/native: 1857.87 / 1861.68 / 1851.81 / 1841.55 / 1852.98us.

warm cute-v206/native: 1853.73 / 1853.74 / 1849.63 / 1847.36 / 1851.66us.

cold trtllm/native: 1908.70 / 1919.04 / 1918.85 / 1921.02 / 1919.98us.

cold cute-v190/native: 1638.38 / 1636.93 / 1636.61 / 1640.24 / 1636.94us.

cold cute-v205/native: 1636.14 / 1634.35 / 1638.45 / 1635.15 / 1634.58us.

cold cute-v197/native: 1838.03 / 1840.96 / 1837.07 / 1838.24 / 1841.22us.

cold cute-v206/native: 1830.70 / 1829.98 / 1828.98 / 1830.88 / 1841.23us.

Five rounds rotate all five cases through every order position, with20 warmups/100 Graph repeats, no endpoint query and no NVML sampler. v205 wins3/5 warm and4/5 cold comparisons. v206 wins4/5 warm and4/5 cold, with the remaining cold result a0.016us near tie. Its warm loss is5.81us. The results justify expanded seed/short equivalence and original standalone timing before any default decision. They also confirm that the earlier query-gapped results do not predict all small candidate differences. No default change yet.


### Reciprocal candidates expanded equivalence and standalone timing

v205 cuda-event20/100: trtllm/native warm 1882.21us, cute-v205/native warm 1641.74us, trtllm/native cold 1914.94us, cute-v205/native cold 1641.49us.

v205 cuda-graph20/100: trtllm/native warm 1867.68us, cute-v205/native warm 1646.02us, trtllm/native cold 1916.83us, cute-v205/native cold 1640.72us.

v206 cuda-event20/100: trtllm/native warm 1880.10us, cute-v206/native warm 1812.67us, trtllm/native cold 1915.14us, cute-v206/native cold 1827.78us.

v206 cuda-graph20/100: trtllm/native warm 1869.92us, cute-v206/native warm 1865.62us, trtllm/native cold 1915.92us, cute-v206/native cold 1835.22us.

Both candidates match their parents on the second full seed5678 and1024-row chunk0/seed5678, three repeats each. Together with earlier full1234/masks they inherit the full audited precision scope. v206's standalone Graph warm1865.62us narrowly beats pairedTRT1869.92us, cold1835.22us beatsTRT1915.92us. This is encouraging relative to later v197 warm regressions, but warm timing is condition-sensitive and endpoint-off rotations retain one warm regression. Keep both as fully validated alternatives pending the new packed-sum comparison; defaults remain v190/v197.

### Packed-sum compile details

v207 reduces registers123→110 with STACK0, but total static instructions remain1448. Its scalarFADD35→5, FADD20→15 and MOV28→43 show that extra register moves can consume the arithmetic issue saving. v208 staysREG128/STACK0; scalarFADD37→7, FADD232→47, MOV38→72 and total1600→1640. Both keep four ELECT and the same MMA counts26/34. Actual timing and dynamic source counts, not the requested packed operation count, decide whether this is useful.


## Iterations 209–210 — keep independent16-leaf trees in the packed lanes

Based on current v190/v197. The previous adjacent-node pairing needs to regroup packed components at every reduction level. Instead, pair corresponding nodes from the first and last16 probabilities: packed lane0 reduces p[0:16], lane1 reduces p[16:32], each using the original balanced tree. The final scalar add joins these two roots in the original order. Each level consumes prior packed results without crossing their lanes, potentially reducing repacking moves while retaining exactly the same31 scalar additions and rounding structure.

This remains15 packed adds and one scalar add; it changes which independent nodes share an instruction, not the arithmetic dependency graph. Initial probability layout/FP8 conversion may still force moves. Compile counts/resources and guarded/full/masked qualification precede timing. No numerical or performance conclusion is inherited from the source-level tree proof alone.


### Iterations 207–208 runtime result — adjacent-node packing regresses

v207 short: trtllm/native 1691.68us, cute-v207/native 1583.20us.

v207 NCU base/stable: gpu__time_duration.sum=2.699872 ms, launch__registers_per_thread=110 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.285379 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=32.774351 %, smsp__warps_eligible.avg.per_cycle_active=0.366384 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.308334 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,228,333 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=4,502,582 .

v208 short: trtllm/native 1691.90us, cute-v208/native 1710.30us.

v208 NCU base/stable: gpu__time_duration.sum=2.910688 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.399432 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=44.707230 %, smsp__warps_eligible.avg.per_cycle_active=0.404229 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.438198 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=7,032,042 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,199,171 .

Both pass guarded b2 smoke/b512 memcheck, b2 synccheck, full1234 three-repeat exact equivalence and masked equivalence; v208 also passes the masked FP32 reference. No second full seed or short-sequence equivalence is claimed. v2071583.20us is slower than v1901549.06us; v2081710.30us is slower than v1971662.30us and pairedTRT1691.90us. NCU local traffic remains zero, but tensor activity falls and diagnostic durations rise. The register reduction alone is not a gain. Extra repacking/control issue is a hypothesis supported by static SASS, with unlocked dynamic source results recorded below. v209/v210 test persistent half-tree pairing to avoid intermediate regrouping; defaults remain unchanged.


### Packed adjacent-node dynamic source and half-tree compile

Unlocked source totals:v207549667719 issued instructions, shared actual=ideal20275200 and zero excessive; v208644309881 issued instructions, shared actual=ideal28663808 and zero excessive. v208 is7.41% above its v197 parent's599875543 instruction record. v207 long/short-scoreboard samples63337/7084 and wait12697; v20870791/3117 and wait11024. These are sampled stall counts, not fractions of total latency. The separate opcode_audit.json quantifies actual issued arithmetic and moves; no exact v190 source capture is claimed for this control.

v209 compilesREG109/STACK0,1424 static instructions, MOV29 versusv20743; v210REG128/STACK0,1616 instructions, MOV52 versusv20872. Both preserve15 new FADD2 operations and four ELECT sites. Their native code changes justify guarded/runtime qualification rather than treating this regrouping as a no-op.


## FP32 recurrence diagnostic

Add microbench_fp32_add.py to compare scalar and packed recurrences on GPU1, with1/4/8 independent pairs and32/128/256 threads in one CTA. Inline PTX brackets32768 recurrence steps with the per-SM clock counter; each pair has distinct runtime seeds, round-to-nearest additions and the same exact powers-of-two checksum. Inspect native SASS to verify scalar/packed instructions. This measures recurrence scheduling including loop, warp, timestamp and checksum overhead, not an isolated ISA latency or MLA throughput. No clock or power settings change.

This control follows the failure of packing to improve the full kernel despite fewer requested add instructions. The NVIDIA forum confirms native FADD2 emission onSM100/103 but does not establish a latency benefit: https://forums.developer.nvidia.com/t/does-blackwell-sm-120-have-native-f32x2-support/344788 . Treat forum performance speculation as unverified; use local measured recurrence results and the full-kernel NCU evidence separately.


### Iterations 209–210 result — half-tree packing remains slower than defaults

v209 short: trtllm/native 1691.78us, cute-v209/native 1579.36us.

v209 NCU base/stable: gpu__time_duration.sum=2.698752 ms, launch__registers_per_thread=109 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.295605 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=32.757114 %, smsp__warps_eligible.avg.per_cycle_active=0.356068 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.585684 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,246,758 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,507,823 .

v210 short: trtllm/native 1690.66us, cute-v210/native 1689.86us.

v210 NCU base/stable: gpu__time_duration.sum=2.873408 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.409159 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.392815 %, smsp__warps_eligible.avg.per_cycle_active=0.397025 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=6.690863 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,934,579 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,675,925 .

Guarded b2 smoke/b512 memcheck, b2 synccheck, full1234 three-repeat exact comparison and mask fixture all pass; v210 passes masked FP32. v2091579.36us improves v2071583.20us slightly but remains above v1901549.06us. v2101689.86us improves v2081710.30us but remains above v1971662.30us. Local sectors stayzero and fewer static moves do not restore default-path latency. No second-seed/short expansion or promotion for these slower candidates.


The first recurrence probe completed its checksum checks but failed when collecting the CUBIN: CuTe had captured the launch working directory at import time, before the script changed into the per-case folder. No timing result is retained from that incomplete run. Fix CLI bootstrap to normalize and enter the output directory before importing CuTe; preserve the failure log. Add256 threads to represent two active compute warps per SM partition. Re-run into a fresh artifact directory and verify native instruction types before interpreting cycle counts.


### FP32 recurrence diagnostic result — packed scheduling helps the toy loop, not the MLA tree

| Threads | Independent pairs | Scalar cycles/step | Packed cycles/step | Packed/scalar |
|---:|---:|---:|---:|---:|
| 32 | 1 | 5.0084 | 4.8752 | 0.9734 |
| 32 | 4 | 9.5092 | 9.1342 | 0.9606 |
| 32 | 8 | 17.5104 | 17.1268 | 0.9781 |
| 128 | 1 | 5.0085 | 4.8835 | 0.9750 |
| 128 | 4 | 9.5096 | 9.1346 | 0.9606 |
| 128 | 8 | 17.5116 | 17.1279 | 0.9781 |
| 256 | 1 | 6.0083 | 5.0085 | 0.8336 |
| 256 | 4 | 16.7598 | 15.7929 | 0.9423 |
| 256 | 8 | 32.7617 | 31.5765 | 0.9638 |

All18 configurations completed with exact checksums and native scalar FADD or packed FADD2 confirmed. The packed loop is faster in every measured configuration, by2.2–16.6%; most cases show2–6%. This does not support the tentative idea that packed dependent additions are intrinsically slower on this GPU. The measurement includes loop/control/timestamp/checksum overhead, uses one CTA and fixed scalar-then-packed order, and does not establish isolated instruction latency, peak throughput or the speedup of a full MLA reduction.

The full kernels still lose: adjacent-node v208 adds about35.68M dynamic MOV instructions while saving15.73M issued additions; fixed-half-tree v210 reduces static moves but remains slower than v197. Thus fewer requested arithmetic operations is insufficient. Keep the four packed-tree variants as rejected performance experiments, preserve the diagnostic raw cycles/SASS, and prioritize a different measured instruction bottleneck. Artifacts: experiments/glm53_sparse_mla/artifacts/fp32_add_probe_v2. No default change.


## Iterations 211–212 — consolidate each five-request TMA gather group

Based on v190/v197. The v197 unlocked source record issues76,144,640 UMOV instructions out of599,875,543 total. Repeated source regions copy the same four row coordinates into four additional contiguous native UR argument blocks for the four128-byte main requests and the64-byte tail. This is observed native operand preparation; the fraction of instructions is not a fraction of runtime.

Replace five separate inline-asm calls per four-row group with one asm scope containing the same five gather4 requests in the same order. Reuse named PTX destination/column variables across main requests, form the tail descriptor inside that scope, and keep tensor maps, shared layout, byte accounting, elected producer lanes, masks and barriers unchanged. PTXAS may still duplicate native argument blocks; compile and compare SASS first. No unsupported native-register constraint, changed swizzle, new warp collective, or altered arithmetic is introduced.

The official gather4 coordinate vector is {column, row0, row1, row2, row3}; SM100+ shared::cta supports this form: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor . A single PTX scope alone is not assumed to force UR reuse. Reject without runtime if native code is identical; otherwise qualify accesses/synchronization and exact outputs before timing.


### Iterations 211–212 compile result

v211 is native-encoding-identical to v190 (1448 instructions,123 registers,zero stack). v212 has1600 instructions,128 registers andzero stack; encoding changes but the targeted counts do not:217 UMOV,66 R2UR and40 static gather4 requests, all equal to v197. Consolidating the asm scope did not remove duplicated native coordinate tuples. No candidate launch or numerical/performance conclusion; the source transformation failed its intended static objective. Inspect the retained native diff to distinguish allocation/scheduling changes from tuple reuse.

## Iterations 213–214 — keep the four main requests in a real loop

Based on the corresponding v211/v212 source controls and therefore v190/v197. Use one four-iteration PTX loop for main columns0/128/256/384, updating destination by16384 bytes and column by128; the existing tail request follows. A statement-level nounroll pragma is placed at the loop header before instructions, per the official PTX rule: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#pragma-strings-nounroll . This should permit one native coordinate tuple to be reused across four requests, at the cost of address/column updates, comparison and branches. The extra loop work may offset the saved copies; static inspection and full-kernel timing decide.

The loop has a uniform constant trip count for the elected producer lane; it adds no warp collective. TMA coordinates, request count/order, shared addresses, completion bytes and barriers stay unchanged. First compile/resources, then guarded access/sync and exact full/mask comparisons before short timing if native tuple reuse appears.

The retained v212 diff changes122 native encoding words. It moves tail-descriptor preparation from loop preheader into the elected request region and shifts instruction addresses; the four-row tuple-copy pattern and targeted counts remain unchanged. This is not a binary no-op, but it does not achieve the intended operand reuse and is not promoted or timed.


### Iterations 213–214 compile and initial fixture selection

Both compile with the default CuTeDSL4.6.2,123/128 registers andzero stack. v213 static instructions1448→1352, UMOV221→133; v2141600→1512, UMOV217→133. Each has16 static gather4 sites instead of40 because eight main sites loop four times; runtime request count is unchanged. R2UR58/66 and MMA26/34 sites remain unchanged. Native loops retain coordinate URs but add destination/column copies, arithmetic, comparisons and backedges; static code reduction is not yet dynamic issue reduction.

The first b2 guarded smoke was accidentally invoked with chunk0 rather than the established chunk3 smoke fixture. It reached the numerical reference assertion and failed493/65536 elements (maxabs0.0458808). GPU1 preflight was idle; no guardrail exception was reported. Preserve that failure and explicitly compare parent v190 to v213 on this exact short chunk0 input before interpreting the failure. Continue the established chunk3 smoke and b512/chunk0 memory fixture only after this distinction is checked. Tolerance is unchanged and the short-input failure is not counted as a reference pass.

The b2/chunk0 diagnostic subsequently matches v190 exactly across all65536 elements and three repeats with TMEM guardrails enabled. Both new variants pass the established b2/chunk3 guarded smoke, b512/chunk0 guarded memcheck, b2/chunk3 synccheck and full8192/seed1234 three-repeat BF16 equivalence to their parents. The initial short-fixture reference failure remains recorded as inherited numerical behavior, not waived or relabelled a reference pass.


### Iterations 213–214 runtime — tuple reuse does not improve latency

v213 short: trtllm/native 1691.52us, cute-v213/native 1678.59us.

v213 NCU base/stable: gpu__time_duration.sum=2.710240 ms, launch__registers_per_thread=123 register/thread, launch__shared_mem_per_block_dynamic=194.920000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.399033 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=32.633187 %, smsp__warps_eligible.avg.per_cycle_active=0.391770 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=5.402664 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=4,800,358 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=3,986,413 .

v214 short: trtllm/native 1691.52us, cute-v214/native 1751.30us.

v214 NCU base/stable: gpu__time_duration.sum=2.857408 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.499890 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.576253 %, smsp__warps_eligible.avg.per_cycle_active=0.422704 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=5.694935 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=5,174,883 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=6,668,148 .

Both match their parents on full8192/seed1234 three repeats and the hole/partial-length mask fixture; v214 passes the masked FP32 reference. v2131678.59us is slower than v1901549.06us; v2141751.30us is slower than v1971662.30us and pairedTRT1691.52us. These short timings use the existing tuning protocol, not20/100 headline measurements. The NCU long-scoreboard ratio improves while time worsens, illustrating why that ratio alone is not the objective. Local-memory traffic remainszero. No default change or expansion to second-seed/short accuracy for these slower candidates.

Native v214 has eight main-gather loops with7 instructions in the first loop and6 in each remaining loop, all executing four times. The loop introduces address/column updates, comparisons and backedges and sometimes still copies destination/column operands. Dynamic source profiling will quantify the issue-count tradeoff; a smaller static kernel is not by itself less executed work.


### Looped gather dynamic instruction tradeoff

v214 issues635349655 instructions versus v197599875543, an increase of5.91%. UMOV falls76,144,640→46,260,224 (−29,884,416), but UIADD3 rises38,461,440→63,102,976 (+24,641,536), the new loop comparison issues16,777,216 times, and BRA.U rises2,334,720→19,111,936 (+16,777,216). MOV also rises8,706,722→10,437,899. Gather4 requests remain20,971,520 and MMA4,456,448. The final total includes additional changes, including sampled wait-loop execution, so it is not exactly the sum of those selected opcodes.

Source shared wavefronts remain28,663,808 actual=ideal withzero excessive. Long/short-scoreboard samples65,807/5,355 and wait16,275 are diagnostic samples, not latency fractions. The core hypothesis of tuple reuse succeeds locally, but the loop's executed control/address work more than erases the desired issue saving. Do not pursue two-way unrolling solely because the static kernel is smaller; this evidence favors retaining the unrolled default and investigating another bottleneck.


## Iterations 215–216 — packed low-term construction on the current residual pipeline

The v197 source profile issues33,554,432 HADD2.F32 instructions while reconstructing high FP8 probabilities through FP16 into scalar FP32, followed by packed FP32 subtraction and low FP8 quantization. Investigate replacing the low-term subtraction/reconstruction with packed FP16 arithmetic while retaining original FP32 exp2 and denominator accumulation. Q/KV and both PV operands stay FP8; maps, layouts and synchronization remain unchanged.

v215 retains high FP8 quantization directly from FP32 probabilities, then rounds probabilities to FP16 solely for subtraction of the exactly representable high FP8 value and converts that residual to low FP8. v216 also forms high FP8 from the rounded FP16 probabilities, porting the old v081 arithmetic as a control onto v197's optimized pipeline and64/176 role register allocation. v081 previously passed both full seeds with slightly higher error and no speed gain; its result is a warning, not a new performance or accuracy claim. The present profile provides a concrete repeated conversion cost to measure under the newer scheduling/allocation.

Both candidates change probability rounding and therefore must undergo independent FP32-reference audits; they cannot inherit v197's bitwise accuracy scope. Preserve original0.01/0.05 tolerances and separately record short/hole cases. Compile/native conversion counts first; guarded memory/sync and at least one full reference seed plus masks precede timing. If the compile or timing gain is absent, do not continue parameter sweeps solely to find a favorable result.


### Iterations 215–216 compile result

Both compile with128 registers andzero stack, reducing static instructions1600→1584. The32 scalar HADD2.F32 reconstruction sites disappear; each candidate adds16 FP32→FP16 packed conversions and uses packed half residual arithmetic. v215 retains16 high-term E4M3-from-FP32 conversions and uses16 low-term E4M3-from-FP16 conversions; v216 uses32 E4M3-from-FP16 sites. Packed FP32 FADD2 sites fall32→16. The retained native counts include all other arithmetic/control changes; this is not yet a measured dynamic instruction or latency reduction.

Original benchmark SHA256 remainsd843320fb5147a807135282a1b87bb4c247cdd7a6a16b9107d546c48c8a8fb66 in both the user's source and the local experiment copy.


### Iterations 215–216 runtime — packed residual conversion does not improve the current kernel

v215: full seed1234 passes all268435456 elements with0 mismatches, maxabs0.005918741226, relativeRMSE0.001750561591. Mask reference passes with maxabs0.004001855850. Guarded b2/b512 memory and b2 sync checks pass. No full second-seed or short-sequence accuracy claim.
Short timing: trtllm/native 1691.84us, cute-v215/native 1667.04us.
NCU base/stable: gpu__time_duration.sum=2.825632 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.408055 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=46.075676 %, smsp__warps_eligible.avg.per_cycle_active=0.369652 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=7.047772 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,940,132 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,855,601 .

v216: full seed1234 passes all268435456 elements with0 mismatches, maxabs0.005918741226, relativeRMSE0.001750537724. Mask reference passes with maxabs0.004001855850. Guarded b2/b512 memory and b2 sync checks pass. No full second-seed or short-sequence accuracy claim.
Short timing: trtllm/native 1691.74us, cute-v216/native 1673.38us.
NCU base/stable: gpu__time_duration.sum=2.840288 ms, launch__registers_per_thread=128 register/thread, launch__shared_mem_per_block_dynamic=203.112000 Kbyte/block, sm__warps_active.avg.pct_of_peak_sustained_active=19.394107 %, sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed=45.830213 %, smsp__warps_eligible.avg.per_cycle_active=0.367605 warp, smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio=7.094404 inst, l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum=0 sector, l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum=0 sector, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum=6,977,759 , l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum=2,862,024 .

v2151667.04us andv2161673.38us are slower than v1971662.30us under the existing short tuning protocol; base/stable NCU durations2.825632/2.840288ms also exceed v1972.817632ms. Eligible warps decline despite fewer static operations and zero local sectors. The first-seed relativeRMSE is slightly higher than v197, so neither accuracy nor measured short performance justifies promotion. Retain these independently checked experiments; do not claim broad reference qualification or expand testing solely to seek a favorable timing. Defaults remain v190/v197.
