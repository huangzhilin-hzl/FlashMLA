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

## Iteration 034 — two compute warpgroups per tile (pending)

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
