set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
mkdir -p artifacts/query_overlap_timeline/guarded
/opt/sglang/bin/python check_gpu_idle.py > artifacts/query_overlap_timeline/guarded/preflight.log 2>&1
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 180s /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 2 --repeat 1 --output-dir artifacts/query_overlap_timeline/guarded/smoke > artifacts/query_overlap_timeline/guarded/smoke.log 2>&1
tail -1 artifacts/query_overlap_timeline/guarded/smoke.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 300s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 512 --chunk 0 --repeat 1 --output-dir artifacts/query_overlap_timeline/guarded/memcheck > artifacts/query_overlap_timeline/guarded/memcheck.log 2>&1
tail -1 artifacts/query_overlap_timeline/guarded/memcheck.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 300s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 2 --repeat 3 --output-dir artifacts/query_overlap_timeline/guarded/synccheck_fixed > artifacts/query_overlap_timeline/guarded/synccheck_fixed.log 2>&1
tail -1 artifacts/query_overlap_timeline/guarded/synccheck_fixed.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 300s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 512 --chunk 0 --repeat 3 --output-dir artifacts/query_overlap_timeline/guarded/synccheck_varlen > artifacts/query_overlap_timeline/guarded/synccheck_varlen.log 2>&1
tail -1 artifacts/query_overlap_timeline/guarded/synccheck_varlen.log
MLA_TMEM_GUARDRAILS=1 timeout --kill-after=10s 300s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_query_overlap.py --local-tokens 513 --chunk 0 --repeat 3 --output-dir artifacts/query_overlap_timeline/guarded/synccheck_odd > artifacts/query_overlap_timeline/guarded/synccheck_odd.log 2>&1
tail -1 artifacts/query_overlap_timeline/guarded/synccheck_odd.log

