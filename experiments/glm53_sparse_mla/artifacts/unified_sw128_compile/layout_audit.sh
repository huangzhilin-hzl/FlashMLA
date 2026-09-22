set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
CUDA_VISIBLE_DEVICES="" CUTE_DSL_ARCH=sm_103a /opt/sglang/bin/python audit_unified_sw128_layout.py > artifacts/unified_sw128_compile/layout_audit.log 2>&1
cat artifacts/unified_sw128_compile/layout_audit.log
