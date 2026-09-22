set -euo pipefail
mkdir -p /tmp/glm53_sparse_mla_dev/artifacts/dual_plane_unguarded_compile
cd /tmp/glm53_sparse_mla_dev/artifacts/dual_plane_unguarded_compile
CUDA_VISIBLE_DEVICES="" CUTE_DSL_ARCH=sm_103a PYTHONPATH=/tmp/glm53_sparse_mla_dev /opt/sglang/bin/python - <<'PY' > compile.log 2>&1
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_tensor,make_fake_stream
from probe_tmem_dual_plane import DualPlaneProbe
specs=[(cutlass.Float8E4M3FN,(64,32),(32,1)),(cutlass.Float8E4M3FN,(512,32),(32,1)),(cutlass.Float32,(1,64,512),(32768,512,1)),(cutlass.Float32,(1,64,64),(4096,64,1))]
args=[make_fake_tensor(dtype,shape,stride,assumed_align=16) for dtype,shape,stride in specs]
cute.compile(DualPlaneProbe(),*args,make_fake_stream(),options='--gpu-arch sm_103a --keep-cubin --keep-ptx')
print('UNGUARDED OFFLINE COMPILE ONLY; NO LAUNCH')
PY
cubin=( *.cubin )
cuobjdump --dump-resource-usage "${cubin[0]}" > resources.txt
cuobjdump --dump-sass "${cubin[0]}" > sass.txt
cat resources.txt

set -euo pipefail
cd /tmp/glm53_sparse_mla_dev
export CUDA_VISIBLE_DEVICES=GPU-2dc4b50c-07a5-26d6-f5ce-54ef728d56b2
export CUTE_DSL_ARCH=sm_103a
/opt/sglang/bin/python check_gpu_idle.py > artifacts/dual_plane_extended_preflight.log 2>&1
timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_dual_plane.py --blocks 1 --output-dir artifacts/dual_plane_synccheck > artifacts/dual_plane_synccheck.log 2>&1
tail -1 artifacts/dual_plane_synccheck.log
timeout --kill-after=10s 180s compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_dual_plane.py --blocks 296 --output-dir artifacts/dual_plane_memcheck296 > artifacts/dual_plane_memcheck296.log 2>&1
tail -1 artifacts/dual_plane_memcheck296.log
timeout --kill-after=10s 180s compute-sanitizer --tool synccheck --error-exitcode 86 /opt/sglang/bin/python -u probe_tmem_dual_plane.py --blocks 296 --output-dir artifacts/dual_plane_synccheck296 > artifacts/dual_plane_synccheck296.log 2>&1
tail -1 artifacts/dual_plane_synccheck296.log
mkdir -p artifacts/snapshots/dual_plane_probe
for folder in dual_plane_compile dual_plane_unguarded_compile dual_plane_memcheck dual_plane_synccheck dual_plane_memcheck296 dual_plane_synccheck296; do
 cp -a "artifacts/$folder" artifacts/snapshots/dual_plane_probe/
done
cp artifacts/dual_plane_*.log artifacts/snapshots/dual_plane_probe/
