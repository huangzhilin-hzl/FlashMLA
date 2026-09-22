#!/usr/bin/env bash
# Process-local compiler selection; never install into the shared Python env.
set -euo pipefail
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
compiler_prefix="$experiment_dir/compiler_envs/cutlass471"
[[ -d "$compiler_prefix/nvidia_cutlass_dsl/dsl_packages" ]] || {
  echo "Missing isolated compiler: $compiler_prefix" >&2
  exit 2
}
export PYTHONPATH="$compiler_prefix:$compiler_prefix/nvidia_cutlass_dsl/dsl_packages${PYTHONPATH:+:$PYTHONPATH}"
/opt/sglang/bin/python - "$compiler_prefix" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
import cutlass
from cutlass._mlir._mlir_libs import _cutlass_ir

prefix = Path(sys.argv[1]).resolve()
for package in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-core", "nvidia-cutlass-dsl-libs-cu12",
                "nvidia-cutlass-dsl-libs-cu13"):
    assert metadata.version(package) == "4.7.1", (package, metadata.version(package))
assert cutlass.__version__ == "4.7.1", cutlass.__version__
for module in (cutlass, _cutlass_ir):
    assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__
PY
exec "$@"
