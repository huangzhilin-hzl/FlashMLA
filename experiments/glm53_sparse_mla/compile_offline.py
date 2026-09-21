"""Compile a TMA candidate for SM103 without allocating or launching GPU work.

Run with CUDA_VISIBLE_DEVICES="" and CUTE_DSL_ARCH=sm_103a. Fake descriptors
match the full benchmark shapes, strides and pointer alignment. Compilation
is not a numerical, memory-safety or performance validation.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor


parser = argparse.ArgumentParser()
parser.add_argument("--kernel-version", required=True)
args = parser.parse_args()
module = importlib.import_module(f"kernel_{args.kernel_version}")
descriptors = [
    (cutlass.Float8E4M3FN, (8192, 64, 576), (36864, 576, 1)),
    (cutlass.Float8E4M3FN, (131072, 576), (576, 1)),
    (cutlass.Int32, (8192, 2048), (2048, 1)),
    (cutlass.Int32, (8192,), (1,)),
    (cutlass.BFloat16, (8192, 64, 512), (32768, 512, 1)),
    (cutlass.Uint8, (getattr(module, "TENSOR_MAP_BYTES", 256),), (1,)),
]
tensors = [make_fake_tensor(dtype, shape, stride, assumed_align=16)
           for dtype, shape, stride in descriptors]
print(json.dumps({"version": args.kernel_version, "device_launch": False,
                  "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}), flush=True)
compiled = cute.compile(module.SparseMLA(128), *tensors, make_fake_stream(),
                        options="--gpu-arch sm_103a --keep-cubin --keep-ptx --ptxas-options -v")
print("OFFLINE COMPILE PASS", flush=True)
