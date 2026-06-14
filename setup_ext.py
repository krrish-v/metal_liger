"""Build script for MetalLiger C++ Metal extension (Phase 4b-d).

Build command (run from metal_liger/ directory):
    python setup_ext.py build_ext --inplace

This compiles rms_norm_metal.mm and swiglu_metal.mm into metal_liger_ext.so
which provides direct-to-Metal dispatch for RMSNorm and SwiGLU ops.

Requirements:
  - macOS 13+ (Metal 3)
  - Xcode Command Line Tools: xcode-select --install
  - PyTorch with MPS: pip install torch torchvision
"""

import os
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).parent
CSRC = HERE / "csrc"
KERNELS = HERE / "kernels"

# The .mm files use the Metal Obj-C API (Metal.h) and do RUNTIME shader compilation
# via [device newLibraryWithSource:...]. NO pre-compiled .metal files needed.
# Only Clang (included in CLT) is required — not the full Xcode.app metal compiler.
import subprocess

# Find the macOS SDK path (works with both CLT and full Xcode)
sdk_path_result = subprocess.run(
    ["xcrun", "--show-sdk-path"], capture_output=True, text=True
)
if sdk_path_result.returncode != 0:
    print("ERROR: Could not find macOS SDK. Run: xcode-select --install")
    sys.exit(1)

sdk_path = sdk_path_result.stdout.strip()
metal_header = os.path.join(sdk_path, "System/Library/Frameworks/Metal.framework/Headers/Metal.h")
if not os.path.exists(metal_header):
    print(f"ERROR: Metal.h not found in SDK: {sdk_path}")
    print("Try: softwareupdate --install -a")
    sys.exit(1)

print(f"SDK: {sdk_path}")
print(f"Metal.h: found ✓")

# PyTorch's MPS headers (ATen/mps/MPSStream.h etc.)
import torch
torch_include = torch.utils.cpp_extension.include_paths()

setup(
    name="metal_liger_ext",
    version="0.1.0",
    author="MetalLiger Contributors",
    description="Native Metal dispatch for fused ML kernels on Apple Silicon",
    ext_modules=[
        CppExtension(
            name="metal_liger_ext",
            sources=[
                str(CSRC / "rms_norm_metal.mm"),
                str(CSRC / "swiglu_metal.mm"),
                str(CSRC / "bindings.mm"),   # single PYBIND11_MODULE — no duplicate symbol
            ],
            include_dirs=torch_include + [str(CSRC)],
            extra_compile_args={
                # Obj-C++ with ARC (Automatic Reference Counting for Metal objects)
                "cxx": [
                    "-std=c++17",
                    "-fobjc-arc",
                    "-fobjc-weak",
                    "-fno-objc-exceptions",
                    "-w",  # suppress warnings during build
                ],
            },
            libraries=[],
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "Foundation",
                "-framework", "CoreFoundation",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
