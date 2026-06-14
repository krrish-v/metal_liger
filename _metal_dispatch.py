# Copyright 2025 MetalLiger Contributors
"""Metal kernel compilation, caching, and dispatch.

This module provides the bridge between Python and Metal Shading Language (MSL)
kernels. It handles:
  1. Loading .metal source files from the kernels/ directory
  2. Runtime compilation via Metal's newLibraryWithSource API
  3. Caching compiled pipelines to avoid re-compilation
  4. Dispatching compute commands to the MPS device

Architecture:
  Currently uses pure PyTorch ops for fusion (Approach A).
  When torch.mps.compile_shader() API stabilizes, this module will
  switch to direct Metal kernel dispatch (Approach B) for maximum perf.

The key insight: even without custom Metal kernels, wrapping multiple ops
in a single torch.autograd.Function eliminates intermediate tensor storage
because PyTorch only saves what ctx.save_for_backward explicitly keeps.
"""

import os
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Directory containing .metal kernel source files
KERNEL_DIR = Path(__file__).parent / "kernels"


def load_metal_source(kernel_name: str) -> str:
    """Load a .metal source file from the kernels directory.

    Args:
        kernel_name: Name without extension, e.g., 'rms_norm'

    Returns:
        MSL source code as string
    """
    path = KERNEL_DIR / f"{kernel_name}.metal"
    if not path.exists():
        raise FileNotFoundError(f"Metal kernel not found: {path}")
    return path.read_text()


class MetalKernelRegistry:
    """Registry for compiled Metal kernels.

    Handles lazy compilation and caching of Metal compute pipelines.
    Currently stores source code for future use when torch.mps.compile_shader
    stabilizes — actual computation uses PyTorch fused ops.
    """

    _instance = None
    _kernels: dict = {}

    @classmethod
    def get(cls) -> "MetalKernelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, source: str):
        """Register a Metal kernel source for future compilation."""
        self._kernels[name] = {
            "source": source,
            "pipeline": None,  # Lazy compiled
        }
        logger.debug(f"MetalLiger: registered kernel '{name}'")

    def has_native_dispatch(self) -> bool:
        """Check if native Metal dispatch is available.

        Returns True when torch.mps.compile_shader() is stable and usable.
        Until then, we fall back to fused PyTorch ops.
        """
        # TODO: Enable when PyTorch MPS compile_shader API is stable
        # return hasattr(torch.mps, 'compile_shader')
        return False

    def list_kernels(self) -> list:
        """List all registered kernel names."""
        return list(self._kernels.keys())


def ensure_mps_device(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is on MPS device."""
    if tensor.device.type != "mps":
        raise ValueError(
            f"MetalLiger requires MPS tensors, got {tensor.device}. "
            f"Move your model to MPS: model.to('mps')"
        )
    return tensor


def check_contiguous(*tensors: torch.Tensor):
    """Verify all tensors are contiguous for Metal kernel compatibility."""
    for i, t in enumerate(tensors):
        if not t.is_contiguous():
            raise ValueError(
                f"MetalLiger: tensor {i} is not contiguous. "
                f"Call .contiguous() before passing to fused ops."
            )
