"""Phase 4e: Register MetalLiger native ops as first-class torch.library operators.

This allows torch.compile to trace THROUGH the custom ops rather than treating
them as opaque black boxes. Without this, torch.compile stops at the boundary of
custom ops and creates a graph break — defeating the purpose of compilation.

With torch.library registration:
  - torch.compile traces the full model including MetalLiger ops
  - The compiled graph is a single flat sequence of Metal commands
  - No Python re-entry anywhere in the forward+backward path

Usage: imported automatically by _native.py when metal_liger_ext is available.
"""

import logging
from typing import List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_metal_liger_ops():
    """Register MetalLiger native ops with torch.library.

    Safe to call multiple times — skips if already registered.
    Only runs if metal_liger_ext (the C++ extension) is importable.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        import metal_liger_ext  # noqa: F401 — just check it's importable
    except ImportError:
        logger.debug("Phase 4e: metal_liger_ext not built yet. Skipping torch.library registration.")
        return

    try:
        # ── RMSNorm ────────────────────────────────────────────────────────────
        torch.library.define(
            "metal_liger::rms_norm_forward",
            "(Tensor x, Tensor weight, float eps, str kernel_dir) -> (Tensor, Tensor)",
            lib=torch.library.Library("metal_liger", "DEF"),
        )

        @torch.library.impl("metal_liger::rms_norm_forward", "mps")
        def _rms_norm_forward_impl(x: Tensor, weight: Tensor, eps: float, kernel_dir: str):
            import metal_liger_ext
            return metal_liger_ext.rms_norm_forward(x, weight, eps, kernel_dir)

        @torch.library.register_fake("metal_liger::rms_norm_forward")
        def _rms_norm_forward_abstract(x: Tensor, weight: Tensor, eps: float, kernel_dir: str):
            # Tells torch.compile what shapes/dtypes to expect — enables full graph tracing
            output    = torch.empty_like(x)
            rms_cache = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
            return output, rms_cache

        # ── SwiGLU ─────────────────────────────────────────────────────────────
        torch.library.define(
            "metal_liger::swiglu_forward",
            "(Tensor gate, Tensor up, str kernel_dir) -> Tensor",
            lib=torch.library.Library("metal_liger", "DEF"),
        )

        @torch.library.impl("metal_liger::swiglu_forward", "mps")
        def _swiglu_forward_impl(gate: Tensor, up: Tensor, kernel_dir: str):
            import metal_liger_ext
            return metal_liger_ext.swiglu_forward(gate, up, kernel_dir)

        @torch.library.register_fake("metal_liger::swiglu_forward")
        def _swiglu_forward_abstract(gate: Tensor, up: Tensor, kernel_dir: str):
            return torch.empty_like(gate)

        _REGISTERED = True
        logger.info(
            "Phase 4e: Registered metal_liger::rms_norm_forward and metal_liger::swiglu_forward "
            "with torch.library. torch.compile will now trace through these ops."
        )

    except Exception as e:
        logger.warning(f"Phase 4e: torch.library registration failed — {e}. "
                       "torch.compile will treat MetalLiger ops as graph breaks.")
