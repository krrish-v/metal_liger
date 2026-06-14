# Copyright 2025 MetalLiger Contributors
# Licensed under the Apache License 2.0
"""
MetalLiger — Fused Metal kernels for PyTorch MPS training on Apple Silicon.

Brings Liger Kernel-style operator fusion to Apple's Metal GPU backend,
eliminating intermediate tensor allocations and reducing Metal dispatch overhead
by 50%+ for transformer training.

Phases:
  Phase 3 (active):  Python autograd.Function fused ops (RMSNorm, SwiGLU, RoPE, CE)
  Phase 4a (active): torch.compile graph capture (use_metal_liger_compile=True)
  Phase 4b-e (build): Native C++ Metal dispatch + torch.library registration
                       Build: python metal_liger/setup_ext.py build_ext --inplace

Usage:
    from metal_liger import apply_metal_liger_to_qwen3vl

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-4B")
    model = apply_metal_liger_to_qwen3vl(model)
    # Training proceeds as normal — all ops are fused under the hood
"""

__version__ = "0.3.0"

from metal_liger.ops.fused_rms_norm import MetalLigerRMSNorm
from metal_liger.ops.fused_swiglu import MetalLigerSwiGLU
from metal_liger.ops.fused_rope import MetalLigerRoPE
from metal_liger.ops.fused_linear_cross_entropy import MetalLigerFusedLinearCrossEntropy
from metal_liger.ops.fused_lora import (
    apply_fused_lora_qkv,
    apply_fused_lora_mlp,
    patch_fused_lora,
)
from metal_liger.patch import apply_metal_liger_to_qwen3vl

# Phase 4a: torch.compile wrapper
from metal_liger._compile import apply_compile

# Phase 4e: Register native ops with torch.library (no-op if C++ ext not built yet)
from metal_liger._torch_library import register_metal_liger_ops
register_metal_liger_ops()

__all__ = [
    # Phase 3: Python fused ops
    "MetalLigerRMSNorm",
    "MetalLigerSwiGLU",
    "MetalLigerRoPE",
    "MetalLigerFusedLinearCrossEntropy",
    "apply_metal_liger_to_qwen3vl",
    # Phase 3.5: Fused LoRA QKV + MLP backward
    "apply_fused_lora_qkv",
    "apply_fused_lora_mlp",
    "patch_fused_lora",
    # Phase 4a
    "apply_compile",
    # Phase 4e
    "register_metal_liger_ops",
]

