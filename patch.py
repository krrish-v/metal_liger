# Copyright 2025 MetalLiger Contributors
"""Monkey-patching system for HuggingFace models.

Replaces standard model operations with MetalLiger fused equivalents.
Designed for minimal API surface — one function call to fuse everything.

Usage:
    from metal_liger import apply_metal_liger_to_qwen3vl
    model = apply_metal_liger_to_qwen3vl(model)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from metal_liger.ops.fused_rms_norm import MetalLigerRMSNorm
from metal_liger.ops.fused_swiglu import MetalLigerSwiGLU
from metal_liger.ops.fused_rope import apply_fused_rotary
from metal_liger.ops.fused_lora import patch_fused_lora, apply_fused_lora_qkv, apply_fused_lora_mlp
from metal_liger.generation import patch_generator

logger = logging.getLogger(__name__)


def _find_layers(model: nn.Module):
    """Find transformer decoder layers regardless of model wrapping.

    Handles:
      - Bare model: model.model.layers (Qwen3VLForConditionalGeneration)
      - PEFT-wrapped: model.base_model.model.model.layers (PeftModel → LoraModel → Qwen3VL)
      - Other nesting: walks up to 5 levels deep looking for a 'layers' attribute
    """
    # All possible paths to transformer layers, ordered most-specific first
    candidates = [
        # PEFT + Qwen3-VL: peft_model.base_model.model.model.language_model.layers
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'layers', None),
        
        # PEFT wrapped (LoRA): PeftModel.base_model.model.model.layers
        lambda m: getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'layers', None),
        
        # Qwen3-VL nested architecture: model.model.language_model.layers
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'layers', None),
        
        # Standard HF: model.model.layers
        lambda m: getattr(getattr(m, 'model', None), 'layers', None),
        
        # Direct: model.layers
        lambda m: getattr(m, 'layers', None),
        
        # Accelerate wrapped
        lambda m: getattr(getattr(getattr(m, 'module', None), 'model', None), 'layers', None),
    ]

    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue

    return None


def _find_final_norm(model: nn.Module):
    """Find the final RMSNorm (model.model.norm or deeper)."""
    candidates = [
        # PEFT + Qwen3-VL
        lambda m: getattr(getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'language_model', None), 'norm', None),
        
        # Qwen3-VL nested architecture
        lambda m: getattr(getattr(getattr(m, 'model', None), 'language_model', None), 'norm', None),
        
        # Standard PEFT
        lambda m: getattr(getattr(getattr(getattr(m, 'base_model', None), 'model', None), 'model', None), 'norm', None),
        
        # Standard HF
        lambda m: getattr(getattr(m, 'model', None), 'norm', None),
    ]
    for getter in candidates:
        try:
            norm = getter(model)
            if norm is not None and hasattr(norm, 'weight'):
                return norm, getter
        except (AttributeError, TypeError):
            continue
    return None, None


def _replace_rms_norms(model: nn.Module) -> int:
    """Replace all RMSNorm layers with MetalLiger fused versions.

    Returns the number of layers replaced.
    """
    count = 0

    layers = _find_layers(model)
    if layers is None:
        logger.warning("MetalLiger: could not find transformer layers for RMSNorm patching")
        return 0

    for layer in layers:
        # Input layernorm
        if hasattr(layer, 'input_layernorm'):
            old_norm = layer.input_layernorm
            eps = getattr(old_norm, 'variance_epsilon',
                          getattr(old_norm, 'eps', 1e-6))
            hidden_size = old_norm.weight.shape[0]
            new_norm = MetalLigerRMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                weight=old_norm.weight.data,
            )
            layer.input_layernorm = new_norm
            count += 1

        # Post-attention layernorm
        if hasattr(layer, 'post_attention_layernorm'):
            old_norm = layer.post_attention_layernorm
            eps = getattr(old_norm, 'variance_epsilon',
                          getattr(old_norm, 'eps', 1e-6))
            hidden_size = old_norm.weight.shape[0]
            new_norm = MetalLigerRMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                weight=old_norm.weight.data,
            )
            layer.post_attention_layernorm = new_norm
            count += 1

    # Final norm (handles PEFT wrapping)
    old_norm, _ = _find_final_norm(model)
    if old_norm is not None:
        eps = getattr(old_norm, 'variance_epsilon',
                      getattr(old_norm, 'eps', 1e-6))
        hidden_size = old_norm.weight.shape[0]
        new_norm = MetalLigerRMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            weight=old_norm.weight.data,
        )
        # Set the norm on the correct parent
        # Try Qwen3VL paths first, then PEFT path, then standard
        try:
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'language_model'):
                 model.base_model.model.model.language_model.norm = new_norm
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                model.base_model.model.model.norm = new_norm
            elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
                model.model.language_model.norm = new_norm
            elif hasattr(model, 'model'):
                model.model.norm = new_norm
            count += 1
        except AttributeError as e:
            logger.debug(f"MetalLiger: could not replace final norm. {e}")

    return count


def _replace_mlps(model: nn.Module) -> int:
    """Replace SwiGLU MLPs with MetalLiger fused versions.

    Returns the number of layers replaced.
    """
    count = 0
    layers = _find_layers(model)
    if layers is None:
        return 0

    for layer in layers:
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            original_mlp = layer.mlp
            layer.mlp = MetalLigerSwiGLU(original_mlp)
            count += 1

    return count


def apply_metal_liger_to_qwen3vl(
    model: nn.Module,
    fuse_rms_norm: bool = True,
    fuse_swiglu: bool = True,
    fuse_rope: bool = False,
    fuse_cross_entropy: bool = False,
    fuse_lora: bool = True,   # Phase 3.5: fused QKV+MLP LoRA backward
    use_optimized_generator: bool = False, # DISABLED: breaks GRPO sampling (no do_sample/temperature/top_p)
) -> nn.Module:
    """Apply MetalLiger fused operations to a Qwen3-VL model.

    This is the main entry point. Call once after loading the model.

    Args:
        model: A Qwen3-VL model (or similar architecture)
        fuse_rms_norm: Replace RMSNorm with fused version (default: True)
        fuse_swiglu: Replace SwiGLU MLP with fused version (default: True)
        fuse_rope: Replace RoPE with fused version (default: False — experimental)
        fuse_cross_entropy: Whether to use fused CE (applied in trainer)

    Returns:
        The same model with fused operations applied in-place
    """
    stats = {}

    if fuse_rms_norm:
        n = _replace_rms_norms(model)
        stats['rms_norm'] = n
        logger.info(f"MetalLiger: replaced {n} RMSNorm layers with fused versions")

    if fuse_swiglu:
        n = _replace_mlps(model)
        stats['swiglu'] = n
        logger.info(f"MetalLiger: replaced {n} MLP layers with fused SwiGLU")

    if fuse_lora:
        n = patch_fused_lora(model)
        stats['fused_lora'] = n
        logger.info(f"MetalLiger: fused LoRA QKV+MLP marked {n} modules (Phase 3.5)")

    if use_optimized_generator:
        patch_generator(model)
        stats['generator'] = 'optimized'
        logger.info("MetalLiger: applied Phase 5 optimized MPS generator")

    if fuse_rope:
        logger.info("MetalLiger: fused RoPE is experimental — use apply_fused_rotary()")
        stats['rope'] = 'available'

    if fuse_cross_entropy:
        logger.info("MetalLiger: fused CE — use MetalLigerFusedLinearCrossEntropy in trainer")
        stats['cross_entropy'] = 'available'

    total = sum(v for v in stats.values() if isinstance(v, int))
    logger.info(
        f"MetalLiger: {total} total layers patched. "
        f"Details: {stats}"
    )

    return model
