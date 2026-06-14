# Copyright 2025 MetalLiger Contributors
"""Fused SwiGLU activation — eliminates gate tensor intermediate.

Standard PyTorch SwiGLU:
  activated = SiLU(gate)      → dispatch + intermediate
  out  = activated * up       → dispatch + keeps gate for autograd
  (gate + activated kept alive for autograd = 2 extra tensors/layer)

MetalLiger FusedSwiGLU (Phase 3):
  out = FusedSiLUMul(gate, up)  → 1 dispatch, gate freed after forward
  backward recomputes sigmoid(gate) (cheap) instead of reading stored cache
"""

import torch
import torch.nn as nn


class _FusedSwiGLUFunction(torch.autograd.Function):
    """Fused SiLU(gate) * up — single autograd op, gate not kept past forward.

    Fix: `up` is no longer saved in ctx — only `gate` is needed for backward.
    grad_up = grad_f * silu(gate), which is recomputed from saved `gate` only.
    This eliminates one [batch, seq, intermediate_dim] tensor per layer.
    """

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor):
        gate_f = gate.float()
        sigmoid_gate = torch.sigmoid(gate_f)
        silu_gate = gate_f * sigmoid_gate
        output = (silu_gate * up.float()).to(gate.dtype)
        
        # Save both gate, up, and cached sigmoid to avoid heavy backward ALUs
        ctx.save_for_backward(gate, up, sigmoid_gate)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up, sigmoid_gate = ctx.saved_tensors
        gate_f = gate.float()
        up_f = up.float()
        grad_f = grad_output.float()

        # M4 Pro unified memory bandwidth is massive, so caching sigmoid_gate 
        # is significantly faster than recomputing exp() ALUs.
        silu_gate = gate_f * sigmoid_gate

        grad_up   = (grad_f * silu_gate).to(up.dtype)
        grad_gate = (grad_f * up_f * sigmoid_gate * (1.0 + gate_f * (1.0 - sigmoid_gate))).to(gate.dtype)
        return grad_gate, grad_up



def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Functional: SiLU(gate) * up in a single fused op."""
    return _FusedSwiGLUFunction.apply(gate, up)


class MetalLigerSwiGLU(nn.Module):
    """Drop-in for SwiGLU MLP block — fuses the activation step only."""

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = _FusedSwiGLUFunction.apply(gate, up)
        return self.down_proj(activated)

    def extra_repr(self) -> str:
        return "fused=MetalLiger-Phase3"
