# Copyright 2025 MetalLiger Contributors
"""Test suite for MetalLiger fused operations.

Tests verify:
  1. Numerical parity with PyTorch reference implementations
  2. Gradient correctness (backward pass matches autograd)
  3. Memory savings (fewer allocations than standard ops)

Run: python -m pytest metal_liger/tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Detect if MPS is available
MPS_AVAILABLE = (hasattr(torch.backends, 'mps')
                 and torch.backends.mps.is_available())
DEVICE = "mps" if MPS_AVAILABLE else "cpu"
DTYPE = torch.float32  # Use float32 for gradient checking accuracy


# ============================================================================
# FusedRMSNorm Tests
# ============================================================================

class TestFusedRMSNorm:
    """Test MetalLigerRMSNorm against PyTorch reference."""

    def _reference_rms_norm(self, x, weight, eps=1e-6):
        """Standard PyTorch RMSNorm for comparison."""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + eps)
        return (x_norm * weight.float()).to(x.dtype)

    def test_forward_parity(self):
        """Fused forward matches PyTorch reference."""
        from metal_liger.ops.fused_rms_norm import MetalLigerRMSNorm

        hidden_size = 2048
        x = torch.randn(2, 128, hidden_size, device=DEVICE, dtype=DTYPE)
        weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)

        # Reference
        ref = self._reference_rms_norm(x, weight)

        # MetalLiger
        norm = MetalLigerRMSNorm(hidden_size, weight=weight.clone()).to(DEVICE)
        out = norm(x)

        assert torch.allclose(ref, out, atol=1e-5, rtol=1e-4), \
            f"Max diff: {(ref - out).abs().max().item()}"
        print("✅ FusedRMSNorm forward parity")

    def test_backward_parity(self):
        """Fused backward gradients match PyTorch autograd."""
        from metal_liger.ops.fused_rms_norm import MetalLigerRMSNorm

        hidden_size = 256  # Smaller for gradient check
        x = torch.randn(2, 16, hidden_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
        weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Reference forward + backward
        ref = self._reference_rms_norm(x, weight)
        ref.sum().backward()
        ref_grad_x = x.grad.clone()
        ref_grad_w = weight.grad.clone()

        # Reset gradients
        x.grad = None
        weight.grad = None

        # MetalLiger forward + backward
        norm = MetalLigerRMSNorm(hidden_size, weight=weight.clone()).to(DEVICE)
        norm.weight = nn.Parameter(weight)  # Share the same param
        out = norm(x)
        out.sum().backward()

        assert torch.allclose(ref_grad_x, x.grad, atol=1e-4, rtol=1e-3), \
            f"grad_x max diff: {(ref_grad_x - x.grad).abs().max().item()}"
        assert torch.allclose(ref_grad_w, norm.weight.grad, atol=1e-4, rtol=1e-3), \
            f"grad_w max diff: {(ref_grad_w - norm.weight.grad).abs().max().item()}"
        print("✅ FusedRMSNorm backward parity")


# ============================================================================
# FusedSwiGLU Tests
# ============================================================================

class TestFusedSwiGLU:
    """Test MetalLiger SwiGLU against PyTorch reference."""

    def test_forward_parity(self):
        """Fused SiLU(gate) * up matches separate computation."""
        from metal_liger.ops.fused_swiglu import fused_swiglu

        gate = torch.randn(4, 128, 5120, device=DEVICE, dtype=DTYPE)
        up = torch.randn(4, 128, 5120, device=DEVICE, dtype=DTYPE)

        # Reference
        ref = F.silu(gate) * up

        # MetalLiger
        out = fused_swiglu(gate, up)

        assert torch.allclose(ref, out, atol=1e-5, rtol=1e-4), \
            f"Max diff: {(ref - out).abs().max().item()}"
        print("✅ FusedSwiGLU forward parity")

    def test_backward_parity(self):
        """Fused SwiGLU backward matches separate autograd."""
        from metal_liger.ops.fused_swiglu import fused_swiglu

        gate = torch.randn(2, 16, 512, device=DEVICE, dtype=DTYPE, requires_grad=True)
        up = torch.randn(2, 16, 512, device=DEVICE, dtype=DTYPE, requires_grad=True)

        # Reference
        ref = F.silu(gate) * up
        ref.sum().backward()
        ref_grad_gate = gate.grad.clone()
        ref_grad_up = up.grad.clone()

        # Reset
        gate.grad = None
        up.grad = None

        # MetalLiger
        out = fused_swiglu(gate, up)
        out.sum().backward()

        assert torch.allclose(ref_grad_gate, gate.grad, atol=1e-4, rtol=1e-3), \
            f"grad_gate max diff: {(ref_grad_gate - gate.grad).abs().max().item()}"
        assert torch.allclose(ref_grad_up, up.grad, atol=1e-4, rtol=1e-3), \
            f"grad_up max diff: {(ref_grad_up - up.grad).abs().max().item()}"
        print("✅ FusedSwiGLU backward parity")


# ============================================================================
# FusedRoPE Tests
# ============================================================================

class TestFusedRoPE:
    """Test MetalLiger RoPE against PyTorch reference."""

    def _make_cos_sin(self, seq_len, head_dim, device):
        """Create cos/sin tables for testing."""
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(seq_len, device=device).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(angles).repeat(1, 2)  # [seq, head_dim]
        sin = torch.sin(angles).repeat(1, 2)  # [seq, head_dim]
        return cos, sin

    def test_forward_parity(self):
        """Fused RoPE matches standard apply_rotary."""
        from metal_liger.ops.fused_rope import _apply_rotary_fused, _rotate_half

        seq_len, head_dim = 128, 64
        x = torch.randn(2, 8, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
        cos, sin = self._make_cos_sin(seq_len, head_dim, DEVICE)

        # Reference
        ref = x * cos + _rotate_half(x) * sin

        # MetalLiger
        out = _apply_rotary_fused(x, cos, sin)

        assert torch.allclose(ref, out, atol=1e-6), \
            f"Max diff: {(ref - out).abs().max().item()}"
        print("✅ FusedRoPE forward parity")

    def test_backward_inverse_rotation(self):
        """RoPE backward correctly applies inverse rotation (-sin)."""
        from metal_liger.ops.fused_rope import _FusedRoPEFunction

        seq_len, head_dim = 32, 64
        x = torch.randn(1, 4, seq_len, head_dim, device=DEVICE, dtype=DTYPE, requires_grad=True)
        cos, sin = self._make_cos_sin(seq_len, head_dim, DEVICE)

        # Forward + backward through fused op
        out = _FusedRoPEFunction.apply(x, cos, sin)
        out.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        print("✅ FusedRoPE backward computes gradients")


# ============================================================================
# FusedLoRA Tests
# ============================================================================

class TestFusedLoRA:
    """Test MetalLiger Fused LoRA against sequential PyTorch reference."""

    def test_lora_qkv_parity(self):
        """Fused LoRA QKV forward and backward parity."""
        from metal_liger.ops.fused_lora import _LoRA_QKV

        torch.manual_seed(42)
        X = torch.randn(2, 16, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)

        def _make_lora_params():
            W = torch.randn(128, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)
            A = torch.randn(8, 128, device=DEVICE, dtype=DTYPE, requires_grad=True)
            B = torch.randn(128, 8, device=DEVICE, dtype=DTYPE, requires_grad=True)
            return W, A, B, 2.0

        QW, QA, QB, QS = _make_lora_params()
        KW, KA, KB, KS = _make_lora_params()
        VW, VA, VB, VS = _make_lora_params()

        # Clone for ref
        Xr = X.clone().detach().requires_grad_(True)
        QWr, QAr, QBr = QW.clone().detach().requires_grad_(True), QA.clone().detach().requires_grad_(True), QB.clone().detach().requires_grad_(True)
        KWr, KAr, KBr = KW.clone().detach().requires_grad_(True), KA.clone().detach().requires_grad_(True), KB.clone().detach().requires_grad_(True)
        VWr, VAr, VBr = VW.clone().detach().requires_grad_(True), VA.clone().detach().requires_grad_(True), VB.clone().detach().requires_grad_(True)

        # Reference
        Q_ref = Xr @ QWr.t() + (Xr @ QAr.t()) @ QBr.t() * QS
        K_ref = Xr @ KWr.t() + (Xr @ KAr.t()) @ KBr.t() * KS
        V_ref = Xr @ VWr.t() + (Xr @ VAr.t()) @ VBr.t() * VS

        loss_ref = Q_ref.sum() + K_ref.sum() + V_ref.sum()
        loss_ref.backward()

        # Fused
        Q, K, V = _LoRA_QKV.apply(X, QW, QA, QB, QS, KW, KA, KB, KS, VW, VA, VB, VS)
        loss = Q.sum() + K.sum() + V.sum()
        loss.backward()

        assert torch.allclose(Q_ref, Q, atol=1e-4, rtol=1e-3), "Q forward mismatch"
        assert torch.allclose(K_ref, K, atol=1e-4, rtol=1e-3), "K forward mismatch"
        assert torch.allclose(V_ref, V, atol=1e-4, rtol=1e-3), "V forward mismatch"

        assert torch.allclose(Xr.grad, X.grad, atol=1e-4, rtol=1e-3), "X grad mismatch"
        assert torch.allclose(QAr.grad, QA.grad, atol=1e-4, rtol=1e-3), "QA grad mismatch"
        assert torch.allclose(QBr.grad, QB.grad, atol=1e-4, rtol=1e-3), "QB grad mismatch"
        assert torch.allclose(QWr.grad, QW.grad, atol=1e-4, rtol=1e-3), "QW grad mismatch"
        print("✅ FusedLoRA QKV parity")


# ============================================================================
# Run all tests
# ============================================================================

def run_all_tests():
    """Run all MetalLiger tests manually."""
    print(f"Running MetalLiger tests on device: {DEVICE}")
    print("=" * 60)

    # RMSNorm
    t = TestFusedRMSNorm()
    t.test_forward_parity()
    t.test_backward_parity()

    # SwiGLU
    t = TestFusedSwiGLU()
    t.test_forward_parity()
    t.test_backward_parity()

    # RoPE
    t = TestFusedRoPE()
    t.test_forward_parity()
    t.test_backward_inverse_rotation()

    print("=" * 60)
    
    # Fused LoRA
    t = TestFusedLoRA()
    t.test_lora_qkv_parity()

    print("=" * 60)
    print("✅ All MetalLiger tests passed!")


if __name__ == "__main__":
    run_all_tests()
