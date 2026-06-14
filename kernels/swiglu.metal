// Copyright 2025 MetalLiger Contributors
// swiglu.metal — Fused SwiGLU kernel
//
// Forward:  output[i] = SiLU(gate[i]) * up[i]
//           SiLU(x)   = x * sigmoid(x) = x / (1 + exp(-x))
//
// Backward (recomputes sigmoid, no stored cache):
//   sig = sigmoid(gate)
//   grad_gate[i] = grad_out[i] * up[i] * sig * (1 + gate[i] * (1 - sig))
//   grad_up[i]   = grad_out[i] * SiLU(gate[i])

#include <metal_stdlib>
using namespace metal;

// ─── Forward ─────────────────────────────────────────────────────────────────

template <typename T>
[[kernel]] void swiglu_forward(
    const device T* gate     [[buffer(0)]],
    const device T* up       [[buffer(1)]],
    device T*       output   [[buffer(2)]],
    constant uint&  n_elems  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_elems) return;
    float g   = float(gate[gid]);
    float u   = float(up[gid]);
    float sig = 1.0f / (1.0f + metal::exp(-g));
    output[gid] = T(g * sig * u);   // SiLU(gate) * up
}

// ─── Backward ────────────────────────────────────────────────────────────────

template <typename T>
[[kernel]] void swiglu_backward(
    const device T* gate       [[buffer(0)]],
    const device T* up         [[buffer(1)]],
    const device T* grad_out   [[buffer(2)]],
    device T*       grad_gate  [[buffer(3)]],
    device T*       grad_up    [[buffer(4)]],
    constant uint&  n_elems    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_elems) return;
    float g   = float(gate[gid]);
    float u   = float(up[gid]);
    float go  = float(grad_out[gid]);

    // Recompute sigmoid — cheaper than a cache read on unified memory
    float sig  = 1.0f / (1.0f + metal::exp(-g));
    float silu = g * sig;

    // d(SiLU(g) * u)/dg = u * sig * (1 + g*(1-sig))
    grad_gate[gid] = T(go * u * sig * (1.0f + g * (1.0f - sig)));
    grad_up[gid]   = T(go * silu);
}

// ─── Explicit instantiations ──────────────────────────────────────────────────

[[kernel]] void swiglu_forward_bfloat16(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
    device bfloat* output [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{ swiglu_forward<bfloat>(gate, up, output, n, gid); }

[[kernel]] void swiglu_forward_float(
    const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{ swiglu_forward<float>(gate, up, output, n, gid); }

[[kernel]] void swiglu_backward_bfloat16(
    const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
    const device bfloat* grad_out [[buffer(2)]],
    device bfloat* grad_gate [[buffer(3)]], device bfloat* grad_up [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{ swiglu_backward<bfloat>(gate, up, grad_out, grad_gate, grad_up, n, gid); }

[[kernel]] void swiglu_backward_float(
    const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
    const device float* grad_out [[buffer(2)]],
    device float* grad_gate [[buffer(3)]], device float* grad_up [[buffer(4)]],
    constant uint& n [[buffer(5)]], uint gid [[thread_position_in_grid]])
{ swiglu_backward<float>(gate, up, grad_out, grad_gate, grad_up, n, gid); }
