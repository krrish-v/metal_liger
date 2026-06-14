// Copyright 2025 MetalLiger Contributors
// csrc/metal_liger_ops.h — Forward declarations for all native Metal ops
//
// Included by bindings.mm only. Each .mm file implements the functions
// declared here, but NONE of them define PYBIND11_MODULE (only bindings.mm does).

#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

// ─── RMSNorm (Phase 4b/c) ────────────────────────────────────────────────────
std::vector<torch::Tensor> metal_rms_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps,
    const std::string& kernel_dir);

std::vector<torch::Tensor> metal_rms_norm_backward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor grad_out,
    double eps,
    const std::string& kernel_dir);

// ─── SwiGLU (Phase 4d) ───────────────────────────────────────────────────────
torch::Tensor metal_swiglu_forward(
    torch::Tensor gate,
    torch::Tensor up,
    const std::string& kernel_dir);

std::vector<torch::Tensor> metal_swiglu_backward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor grad_out,
    const std::string& kernel_dir);
