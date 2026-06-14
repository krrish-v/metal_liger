// Copyright 2025 MetalLiger Contributors
// Metal Shading Language (MSL) kernel for fused RMSNorm
// Ported from: mlx/backend/metal/kernels/rms_norm.metal (Apple MIT License)
//
// This file is reserved for native Metal kernel dispatch.
// Currently, MetalLiger uses PyTorch fused ops (Approach A).
// When torch.mps.compile_shader() stabilizes, these kernels
// will be compiled and dispatched directly for maximum performance.

#include <metal_stdlib>
using namespace metal;

constant int N_READS = 4;  // Elements per thread per iteration
constant int SIMD_SIZE = 32;  // Apple GPU SIMD width

// ==
// Forward: rms_norm_forward
// ==
// Computes: output[i] = input[i] * weight[i] * rsqrt(mean(input²) + eps)
// One threadgroup per token (row). Parallel reduction for mean(x²).

template <typename T>
[[kernel]] void rms_norm_forward(
    const device T* input     [[buffer(0)]],   // [num_tokens, hidden_dim]
    const device T* weight    [[buffer(1)]],   // [hidden_dim]
    device T* output          [[buffer(2)]],   // [num_tokens, hidden_dim]
    device float* rms_cache   [[buffer(3)]],   // [num_tokens] for backward
    constant uint& hidden_dim [[buffer(4)]],
    constant float& eps       [[buffer(5)]],
    uint gid      [[threadgroup_position_in_grid]],    // token index
    uint lid      [[thread_position_in_threadgroup]],   // thread within group
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_mean[1];

    float acc = 0.0f;
    const device T* row = input + gid * size_t(hidden_dim) + lid * N_READS;

    if (lid * N_READS + N_READS <= hidden_dim) {
        for (int i = 0; i < N_READS; i++) {
            float val = float(row[i]);
            acc += val * val;
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            if (lid * N_READS + i < hidden_dim) {
                float val = float(row[i]);
                acc += val * val;
            }
        }
    }

    // Step 2: SIMD-level reduction (warp-level sum)
    acc = simd_sum(acc);

    // Step 3: Cross-SIMD reduction via threadgroup memory
    if (simd_gid == 0) {
        local_sums[simd_lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) {
        local_sums[simd_gid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_gid == 0) {
        acc = simd_sum(local_sums[simd_lid]);
        if (simd_lid == 0) {
            float inv_rms = metal::precise::rsqrt(acc / float(hidden_dim) + eps);
            local_inv_mean[0] = inv_rms;
            rms_cache[gid] = inv_rms;  // Cache for backward pass
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Normalize and scale by weight
    float inv_rms = local_inv_mean[0];
    device T* out_row = output + gid * size_t(hidden_dim) + lid * N_READS;

    if (lid * N_READS + N_READS <= hidden_dim) {
        for (int i = 0; i < N_READS; i++) {
            out_row[i] = T(float(row[i]) * inv_rms * float(weight[lid * N_READS + i]));
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            if (lid * N_READS + i < hidden_dim) {
                out_row[i] = T(float(row[i]) * inv_rms * float(weight[lid * N_READS + i]));
            }
        }
    }
}

// ============================================================================
// Backward (VJP): rms_norm_backward
// ============================================================================
// MLX strategy: outputs per-row grad_weight. Host reduces with sum(dim=0).
// NO ATOMIC CONTENTION.
//
// Math:
//   x_norm = x * rrms
//   grad_x = rrms * (g*w - x_norm * mean(g*w*x_norm))
//   grad_w[per_row] = g * x_norm

template <typename T>
[[kernel]] void rms_norm_backward(
    const device T* x         [[buffer(0)]],   // [tokens, hidden]
    const device T* weight    [[buffer(1)]],   // [hidden]
    const device T* grad_out  [[buffer(2)]],   // [tokens, hidden]
    device T* grad_x          [[buffer(3)]],   // [tokens, hidden]
    device T* grad_w          [[buffer(4)]],   // [tokens, hidden] — PER ROW!
    constant float& eps       [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    uint gid      [[threadgroup_position_in_grid]],
    uint lid      [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]])
{
    // Advance to this token's row
    const device T* x_row = x + gid * size_t(hidden_dim) + lid * N_READS;
    const device T* g_row = grad_out + gid * size_t(hidden_dim) + lid * N_READS;
    const device T* w_ptr = weight + lid * N_READS;

    // Thread-local registers
    float thread_x[N_READS];
    float thread_w[N_READS];
    float thread_g[N_READS];
    float sumx2 = 0.0f;
    float sumgwx = 0.0f;

    // Shared memory for reductions
    threadgroup float local_sumx2[SIMD_SIZE];
    threadgroup float local_sumgwx[SIMD_SIZE];
    threadgroup float local_normalizer[1];
    threadgroup float local_meangwx[1];

    // Read and accumulate
    if (lid * N_READS + N_READS <= hidden_dim) {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = float(x_row[i]);
            thread_w[i] = float(w_ptr[i]);
            thread_g[i] = float(g_row[i]);
            sumx2 += thread_x[i] * thread_x[i];
            sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            thread_x[i] = 0; thread_w[i] = 0; thread_g[i] = 0;
            if (lid * N_READS + i < hidden_dim) {
                thread_x[i] = float(x_row[i]);
                thread_w[i] = float(w_ptr[i]);
                thread_g[i] = float(g_row[i]);
                sumx2 += thread_x[i] * thread_x[i];
                sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
            }
        }
    }

    // Cross-thread reduction
    sumx2 = simd_sum(sumx2);
    sumgwx = simd_sum(sumgwx);
    if (simd_gid == 0) { local_sumx2[simd_lid] = 0; local_sumgwx[simd_lid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lid == 0) { local_sumx2[simd_gid] = sumx2; local_sumgwx[simd_gid] = sumgwx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0) {
        sumx2 = simd_sum(local_sumx2[simd_lid]);
        sumgwx = simd_sum(local_sumgwx[simd_lid]);
        if (simd_lid == 0) {
            local_meangwx[0] = sumgwx / float(hidden_dim);
            local_normalizer[0] = metal::precise::rsqrt(sumx2 / float(hidden_dim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float meangwx = local_meangwx[0];
    float normalizer = local_normalizer[0];
    float normalizer3 = normalizer * normalizer * normalizer;

    // Write gradients
    device T* gx_row = grad_x + gid * size_t(hidden_dim) + lid * N_READS;
    device T* gw_row = grad_w + gid * size_t(hidden_dim) + lid * N_READS;

    if (lid * N_READS + N_READS <= hidden_dim) {
        for (int i = 0; i < N_READS; i++) {
            gx_row[i] = T(thread_g[i] * thread_w[i] * normalizer
                          - thread_x[i] * meangwx * normalizer3);
            gw_row[i] = T(thread_g[i] * thread_x[i] * normalizer);
        }
    } else {
        for (int i = 0; i < N_READS; i++) {
            if (lid * N_READS + i < hidden_dim) {
                gx_row[i] = T(thread_g[i] * thread_w[i] * normalizer
                              - thread_x[i] * meangwx * normalizer3);
                gw_row[i] = T(thread_g[i] * thread_x[i] * normalizer);
            }
        }
    }
}

// ============================================================================
// Explicit instantiations — required for C++ extension lookup by name
// ============================================================================
// The C++ bridge calls [lib newFunctionWithName:@"rms_norm_forward_bfloat16"].
// Metal templates are not externally visible without these wrappers.

[[kernel]] void rms_norm_forward_bfloat16(
    const device bfloat* input [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]], device float* rms_cache [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]])
{ rms_norm_forward<bfloat>(input, weight, output, rms_cache, hidden_dim, eps, gid, lid, simd_lid, simd_gid); }

[[kernel]] void rms_norm_forward_float(
    const device float* input [[buffer(0)]], const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]], device float* rms_cache [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]], constant float& eps [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]])
{ rms_norm_forward<float>(input, weight, output, rms_cache, hidden_dim, eps, gid, lid, simd_lid, simd_gid); }

[[kernel]] void rms_norm_backward_bfloat16(
    const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
    const device bfloat* grad_out [[buffer(2)]], device bfloat* grad_x [[buffer(3)]],
    device bfloat* grad_w [[buffer(4)]], constant float& eps [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]])
{ rms_norm_backward<bfloat>(x, weight, grad_out, grad_x, grad_w, eps, hidden_dim, gid, lid, simd_lid, simd_gid); }

[[kernel]] void rms_norm_backward_float(
    const device float* x [[buffer(0)]], const device float* weight [[buffer(1)]],
    const device float* grad_out [[buffer(2)]], device float* grad_x [[buffer(3)]],
    device float* grad_w [[buffer(4)]], constant float& eps [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]])
{ rms_norm_backward<float>(x, weight, grad_out, grad_x, grad_w, eps, hidden_dim, gid, lid, simd_lid, simd_gid); }

{ rms_norm_backward<float>(x, weight, grad_out, grad_x, grad_w, eps, hidden_dim, gid, lid, simd_lid, simd_gid); }

