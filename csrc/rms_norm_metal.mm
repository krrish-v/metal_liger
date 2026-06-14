// Copyright 2025 MetalLiger Contributors
// Phase 4b: Native Metal dispatch for fused RMSNorm — Forward + Backward
//
// This replaces the Python autograd.Function in fused_rms_norm.py.
// Metal kernels fire directly from C++ → zero Python re-entry per op.
//
// Architecture:
//   Python call → C++ (this file) → MTLComputePipelineState → GPU
//   ↑ one boundary crossing         ↑ single command buffer submit
//
// The key trick: we reuse PyTorch's own MPS command buffer via:
//   at::mps::getCurrentMPSStream()->commandBuffer()
// This means our kernel runs IN SEQUENCE with all other PyTorch MPS ops,
// with no extra synchronization needed.

#include "metal_liger_ops.h"
#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>

#include <fstream>
#include <string>
#include <stdexcept>


// ─── MTLBuffer helper ────────────────────────────────────────────────────────
// PyTorch 2.x MPS does not expose getMTLBufferStorage() publicly.
// The correct approach: cast storage data pointer (which IS the MTLBuffer ptr)
// using __bridge (needed for ARC — does NOT transfer ownership).
#define MPS_BUF(t)  ((__bridge id<MTLBuffer>)(t).storage().data())
#define MPS_OFS(t)  ((NSUInteger)((t).storage_offset() * (t).element_size()))


// ─── Pipeline cache ──────────────────────────────────────────────────────────
// Compiled once on first call, reused every subsequent step.
// Thread-safety: training is single-threaded on Mac, but we use a static
// initializer anyway for safety.

struct PipelineCache {
    id<MTLComputePipelineState> fwd_bf16 = nil;
    id<MTLComputePipelineState> bwd_bf16 = nil;
    id<MTLComputePipelineState> fwd_f32  = nil;
    id<MTLComputePipelineState> bwd_f32  = nil;
    bool compiled = false;
};

static PipelineCache g_rms_cache;

static NSString* load_metal_source(const std::string& metal_path) {
    NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) {
        throw std::runtime_error(
            std::string("MetalLiger: failed to load kernel source: ") +
            metal_path + " — " + [[err localizedDescription] UTF8String]
        );
    }
    return src;
}

static void ensure_compiled(const std::string& kernel_dir) {
    if (g_rms_cache.compiled) return;

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    if (!device) throw std::runtime_error("MetalLiger: could not get MTLDevice");

    NSString* src = load_metal_source(kernel_dir + "/rms_norm.metal");
    MTLCompileOptions* opts = [MTLCompileOptions new];

    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        throw std::runtime_error(
            std::string("MetalLiger: kernel compile failed: ") +
            [[err localizedDescription] UTF8String]
        );
    }

    // Forward kernels
    id<MTLFunction> fwd_bf16_fn = [lib newFunctionWithName:@"rms_norm_forward_bfloat16"];
    id<MTLFunction> fwd_f32_fn  = [lib newFunctionWithName:@"rms_norm_forward_float"];

    // Backward kernels
    id<MTLFunction> bwd_bf16_fn = [lib newFunctionWithName:@"rms_norm_backward_bfloat16"];
    id<MTLFunction> bwd_f32_fn  = [lib newFunctionWithName:@"rms_norm_backward_float"];

    if (!fwd_bf16_fn || !bwd_bf16_fn) {
        throw std::runtime_error(
            "MetalLiger: could not find rms_norm_forward_bfloat16 or rms_norm_backward_bfloat16. "
            "Ensure the .metal file has explicit template instantiations."
        );
    }

    g_rms_cache.fwd_bf16 = [device newComputePipelineStateWithFunction:fwd_bf16_fn error:&err];
    g_rms_cache.bwd_bf16 = [device newComputePipelineStateWithFunction:bwd_bf16_fn error:&err];
    if (fwd_f32_fn)
        g_rms_cache.fwd_f32 = [device newComputePipelineStateWithFunction:fwd_f32_fn error:&err];
    if (bwd_f32_fn)
        g_rms_cache.bwd_f32 = [device newComputePipelineStateWithFunction:bwd_f32_fn error:&err];

    g_rms_cache.compiled = true;
}

// ─── Forward ─────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> metal_rms_norm_forward(
    torch::Tensor x,          // [tokens, hidden] bfloat16/float32 on MPS
    torch::Tensor weight,     // [hidden]
    double eps,
    const std::string& kernel_dir)
{
    TORCH_CHECK(x.device().is_mps(), "MetalLiger RMSNorm: input must be on MPS");
    TORCH_CHECK(x.dim() >= 2, "MetalLiger RMSNorm: input must be ≥2D");

    ensure_compiled(kernel_dir);

    // Flatten to 2D: [tokens, hidden]
    auto x_2d = x.view({-1, x.size(-1)});
    int64_t num_tokens = x_2d.size(0);
    int64_t hidden_dim = x_2d.size(1);

    auto output    = torch::empty_like(x);
    auto rms_cache = torch::empty({num_tokens},
                        x.options().dtype(torch::kFloat32));

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    id<MTLCommandBuffer> cmd_buf = at::mps::getCurrentMPSStream()->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    id<MTLComputePipelineState> pipeline = is_bf16 ?
        g_rms_cache.fwd_bf16 : g_rms_cache.fwd_f32;

    [enc setComputePipelineState:pipeline];

    // Bind tensors — unified memory: MTLBuffer pointer IS the storage data ptr
    [enc setBuffer:MPS_BUF(x_2d)    offset:MPS_OFS(x_2d)    atIndex:0];
    [enc setBuffer:MPS_BUF(weight)   offset:MPS_OFS(weight)   atIndex:1];
    [enc setBuffer:MPS_BUF(output)   offset:MPS_OFS(output)   atIndex:2];
    [enc setBuffer:MPS_BUF(rms_cache) offset:0               atIndex:3];

    uint32_t h = (uint32_t)hidden_dim;
    float    e = (float)eps;
    [enc setBytes:&h length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&e length:sizeof(float)    atIndex:5];

    // Grid: one threadgroup per token, threads = ceil(hidden / N_READS)
    NSUInteger tg_size = MIN(((NSUInteger)hidden_dim + 3) / 4, (NSUInteger)1024);
    MTLSize grid = MTLSizeMake((NSUInteger)num_tokens, 1, 1);
    MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    // DO NOT call commit() here — let PyTorch's MPS scheduler manage flushing
    return {output.view(x.sizes()), rms_cache};
}

// ─── Backward ────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> metal_rms_norm_backward(
    torch::Tensor x,          // [tokens, hidden]
    torch::Tensor weight,     // [hidden]
    torch::Tensor grad_out,   // [tokens, hidden]
    double eps,
    const std::string& kernel_dir)
{
    TORCH_CHECK(x.device().is_mps(), "MetalLiger RMSNorm backward: input must be on MPS");

    ensure_compiled(kernel_dir);

    auto x_2d   = x.view({-1, x.size(-1)});
    auto g_2d   = grad_out.view({-1, grad_out.size(-1)});
    int64_t num_tokens = x_2d.size(0);
    int64_t hidden_dim = x_2d.size(1);

    auto grad_x   = torch::empty_like(x_2d);
    // Per-row grad_w (MLX strategy: each threadgroup writes its own row → no atomic contention)
    auto grad_w_per_row = torch::empty({num_tokens, hidden_dim}, x_2d.options());

    id<MTLCommandBuffer> cmd_buf = at::mps::getCurrentMPSStream()->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    id<MTLComputePipelineState> pipeline = is_bf16 ?
        g_rms_cache.bwd_bf16 : g_rms_cache.bwd_f32;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:MPS_BUF(x_2d)          offset:0 atIndex:0];
    [enc setBuffer:MPS_BUF(weight)         offset:0 atIndex:1];
    [enc setBuffer:MPS_BUF(g_2d)           offset:0 atIndex:2];
    [enc setBuffer:MPS_BUF(grad_x)         offset:0 atIndex:3];
    [enc setBuffer:MPS_BUF(grad_w_per_row) offset:0 atIndex:4];

    float    e = (float)eps;
    uint32_t h = (uint32_t)hidden_dim;
    [enc setBytes:&e length:sizeof(float)    atIndex:5];
    [enc setBytes:&h length:sizeof(uint32_t) atIndex:6];

    NSUInteger tg_size = MIN(((NSUInteger)hidden_dim + 3) / 4, (NSUInteger)1024);
    MTLSize grid = MTLSizeMake((NSUInteger)num_tokens, 1, 1);
    MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    // Cross-token reduction for grad_weight: sum over all tokens
    // This is a single BLAS call — PyTorch dispatches it optimally on MPS
    auto grad_weight = grad_w_per_row.sum(0);

    return {grad_x.view(x.sizes()), grad_weight};
}

// Bindings are in csrc/bindings.mm (single PYBIND11_MODULE for all ops)
