// Copyright 2025 MetalLiger Contributors
// Phase 4d: Native Metal dispatch for fused SwiGLU — Forward + Backward
//
// SiLU(gate) * up in a single kernel pass — no intermediate gate tensor stored.
// Backward recomputes sigmoid(gate) (cheap) instead of reading from cache (expensive).

#include "metal_liger_ops.h"
#import  <Metal/Metal.h>
#import  <Foundation/Foundation.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#include <stdexcept>


// Correct MTLBuffer access for PyTorch 2.x MPS (no getMTLBufferStorage in public API)
#define MPS_BUF(t)  ((__bridge id<MTLBuffer>)(t).storage().data())
#define MPS_OFS(t)  ((NSUInteger)((t).storage_offset() * (t).element_size()))


// ─── Pipeline cache ──────────────────────────────────────────────────────────

struct SwiGLUCache {
    id<MTLComputePipelineState> fwd_bf16 = nil;
    id<MTLComputePipelineState> bwd_bf16 = nil;
    id<MTLComputePipelineState> fwd_f32  = nil;
    id<MTLComputePipelineState> bwd_f32  = nil;
    bool compiled = false;
};

static SwiGLUCache g_swiglu_cache;

static void ensure_swiglu_compiled(const std::string& kernel_dir) {
    if (g_swiglu_cache.compiled) return;

    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();

    NSString* path = [NSString stringWithUTF8String:(kernel_dir + "/swiglu.metal").c_str()];
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
    if (!src) {
        throw std::runtime_error(std::string("MetalLiger SwiGLU: failed to load kernel: ") +
                                 [[err localizedDescription] UTF8String]);
    }

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        throw std::runtime_error(std::string("MetalLiger SwiGLU: kernel compile failed: ") +
                                 [[err localizedDescription] UTF8String]);
    }

    auto get_pipeline = [&](NSString* fn_name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
        if (!fn) return nil;
        return [device newComputePipelineStateWithFunction:fn error:&err];
    };

    g_swiglu_cache.fwd_bf16 = get_pipeline(@"swiglu_forward_bfloat16");
    g_swiglu_cache.bwd_bf16 = get_pipeline(@"swiglu_backward_bfloat16");
    g_swiglu_cache.fwd_f32  = get_pipeline(@"swiglu_forward_float");
    g_swiglu_cache.bwd_f32  = get_pipeline(@"swiglu_backward_float");
    g_swiglu_cache.compiled = true;
}

// ─── Forward: SiLU(gate) * up ─────────────────────────────────────────────

torch::Tensor metal_swiglu_forward(
    torch::Tensor gate,        // [batch, seq, intermediate]
    torch::Tensor up,          // [batch, seq, intermediate]
    const std::string& kernel_dir)
{
    TORCH_CHECK(gate.device().is_mps(), "MetalLiger SwiGLU: gate must be on MPS");
    TORCH_CHECK(gate.sizes() == up.sizes(), "MetalLiger SwiGLU: gate and up must match");

    ensure_swiglu_compiled(kernel_dir);

    auto output = torch::empty_like(gate);

    int64_t numel = gate.numel();

    id<MTLCommandBuffer> cmd_buf = at::mps::getCurrentMPSStream()->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    bool is_bf16 = (gate.scalar_type() == torch::kBFloat16);
    id<MTLComputePipelineState> pipeline =
        is_bf16 ? g_swiglu_cache.fwd_bf16 : g_swiglu_cache.fwd_f32;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:MPS_BUF(gate)   offset:MPS_OFS(gate)   atIndex:0];
    [enc setBuffer:MPS_BUF(up)     offset:MPS_OFS(up)     atIndex:1];
    [enc setBuffer:MPS_BUF(output) offset:MPS_OFS(output) atIndex:2];
    uint32_t n = (uint32_t)numel;
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];

    // SwiGLU is elementwise — flat dispatch over all elements
    NSUInteger tg_size = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger grid_size = ((NSUInteger)numel + tg_size - 1) / tg_size;
    [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];

    return output;
}

// ─── Backward ────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> metal_swiglu_backward(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor grad_out,
    const std::string& kernel_dir)
{
    TORCH_CHECK(gate.device().is_mps(), "MetalLiger SwiGLU backward: gate must be on MPS");

    ensure_swiglu_compiled(kernel_dir);

    auto grad_gate = torch::empty_like(gate);
    auto grad_up   = torch::empty_like(up);

    int64_t numel = gate.numel();

    id<MTLCommandBuffer> cmd_buf = at::mps::getCurrentMPSStream()->commandBuffer();
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

    bool is_bf16 = (gate.scalar_type() == torch::kBFloat16);
    id<MTLComputePipelineState> pipeline =
        is_bf16 ? g_swiglu_cache.bwd_bf16 : g_swiglu_cache.bwd_f32;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:MPS_BUF(gate)      offset:0 atIndex:0];
    [enc setBuffer:MPS_BUF(up)        offset:0 atIndex:1];
    [enc setBuffer:MPS_BUF(grad_out)  offset:0 atIndex:2];
    [enc setBuffer:MPS_BUF(grad_gate) offset:0 atIndex:3];
    [enc setBuffer:MPS_BUF(grad_up)   offset:0 atIndex:4];
    uint32_t n = (uint32_t)numel;
    [enc setBytes:&n length:sizeof(uint32_t) atIndex:5];

    NSUInteger tg_size = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger grid_size = ((NSUInteger)numel + tg_size - 1) / tg_size;
    [enc dispatchThreadgroups:MTLSizeMake(grid_size, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [enc endEncoding];

    return {grad_gate, grad_up};
}

// Bindings are in csrc/bindings.mm (single PYBIND11_MODULE for all ops)
