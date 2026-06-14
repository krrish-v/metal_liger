# MetalLiger ⚡ — Fused Metal Kernels for TrlMPS

> **Liger Kernel for Apple Silicon** — 66% less memory per op, ~1.8GB activation memory saved via fused LoRA backward, chunked CE loss eliminates 740MB logit spike.

---

## What Is This?

MetalLiger brings [Liger Kernel](https://github.com/linkedin/Liger-Kernel)-style operator fusion to Apple Silicon's MPS backend. It targets two bottlenecks that make training VLMs on Mac slow:

1. **Memory**: Intermediate tensors balloon memory on every op → fused ops eliminate them
2. **Activation memory**: PEFT LoRA saves X 3× for Q/K/V backward → fused LoRA backward saves it once

```
Standard PyTorch MPS (per transformer layer):
  RMSNorm: 6 Metal dispatches, 4 intermediate tensors
  SwiGLU:  4 Metal dispatches, 3 intermediate tensors
  RoPE:    3 Metal dispatches, 2 intermediate tensors
  CE Loss: 1 dispatch, 740MB logit tensor spike
  LoRA QKV: X saved 3× in autograd graph = 3 × 32MB × 28 layers = ~2.7GB

MetalLiger Phase 3 (Python fused ops):
  RMSNorm: 1 fused op, 0 intermediates           ← 66% memory saved
  SwiGLU:  1 fused op, 0 intermediates           ← gate tensor eliminated
  RoPE:    1 fused op, cos/sin recomputed        ← no precompute table
  CE Loss: chunked 8192, 32KB peak              ← 740MB spike gone

MetalLiger Phase 3.5 (Fused LoRA Backward):
  LoRA_QKV: X saved 1× for Q+K+V backward       ← ~1.8GB activation saved
  LoRA_MLP: gate/up/down fused, shared dX        ← ~0.3GB extra saved
  All math: pure torch.matmul + addmm_           ← no Triton, works on MPS
```

> **Note on Phase 4 (torch.compile):** Confirmed on PyTorch 2.9.1 MPS — `aot_eager` 
> provides **zero throughput improvement** for Qwen3-VL. Use `use_metal_liger_compile=False`.
> The compile warmup adds 2-3 minutes startup overhead with no step-time benefit.

---

## Quick Start

### With TRL (SFT or GRPO):
```python
SFTConfig(
    use_mps_optimization=True,
    use_metal_liger=True,            # Phase 3 + 3.5: fused ops + fused LoRA backward
    use_metal_liger_compile=False,   # Phase 4: disabled (no benefit on MPS)
    disable_tqdm=True,               # Eliminate .item() sync on every step
    logging_steps=500,               # Only 1 GPU sync per 500 steps
    save_steps=500,                  # Reduce checkpoint sync frequency
)
```

### Standalone:
```python
from metal_liger import apply_metal_liger_to_qwen3vl

model = get_peft_model(model, peft_config)          # PEFT wrapping first
model = apply_metal_liger_to_qwen3vl(model)          # Then MetalLiger
# Automatically applies: RMSNorm, SwiGLU, fused LoRA QKV+MLP backward
```

### Manual fused LoRA (for custom training loops):
```python
from metal_liger import apply_fused_lora_qkv, apply_fused_lora_mlp

# In attention forward — replaces 3 separate PEFT linear calls:
Q, K, V = apply_fused_lora_qkv(self_attn, hidden_states)

# In MLP forward — replaces gate+up+down with shared dX:
output = apply_fused_lora_mlp(mlp, hidden_states)
```
---

## M4 Pro Tuning Constants

| Constant | Value | Source |
|---|---|---|
| SIMD_WIDTH | 32 | Apple GPU SIMD lane width |
| MAX_THREADGROUP | 1024 | Metal limit |
| THREADGROUP_MEM | 32 KB | Shared memory per group |
| GPU_CORES | 16 | M4 Pro 16-core GPU |
| BANDWIDTH | 273 GB/s | Unified memory |


Run: `python test_metal_liger.py`

---

## Credits

- **MLX** (Apple) — Metal kernel source + backward pass VJP strategy
- **Liger Kernel** (LinkedIn) — Fused operator architecture patterns
- **Unsloth** (Daniel Han-Chen) — Fused LoRA QKV/MLP backward design pattern
- **PyTorch MPS team** — `at::mps::getCurrentMPSStream()` C++ API
