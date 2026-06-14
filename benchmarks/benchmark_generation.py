import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from metal_liger.patch import apply_metal_liger_to_qwen3vl
import os

def benchmark_generation():
    model_id = "Qwen/Qwen2.5-1.5B" # Using a smaller model for fast benchmarking
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="mps"
    )
    
    prompt = "Explain quantum computing in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    # 1. Benchmark Standard Generate
    print("\nBenchmarking Standard Generate...")
    start_time = time.time()
    with torch.no_grad():
        outputs_std = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    std_time = time.time() - start_time
    std_tokens = outputs_std.shape[1] - inputs.input_ids.shape[1]
    print(f"Standard TPS: {std_tokens / std_time:.2f} tokens/s")
    
    # 2. Apply MetalLiger
    print("\nApplying MetalLiger (including Phase 5 Generator)...")
    model = apply_metal_liger_to_qwen3vl(model, use_optimized_generator=True)
    
    # 3. Benchmark Optimized Generate
    print("Benchmarking Optimized Generate...")
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)
        
    start_time = time.time()
    with torch.no_grad():
        outputs_opt = model.generate(**inputs, max_new_tokens=64)
    opt_time = time.time() - start_time
    opt_tokens = outputs_opt.shape[1] - inputs.input_ids.shape[1]
    print(f"Optimized TPS: {opt_tokens / opt_time:.2f} tokens/s")
    
    print(f"\nSpeedup: { (std_tokens / std_time) / (opt_tokens / opt_time) :.2f}x")
    
    # Verification
    decoded_std = tokenizer.decode(outputs_std[0], skip_special_tokens=True)
    decoded_opt = tokenizer.decode(outputs_opt[0], skip_special_tokens=True)
    print(f"\nStandard Output: {decoded_std[:50]}...")
    print(f"Optimized Output: {decoded_opt[:50]}...")

if __name__ == "__main__":
    benchmark_generation()
