# Copyright 2025 MetalLiger Contributors
"""Native Metal Generator for Apple Silicon.

Optimized token-by-token generation for unified memory.
Eliminates overhead by using a static KV-Cache and minimizing
Python->C++ crossings.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union

try:
    from transformers.cache_utils import Cache
except ImportError:
    class Cache: pass

class MetalLigerStaticCache(Cache):
    """Static KV-cache for MPS that fits the Transformers Cache API."""
    
    def __init__(self, model, batch_size, max_seq_len):
        super().__init__()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.config = model.config
        
        self.num_layers = getattr(self.config, "num_hidden_layers", 28)
        self.num_heads = getattr(self.config, "num_attention_heads", 16)
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.config.hidden_size // self.num_heads
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        
        # Buffer: [num_layers, 2, batch, num_kv_heads, max_seq_len, head_dim]
        self.key_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
            device=self.device, dtype=self.dtype
        )
        self.value_cache = torch.zeros(
            (self.num_layers, batch_size, self.num_kv_heads, max_seq_len, self.head_dim),
            device=self.device, dtype=self.dtype
        )
        self.seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update the cache and return the full history."""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        start = self.seen_tokens
        end = start + seq_len
        
        self.key_cache[layer_idx, :, :, start:end, :] = key_states
        self.value_cache[layer_idx, :, :, start:end, :] = value_states
        
        # Return the content up to the current length
        return self.key_cache[layer_idx, :, :, :end, :], self.value_cache[layer_idx, :, :, :end, :]

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def get_max_length(self):
        return self.max_seq_len

class MetalLigerGenerator:
    """High-speed token generator optimized for MPS."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.tokenizer = None # Should be set if used standalone

    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Optimized generation loop for MPS."""
        batch_size, prompt_len = input_ids.shape
        max_len = prompt_len + max_new_tokens
        
        # 1. Initialize Static Cache
        cache = MetalLigerStaticCache(self.model, batch_size, max_len)
        
        # 2. Burst Prompt (Prefill)
        outputs = self.model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            **kwargs
        )
        cache.seen_tokens += prompt_len
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        all_generated_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 3. Token-by-token Decode
        for i in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
            outputs = self.model(
                next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                **kwargs
            )
            cache.seen_tokens += 1
            
            logits = outputs.logits[:, -1, :]
            
            if temperature == 0 or (temperature == 1.0 and top_p == 1.0):
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
            all_generated_ids = torch.cat([all_generated_ids, next_token], dim=-1)
            
        return all_generated_ids

def patch_generator(model: nn.Module, force: bool = False):
    """Replace model.generate with MetalLigerGenerator.generate.
    
    WARNING: The custom generator does NOT support:
      - do_sample=True (no temperature/top_p/top_k sampling)
      - GenerationConfig 
      - attention_mask / pad_token_id handling
      - Any standard HF generate kwargs
    
    This means it is INCOMPATIBLE with GRPO/RL training which relies on 
    Transformers' native generate() for sampling. Only use for standalone
    greedy inference.
    
    Args:
        model: Model to patch
        force: If True, patch even if model.training is True (unsafe for GRPO)
    """
    if model.training and not force:
        import logging
        logging.getLogger(__name__).info(
            "MetalLiger: Skipping generator patch — model is in training mode. "
            "The custom generator is incompatible with GRPO/RL sampling. "
            "Use force=True to override."
        )
        return model
    
    generator = MetalLigerGenerator(model)
    # We store the original generate just in case
    if not hasattr(model, "_orig_generate"):
        model._orig_generate = model.generate
    
    # Monkey-patch
    model.generate = generator.generate
    return model
