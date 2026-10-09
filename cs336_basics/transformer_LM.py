from typing import Any, NamedTuple, cast
import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu, Embedding

class KVCache(NamedTuple):
    k: Tensor
    v: Tensor

class TransformerLM(nn.Module):
    def __init__(self, input_dim:int = 512, hidden_dim:int = 512, 
                 vocab_size:int = 10000, max_seq_len:int = 65_536, kv_update_seq_len_limit:int = 60000, num_layers:int = 8, 
                 num_heads:int = 8, device: str|None = None, dtype=torch.bfloat16):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")
        # self.vocab_size = self.tokenizer.max_token_value + 1

        self.input = Embedding(vocab_size, hidden_dim, device, dtype)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(hidden_dim, num_heads, vocab_size, max_seq_len, device, dtype)
            for _ in range(num_layers)
        ])
        
        self.kv_cache:list[KVCache|None] = []
        self.reset_kv_cache()

        self.norm = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.output = Linear(hidden_dim, vocab_size, device, dtype)
    
    def forward(self, x:Tensor, use_cache:bool = False):
        x = self.input(x)
        for i,block in enumerate(self.transformer_blocks):
            cur_kv = self.kv_cache[i]
            x, new_kv = block(x, cur_kv)
            
            if use_cache:
                self.kv_cache[i] = new_kv

        x = self.norm(x)
        return self.output(x)
    
    def reset_kv_cache(self) -> None:
        self.kv_cache = [None] * self.num_layers
        
    def update_kv_cache(self, block_idx:int, new_kv:KVCache) -> None:
        cur_context_len = len(self.kv_cache[block_idx].k)
        new_add_context_len = 
        if  cur_context_len + len(new_kv.k) > 60000:
            (k, v) = self.kv_cache[block_idx]
            new_k = k[-60000:].detach()
            new_v = v[-60000:].detach()
            
        

class PreNormTransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int = 512, num_heads:int = 8,
                 vocab_size:int = 10000, context_length:int = 65_536, device:str|None = None, dtype=torch.bfloat16):
        super().__init__()
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        self.norm1 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.multi_head_attention_rope = MultiheadSelfAttention(hidden_dim, num_heads, context_length, device=device, dtype=dtype)
        self.norm2 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.ff = Swiglu(d_model=hidden_dim, d_ff=128, device=device, dtype=dtype)
        
    def forward(self, x:Tensor, past_kv:KVCache|None=None):
        res1 = self.norm1(x)
        res1, new_k, new_v = self.multi_head_attention_rope(res1, past_kv)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x, KVCache(new_k, new_v)

class CacheBuffer(nn.Module):
    buffer:Tensor #[batch, heads, seq, embed_dim]
    
    def __init__(self, shape:tuple[int,...], max_len_dim_idx:int = -2, device:str="cpu", dtype=torch.float32):
        super().__init__()
        self.register_buffer("buffer", torch.empty(0, device=device, dtype=dtype), persistent=False)
        self.refresh_to_new_buffer(shape, max_len_dim_idx)
    
    def refresh_to_new_buffer(self, shape:tuple[int,...], max_len_dim_idx:int = -2):
        self.buffer = self.buffer.new_empty(shape)
        self.len = 0
        self.cache_seq_dim = max_len_dim_idx
        self.max_len = shape[max_len_dim_idx]
    
    def clean_buffer(self):
        self.len = 0
    
    def append_to_buffer(self, additional_cache:Tensor):
        additional_context_len = additional_cache.shape[self.cache_seq_dim]
        if self.len + additional_context_len > self.max_len:
            raise RuntimeError("Buffer overflow: max len reached")
        start = self.len
        end = start + additional_context_len
                
        target_dims = self._get_target_dim_buffer_slice(self.cache_seq_dim, start, end)        
        target_buffer_place = self.buffer[tuple(target_dims)] #where to write?

        if target_buffer_place.shape != additional_cache.shape:
            raise ValueError(f"Shape mismatch when appending to buffer: target shape {target_buffer_place.shape}, additional_cache shape {additional_cache.shape}")
       
        target_buffer_place.copy_(additional_cache) #attach additional_cache at the end of current cache
        self.len = end
    
    def get_cache(self) -> Tensor:
        target_dims = self._get_target_dim_buffer_slice(self.cache_seq_dim, 0, self.len)
        return self.buffer[target_dims]
    
    def _get_target_dim_buffer_slice(self, target_dim:int, start:int, end:int) -> tuple[slice, ...]:
        result = [slice(None)] * self.buffer.dim()
        result[target_dim] = slice(start, end)
        return tuple(result)