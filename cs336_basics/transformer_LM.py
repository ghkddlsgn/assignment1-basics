from typing import NamedTuple
import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu, Embedding
from cs336_basics.cache_buffer import CacheBuffer


class TransformerLM(nn.Module):
    def __init__(self, batch_dim:int = 1, input_dim:int = 512, hidden_dim:int = 512, 
                 vocab_size:int = 10000, max_kv_len:int = 65_536, kv_update_seq_len_limit:int = 60000, num_layers:int = 8, 
                 num_heads:int = 8, device: str|None = None, dtype=torch.bfloat16):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_kv_len = max_kv_len
        self.num_layers = num_layers

        self.input = Embedding(vocab_size, hidden_dim, device, dtype)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(hidden_dim, num_heads, vocab_size, max_kv_len, device, dtype)
            for _ in range(num_layers)
        ])
        
        self.kv_cache = nn.ModuleList([
            (
                CacheBuffer(buffer_max_len=max_kv_len, cache_seq_dim=-2, device=device, dtype=dtype),
                CacheBuffer(buffer_max_len=max_kv_len, cache_seq_dim=-2, device=device, dtype=dtype))
                for _ in range(num_layers)
            ])
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
        for k, v in self.kv_cache:
            k.clean_buffer()
            v.clean_buffer()
        
    def update_kv_cache(self, block_idx:int, new_kv) -> None:
        k,v = self.kv_cache[block_idx]
        new_q, new_k = new_kv
        k.append_to_buffer(new_q)
        v.append_to_buffer(new_k)

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
        
    def forward(self, x:Tensor, past_kv=None):
        res1 = self.norm1(x)
        res1, new_k, new_v = self.multi_head_attention_rope(res1, past_kv)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x, (new_k, new_v)
