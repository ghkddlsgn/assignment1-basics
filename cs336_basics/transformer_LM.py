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

        self.kv_cache = nn.ModuleList([
            nn.ModuleDict({
                "k": CacheBuffer(buffer_max_len=max_kv_len, cache_seq_dim=-2, device=device, dtype=dtype),
                "v": CacheBuffer(buffer_max_len=max_kv_len, cache_seq_dim=-2, device=device, dtype=dtype),
            })
            for _ in range(num_layers)
        ])
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(
                hidden_dim=hidden_dim, num_heads=num_heads, vocab_size=vocab_size,
                context_length=max_kv_len, device=device, dtype=dtype,
                k_cache=self.kv_cache[i]["k"], v_cache=self.kv_cache[i]["v"],
            )
            for i in range(num_layers)
        ])
        self.norm = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.output = Linear(hidden_dim, vocab_size, device, dtype)
    
    def forward(self, x:Tensor, use_cache:bool = False):
        x = self.input(x)
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)

        x = self.norm(x)
        return self.output(x)
    
    def reset_kv_cache(self) -> None:
        for cache in self.kv_cache:
            cache["k"].clean_buffer()
            cache["v"].clean_buffer()
        
class PreNormTransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int = 512, num_heads:int = 8,
                 vocab_size:int = 10000, context_length:int = 65_536, device:str|None = None, dtype=torch.bfloat16,
                 *, k_cache:CacheBuffer|None = None, v_cache:CacheBuffer|None = None):
        super().__init__()
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        self.norm1 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        if k_cache is None:
            k_cache = CacheBuffer(buffer_max_len=context_length, device=device, dtype=dtype)
        if v_cache is None:
            v_cache = CacheBuffer(buffer_max_len=context_length, device=device, dtype=dtype)
        self.multi_head_attention_rope = MultiheadSelfAttention(
            k_cache=k_cache, v_cache=v_cache, d_model=hidden_dim, num_heads=num_heads,
            max_seq_len=context_length, device=device, dtype=dtype,
        )
        self.norm2 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.ff = Swiglu(d_model=hidden_dim, d_ff=128, device=device, dtype=dtype)
        
    def forward(self, x:Tensor, use_cache:bool = False):
        res1 = self.norm1(x)
        res1 = self.multi_head_attention_rope(res1, use_cache=use_cache)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x