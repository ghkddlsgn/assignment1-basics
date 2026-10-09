from typing import NamedTuple, cast
import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu, Embedding

class KVCache(NamedTuple):
    k: Tensor
    v: Tensor

class TransformerLM(nn.Module):
    def __init__(self, input_dim:int = 512, hidden_dim:int = 512, 
                 vocab_size:int = 10000, context_length:int = 10000, num_layers:int = 8, 
                 num_heads:int = 8, device: str|None = None, dtype=torch.bfloat16):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
            
        # self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")
        # self.vocab_size = self.tokenizer.max_token_value + 1

        self.input = Embedding(vocab_size, hidden_dim, device, dtype)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(hidden_dim, num_heads, vocab_size, context_length, device, dtype)
            for _ in range(num_layers)
        ])
        
        self.kv_cache:list[KVCache|None] = []
        self.reset_kv_cache()

        self.norm = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.output = Linear(hidden_dim, vocab_size, device, dtype)
    
    def forward(self, x:Tensor):
        x = self.input(x)
        for i, (block, cur_kv) in enumerate(zip(self.transformer_blocks, self.kv_cache)):
            x, new_kv = block(x, cur_kv)
            self.kv_cache[i] = new_kv

        x = self.norm(x)
        x = self.output(x)
        return x
    
    def reset_kv_cache(self) -> None:
        self.kv_cache = [None] * self.num_layers

class PreNormTransformerBlock(nn.Module):
    def __init__(self, hidden_dim:int = 512, num_heads:int = 8,
                 vocab_size:int = 10000, context_length:int = 10000, device:str|None = None, dtype=torch.bfloat16):
        super().__init__()
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        self.norm1 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.multi_head_attention_rope = MultiheadSelfAttention(hidden_dim, num_heads, context_length, device=device, dtype=dtype)
        self.norm2 = RMSnorm(hidden_dim, device=device, dtype=dtype)
        self.ff = Swiglu(hidden_dim, device=device, dtype=dtype)
        
    def forward(self, x:Tensor, past_kv:KVCache|None=None):
        res1 = self.norm1(x)
        res1, new_k, new_v = self.multi_head_attention_rope(res1, past_kv)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x, KVCache(new_k, new_v)