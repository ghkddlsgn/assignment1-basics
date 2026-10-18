import torch
import torch.nn as nn
from torch import Tensor
from source.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu, Embedding, softmax
from source.cache_buffer import CacheBuffer
from config import end_of_text_token_id


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

    
    @torch.compile()
    def forward(self, x:Tensor, use_cache:bool = False) -> Tensor:
        x = self.input(x)
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)

        x = self.norm(x)
        return self.output(x)
    
    def reset_kv_cache(self) -> None:
        for cache in self.kv_cache:
            cache["k"].clean_buffer()
            cache["v"].clean_buffer()
    
    @torch.compile()
    def generate_text(self, query_token_ids:Tensor, dtype=torch.long, device="cuda", max_new_tokens:int = 512, temperature:float = 1.0) -> Tensor:
        #only work for batch size == 1
        last_token_id:Tensor = Tensor()
        query = torch.tensor(query_token_ids, dtype, device).unsqueeze(0) #add batch dim
        if len(query_token_ids.shape) > 2:
            raise ValueError("Input query_token_ids must be a 1D or 2D tensor (sequence or [batch=1, sequence]).")
       
        generated = []

        for _ in range(max_new_tokens):
            logits = self.forward(query, use_cache=True)
            p_logit = softmax(logits[:, -1, :], -1, temperature)
            last_token_id = torch.multinomial(p_logit, 1)

            if last_token_id.item() == end_of_text_token_id:
                break

            generated.append(last_token_id)
            query = last_token_id
        
        if generated:
            return torch.cat(generated, dim=1)
        else:
            return query.new_empty((1, 0), dtype=query.dtype, device=query.device)
        
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
    
    @torch.compile()
    def forward(self, x:Tensor, use_cache:bool = False):
        res1 = self.norm1(x)
        res1 = self.multi_head_attention_rope(res1, use_cache=use_cache)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x