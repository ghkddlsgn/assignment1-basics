import torch
import torch.nn as nn
from torch import Tensor
from source.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu, Embedding, softmax
from source.cache_buffer import CacheBuffer
from source.model_config.config_data_strucrue import ModelConfig, PrenormTransformerConfig
from copy import deepcopy
from typing import cast


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = deepcopy(config)
        self.config.__post_init__()
        config = self.config
        model_device = config.device
        model_dtype = config.dtype
        self.input_dim = config.layers[0].embed_dim
        self.hidden_dim = self.input_dim
        self.vocab_size = config.vocab_size
        self.max_kv_len = config.context_length
        self.num_layers = len(config.layers)
        self.end_of_text_token_id = config.tokenizer_info.end_of_text_token_id

        self.input = Embedding(self.vocab_size, self.input_dim, model_device, model_dtype)
        self.kv_cache = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()
        for layer in config.layers:
            block = PreNormTransformerBlock(layer, context_length=config.context_length)
            self.kv_cache.append(nn.ModuleDict({
                "k": block.multi_head_attention_rope.k_cache,
                "v": block.multi_head_attention_rope.v_cache,
            }))
            self.transformer_blocks.append(block)
        self.norm = RMSnorm(config.layers[-1].out_dim, eps=config.layers[-1].norm_eps, device=model_device, dtype=model_dtype)
        self.output = Linear(config.layers[-1].out_dim, self.vocab_size, model_device, model_dtype)

    @torch.compile()
    def forward(self, x:Tensor, use_cache:bool = False) -> Tensor:
        x = self.input(x.to(device=self.input.weight.device))
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)

        x = x.to(device=self.norm.g.device, dtype=self.norm.g.dtype)
        x = self.norm(x)
        return self.output(x)
    
    def reset_kv_cache(self) -> None:
        for cache in self.kv_cache:
            buffers = cast(nn.ModuleDict, cache)
            cast(CacheBuffer, buffers["k"]).clean_buffer()
            cast(CacheBuffer, buffers["v"]).clean_buffer()
    
    @torch.compile()
    def generate_text(self, query_token_ids:Tensor, max_new_tokens:int = 512, temperature:float = 1.0) -> Tensor:
        #only work for batch size == 1
        last_token_id:Tensor = Tensor()
        query = torch.as_tensor(
            query_token_ids, dtype=torch.long, device=self.input.weight.device,
        )
        if query.ndim == 1:
            query = query.unsqueeze(0) #add batch dim
        if len(query_token_ids.shape) > 2:
            raise ValueError("Input query_token_ids must be a 1D or 2D tensor (sequence or [batch=1, sequence]).")
       
        generated = []

        for _ in range(max_new_tokens):
            logits = self.forward(query, use_cache=True)
            p_logit = softmax(logits[:, -1, :], -1, temperature)
            last_token_id = torch.multinomial(p_logit, 1)

            if last_token_id.item() == self.end_of_text_token_id:
                break

            generated.append(last_token_id)
            query = last_token_id
        
        if generated:
            return torch.cat(generated, dim=1)
        else:
            return query.new_empty((1, 0), dtype=query.dtype, device=query.device)
        
class PreNormTransformerBlock(nn.Module):
    def __init__(self, config: PrenormTransformerConfig, context_length: int,
                 *, k_cache: CacheBuffer | None = None, v_cache: CacheBuffer | None = None):
        super().__init__()
        self.config = deepcopy(config)
        self.config.__post_init__()
        config = self.config
        device = config.device
        dtype = config.dtype
        self.hidden_dim = config.embed_dim
        self.context_length = context_length

        self.norm1 = RMSnorm(config.embed_dim, eps=config.norm_eps, device=device, dtype=dtype)
        if k_cache is None:
            k_cache = CacheBuffer(buffer_max_len=context_length, device=device, dtype=dtype)
        if v_cache is None:
            v_cache = CacheBuffer(buffer_max_len=context_length, device=device, dtype=dtype)
        k_cache = k_cache.to(device=device, dtype=dtype)
        v_cache = v_cache.to(device=device, dtype=dtype)
        self.multi_head_attention_rope = MultiheadSelfAttention(
            k_cache=k_cache, v_cache=v_cache, d_model=config.embed_dim, num_heads=config.num_heads,
            max_seq_len=context_length, rope_theta=config.rope_theta, device=device, dtype=dtype,
        )
        self.norm2 = RMSnorm(config.embed_dim, eps=config.norm_eps, device=device, dtype=dtype)
        self.ff = Swiglu(d_model=config.embed_dim, d_ff=config.hidden_dim, device=device, dtype=dtype)

    @torch.compile()
    def forward(self, x:Tensor, use_cache:bool = False):
        x = x.to(device=self.norm1.g.device, dtype=self.norm1.g.dtype)
        res1 = self.norm1(x)
        res1 = self.multi_head_attention_rope(res1, use_cache=use_cache)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2

        return x