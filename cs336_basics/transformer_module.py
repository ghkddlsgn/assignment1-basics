import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
import math

def softmax(x:Tensor, i:int):
    xi_max = x.max(dim=i, keepdim=True).values
    exps = torch.exp(x - xi_max)
    exp_sum = exps.sum(dim=i, keepdim=True)
    
    return exps/exp_sum

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
    # q = [batch, ..., seq_len_q, d_k]
    # k = [batch, ..., seq_len_k, d_k]
    # v = [batch, ..., seq_len_k, d_v]

    qk = einsum(q, k, "... seq_len_q emb, ... seq_len_k emb -> ... seq_len_q seq_len_k")
    presoftmax = qk / math.sqrt(q.shape[-1])  # [seq_len_q seq_len_k]

    if attn_mask is not None:
        presoftmax.masked_fill_(~attn_mask, float("-inf"))

    scores = softmax(presoftmax, -1)
    result = scores @ v
    return result

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
   
        nn.init.trunc_normal_(self.weight, 0, std, a = -3 * std, b=3*std)
    
    def forward(self, x:Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, a=-3.0, b=3.0)

    
    def forward(self, token_ids:Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:Tensor) -> Tensor:
        original_type = x.dtype
        x = x.to(torch.float32)
        result = ((x / self.rms(x)) * self.g)
        return result.to(original_type)

    def rms(self, x:Tensor):
        return torch.sqrt(
            (x**2).sum(dim=-1, keepdim=True) / self.d_model + self.eps
        )

class Swiglu(nn.Module):
    def __init__(self, d_model:int = 64, d_ff:int = 128, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        
        for w in (self.w1, self.w2, self.w3):
            d_in, d_out = w.shape
            std = (2 / (d_in + d_out)) ** 0.5
            nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)

    def silu(self, x:Tensor):
        return torch.sigmoid(x) * x
    
    def forward(self, x:Tensor):
        gate = self.silu(x @ self.w1.T)
        value = x @ self.w3.T
        swiglu = (gate * value) @ self.w2.T
        return swiglu

class RopeModule(nn.Module):
    cos_cache: Tensor
    sin_cache: Tensor

    def __init__(self, theta: float, d_head: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_head
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv_freq = theta ** (-torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head)
        
        angles = positions[:, None] * inv_freq[None, :]
        
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)
        
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
                
        a = x[..., 0::2]
        b = x[..., 1::2]
        
        a_rot = a * cos - b * sin
        b_rot = a * sin + b * cos
        
        return torch.stack((a_rot, b_rot), dim=-1).flatten(-2).to(x.dtype)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int=65_536, rope_theta:float = 10000, device=None, dtype=None):
        super().__init__()
        self.num_heads:int = num_heads
        head_dim:int = d_model // num_heads
        
        assert num_heads > 0
        assert d_model % num_heads == 0
        assert (d_model // num_heads) % 2 == 0
        
        self.rope = RopeModule(rope_theta, head_dim, max_seq_len, device)
        
        self.wq = nn.Parameter(torch.empty(num_heads * head_dim, d_model, device=device, dtype=dtype))
        self.wk = nn.Parameter(torch.empty(num_heads * head_dim, d_model, device=device, dtype=dtype))
        self.wv = nn.Parameter(torch.empty(num_heads * head_dim, d_model, device=device, dtype=dtype))
        self.w0 = nn.Parameter(torch.empty(num_heads * head_dim, d_model, device=device, dtype=dtype))
   
        
        for w in (self.wq, self.wk, self.wv, self.w0):
            std = math.sqrt(2/(w.shape[0] + w.shape[1]))
            nn.init.trunc_normal_(w, std=std, a=-3*std, b=3*std)
        
        
    def forward(self, x:Tensor, past_kv=None):
        """
        return attention score + (old kv + new kv)
        """
        #[batch head seq emb]
        new_q = self.get_xw(x, self.wq)
        new_k = self.get_xw(x, self.wk)
        new_v = self.get_xw(x, self.wv)
        
        past_kv_len = 0 if past_kv is None else past_kv[0].shape[-2]
        
        new_q_len, new_k_len = new_q.shape[-2], new_k.shape[-2]
        
        
        token_positions = torch.arange(
            past_kv_len, past_kv_len + new_q_len, device=new_q.device, dtype=torch.long
        )
        new_q_embed = self.rope(new_q, token_positions)
        new_k_embed = self.rope(new_k, token_positions)
        
        if past_kv is None:
            total_k = new_k_embed
            total_v = new_v
        else:
            past_k, past_v = past_kv
            total_k = torch.cat((past_k, new_k_embed), dim=-2)
            total_v = torch.cat((past_v, new_v), dim=-2)
        
        mask = torch.ones(
            new_q_len, total_k.shape[-2], dtype=torch.bool, device=new_q.device
        ).tril(diagonal=total_k.shape[-2] - new_q_len)

        result = scaled_dot_product_attention(new_q_embed, total_k, total_v, mask)
        result = rearrange(result, "... head seq emb -> ... seq (head emb)")
        result = result @ self.w0.T
        return result, total_k, total_v
        
    def get_xw(self, x:Tensor, w:Tensor):
        result = einsum(x, w, "... seq emb, head_total emb -> ... seq head_total")
        result = rearrange(result, "... seq (head emb) -> ... head seq emb", head=self.num_heads)
        return result