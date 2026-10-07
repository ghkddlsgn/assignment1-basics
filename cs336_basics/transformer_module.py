import math
import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum

class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.weight, 0, std, a = -3 * std, b=3*std)
    
    def forward(self, x:Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, a=-3.0, b=3.0)

    
    def forward(self, token_ids:Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSnormModule(nn.Module):
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

class SwigluModule(nn.Module):
    def __init__(self, )