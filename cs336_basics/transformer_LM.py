import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.transformer_module import MultiheadSelfAttention, RMSnorm, Linear, Swiglu

class TransformerLM(nn.Module):
    def __init__(self, input_dim:int = 512, hidden_dim:int = 512, output_dim:int = 512, 
                 vocab_size:int = 10000, context_length:int = 10000, num_layers:int = 8, num_heads:int = 8, device: str = "cuda" ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers                 
            
        # self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")
        # self.vocab_size = self.tokenizer.max_token_value + 1
    
    def forward(self, x:Tensor):
        pass

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
        
    def forward(self, x:Tensor, past_kv=None):
        res1 = self.norm1(x)
        res1, new_k, new_v = self.multi_head_attention_rope(res1, past_kv)
        x = x + res1
        del res1

        res2 = self.norm2(x)
        res2 = self.ff(res2)
        x = x + res2
        del res2

        return x