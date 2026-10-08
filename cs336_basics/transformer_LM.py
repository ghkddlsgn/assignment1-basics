import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.transformer_module import MultiheadSelfAttention

import tiktoken

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
                 vocab_size:int = 10000, context_length:int = 10000, num_layers:int = 8, device: str = "cuda" ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        self.multi_head_attention_rope = nn.ModuleList[MultiheadSelfAttention() * num_layers]
        
    def forward(self, x:Tensor):
        pass