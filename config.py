import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
end_of_text_token = "<|endoftext|>"
end_of_text_token_id = 50256 #gpt2