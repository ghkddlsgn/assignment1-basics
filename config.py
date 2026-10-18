import torch
import torch.nn as nn
import tiktoken
from datasets import load_dataset, Features, List, Value
from dataclasses import dataclass

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TokenizerInfo:
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    end_of_text_token = "<|endoftext|>"
    end_of_text_token_id = tokenizer.eot_token #gpt2

#model architecture
@dataclass
class ModelConfig:
    tokenizer_info = TokenizerInfo()
    context_length = 128
    num_layers = 8



dataset = load_dataset(
    "text",
    data_files="data/TinyStoriesV2-GPT4-train.txt",
    split="train",
    cache_dir="data/hf_cache",
)