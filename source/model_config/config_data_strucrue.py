from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TypedDict

import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported model dtype: {name}")
    return dtypes[name]


def _validate_placement(device: str, dtype: torch.dtype) -> None:
    if not isinstance(device, str):
        raise ValueError("device must be a device string, not None")
    torch.device(device)
    if dtype not in (
        torch.float16, torch.bfloat16, torch.float32, torch.float64,
    ):
        raise ValueError("dtype must be a supported floating-point torch.dtype")


class TokenizerInfoDict(TypedDict):
    encoding_name: str
    vocab_size: int
    end_of_text_token: str
    end_of_text_token_id: int


class PrenormTransformerConfigDict(TypedDict):
    embed_dim: int
    hidden_dim: int
    out_dim: int
    activation_func: str
    num_heads: int
    rope_theta: float
    norm_eps: float
    device: str
    dtype: str


class ModelConfigDict(TypedDict):
    tokenizer_info: TokenizerInfoDict
    context_length: int
    layers: list[PrenormTransformerConfigDict]
    device: str
    dtype: str


@dataclass
class TokenizerInfo:
    encoding_name: str = "gpt2"
    vocab_size: int = 50257
    end_of_text_token: str = "<|endoftext|>"
    end_of_text_token_id: int = 50256

    @cached_property
    def get_tokenizer(self):
        # Load only when preprocessing text, not when constructing a model.
        import tiktoken

        tokenizer = tiktoken.get_encoding(self.encoding_name)
        if (tokenizer.n_vocab != self.vocab_size
                or tokenizer.eot_token != self.end_of_text_token_id):
            raise ValueError("Tokenizer metadata does not match encoding_name")
        return tokenizer


@dataclass
class PrenormTransformerConfig:
    embed_dim: int = 512
    hidden_dim: int = 2048  # FFN intermediate dimension
    out_dim: int = 512
    activation_func: str = "swiglu"
    num_heads: int = 8
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    device: str = device
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        _validate_placement(self.device, self.dtype)
        if min(self.embed_dim, self.hidden_dim, self.out_dim, self.num_heads) <= 0:
            raise ValueError("Layer dimensions and num_heads must be positive")
        if self.out_dim != self.embed_dim:
            raise ValueError("Residual connections require out_dim == embed_dim")
        if self.embed_dim % self.num_heads or (self.embed_dim // self.num_heads) % 2:
            raise ValueError("embed_dim must be divisible by num_heads with an even RoPE head dimension")
        if self.activation_func != "swiglu":
            raise NotImplementedError("Only the existing swiglu implementation is supported")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")


@dataclass
class ModelConfig:
    tokenizer_info: TokenizerInfo = field(default_factory=TokenizerInfo)
    context_length: int = 128
    layers: list[PrenormTransformerConfig] = field(
        default_factory=lambda: [PrenormTransformerConfig() for _ in range(8)]
    )

    device: str = device
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        _validate_placement(self.device, self.dtype)
        if self.context_length <= 0 or not self.layers:
            raise ValueError("context_length must be positive and layers must not be empty")
        if self.tokenizer_info.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.tokenizer_info.end_of_text_token_id < self.tokenizer_info.vocab_size:
            raise ValueError("end_of_text_token_id must be inside the vocabulary")
        for layer in self.layers:
            layer.__post_init__()
            if layer.embed_dim != self.layers[0].embed_dim:
                raise ValueError("All residual blocks must have the same embed_dim")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer_info.vocab_size

    def to_dict(self) -> ModelConfigDict:
        return {
            "tokenizer_info": {
                "encoding_name": self.tokenizer_info.encoding_name,
                "vocab_size": self.tokenizer_info.vocab_size,
                "end_of_text_token": self.tokenizer_info.end_of_text_token,
                "end_of_text_token_id": self.tokenizer_info.end_of_text_token_id,
            },
            "context_length": self.context_length,
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "layers": [
                {
                    "embed_dim": layer.embed_dim,
                    "hidden_dim": layer.hidden_dim,
                    "out_dim": layer.out_dim,
                    "activation_func": layer.activation_func,
                    "num_heads": layer.num_heads,
                    "rope_theta": layer.rope_theta,
                    "norm_eps": layer.norm_eps,
                    "device": layer.device,
                    "dtype": str(layer.dtype).removeprefix("torch."),
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, values: ModelConfigDict) -> "ModelConfig":
        """
        create new instance of same class from values
        """
        return cls(
            tokenizer_info=TokenizerInfo(**values["tokenizer_info"]),
            context_length=values["context_length"],
            device=values["device"],
            dtype=dtype_from_name(values["dtype"]),
            layers=[
                PrenormTransformerConfig(
                    embed_dim=layer["embed_dim"], hidden_dim=layer["hidden_dim"],
                    out_dim=layer["out_dim"], activation_func=layer["activation_func"],
                    num_heads=layer["num_heads"], rope_theta=layer["rope_theta"],
                    norm_eps=layer["norm_eps"], device=layer["device"],
                    dtype=dtype_from_name(layer["dtype"]),
                )
                for layer in values["layers"]
            ],
        )


@dataclass
class DatasetConfig:
    data_files: str = "data/TinyStoriesV2-GPT4-train.txt"
    cache_dir: str = "data/hf_cache"
    split: str = "train"

    def load(self):
        from datasets import load_dataset

        project_root = Path(__file__).resolve().parents[2]
        return load_dataset(
            "text",
            data_files=str(project_root / self.data_files),
            split=self.split,
            cache_dir=str(project_root / self.cache_dir),
        )
