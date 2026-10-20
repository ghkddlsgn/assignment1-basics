import torch

from source.model_config.config_data_strucrue import (
    ModelConfig, PrenormTransformerConfig, TokenizerInfo, device
    )

#start of model configure
dtype = torch.float32

model_config = ModelConfig(
    tokenizer_info=TokenizerInfo(), #use default tokenizer gpt2
    context_length=128,
    device=device,
    dtype=dtype,
    layers=[
        PrenormTransformerConfig(
            embed_dim=512,
            hidden_dim=2048,
            out_dim=512,
            activation_func="swiglu",
            num_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            device=device,
            dtype=dtype,
        )
        for _ in range(8)
    ],
)