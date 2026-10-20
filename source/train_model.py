import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

from source.transformer_LM import TransformerLM
from source.Optimizer import AdamW
from source.checkpoint import ModelCheckpointDict

from source.model_config.config_data_strucrue import ModelConfig, PrenormTransformerConfig, TokenizerInfo

def train(dataset: Dataset, config: ModelConfig, num_epoch:int=3):
    if num_epoch <= 0 or len(dataset) == 0:
        raise ValueError("Training requires a nonempty dataset and positive num_epoch.")
    
    config.__post_init__()
    device = config.device
    vocab_size = config.vocab_size
    context_length = config.context_length
    feature = dataset.features.get("input_ids")
    if getattr(feature, "length", -1) != context_length + 1:
        raise ValueError("input_ids length must equal config.context_length + 1")
    
    dataset = dataset.with_format("torch", columns=["input_ids"], dtype=torch.long)
    train_data = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=(torch.device(device).type == "cuda"), drop_last=False)
    print(
        f"tokens_per_epoch={len(dataset) * context_length:,} samples={len(dataset):,} "
        f"steps_per_epoch={len(train_data):,} total_steps={len(train_data) * num_epoch:,}",
        flush=True,
    )
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    global_step = 0
    
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in train_data:
            tokens = batch["input_ids"]
            if tokens.min().item() < 0 or tokens.max().item() >= vocab_size:
                raise ValueError("token ID outside vocabulary")
            tokens = tokens.to(device, non_blocking=True)
            x, y = tokens[:, :-1], tokens[:, 1:]
            
            optimizer.zero_grad()
            logits = model(x, use_cache=False)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"loss unstable. loss:{loss}")
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            
            global_step += 1
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            if global_step % 20 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={global_step} "
                    f"loss={loss.item():.4f}"
                )
        
        print(
            f"epoch={epoch + 1} "
            f"train_loss={total_loss / total_tokens:.4f}"
        )
        
        checkpoint: ModelCheckpointDict = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_t": optimizer.t,
        }
        torch.save(checkpoint, "checkpoint.pt")
    
    return model, optimizer, checkpoint
    
def load_checkpoint():
    checkpoint = torch.load(
        "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )

    values = checkpoint["model_config"]
    if "layers" in values:
        model_config = ModelConfig.from_dict(values)
    else:
        # Existing checkpoints used a fixed FFN dimension of 128.
        model_config = ModelConfig(
            tokenizer_info=TokenizerInfo(vocab_size=values["vocab_size"]),
            context_length=values["max_kv_len"],
            layers=[
                PrenormTransformerConfig(
                    embed_dim=values["hidden_dim"], hidden_dim=128,
                    out_dim=values["hidden_dim"], num_heads=values["num_heads"],
                )
                for _ in range(values["num_layers"])
            ],
        )
    model = TransformerLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = AdamW(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer.t = checkpoint["optimizer_t"]
    
    return model, optimizer, checkpoint
