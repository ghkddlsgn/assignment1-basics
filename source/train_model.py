import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

from cs336_basics.transformer_LM import TransformerLM
from cs336_basics.Optimizer import AdamW
from cs336_basics.checkpoint import ModelConfig, ModelCheckpointDict

def train(dataset: Dataset, vocab_size, num_epoch:int=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if num_epoch <= 0 or len(dataset) == 0:
        raise ValueError("Training requires a nonempty dataset and positive num_epoch.")
    
    feature = dataset.features.get("input_ids")
    context_length = getattr(feature, "length", -1) - 1
    
    dataset = dataset.with_format("torch", columns=["input_ids"], dtype=torch.long)
    train_data = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=device == "cuda", drop_last=False)
    print(
        f"tokens_per_epoch={len(dataset) * context_length:,} samples={len(dataset):,} "
        f"steps_per_epoch={len(train_data):,} total_steps={len(train_data) * num_epoch:,}",
        flush=True,
    )
    model_config:ModelConfig = {
        "hidden_dim":128,
        "vocab_size": vocab_size,
        "max_kv_len":context_length,
        "num_layers":2,
        "num_heads":4
    }
    
    model = TransformerLM(**model_config, device=device, dtype=torch.float32)
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
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_t": optimizer.t,
        }
        torch.save(checkpoint, "checkpoint.pt")
    
    return model, optimizer, checkpoint
    
def load_checkpoint():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        "checkpoint.pt",
        map_location=device,
        weights_only=True,
    )

    model = TransformerLM(
        **checkpoint["model_config"],
        device=device,
        dtype=torch.float32,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = AdamW(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer.t = checkpoint["optimizer_t"]
    
    return model, optimizer, checkpoint
