from typing import TypedDict
from unittest import loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from cs336_basics.transformer_LM import TransformerLM
from cs336_basics.Optimizer import AdamW

class TokenDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.tokens = torch.as_tensor(
            token_ids, dtype=torch.long, device="cpu"
        )
        self.context_length = context_length

        if self.tokens.ndim != 1:
            raise ValueError("token_ids must be 1-dimensional.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if len(self.tokens) < context_length + 1:
            raise ValueError("Not enough tokens for training.")
       

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, index):
        start = index * self.context_length
        end = start + self.context_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y

class ModelConfig(TypedDict):
    hidden_dim: int
    vocab_size: int
    max_kv_len: int
    num_layers: int
    num_heads: int

def train(token_ids, vocab_size, num_epoch:int=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    context_length = 128
    
    dataset = TokenDataset(token_ids, context_length)

    if dataset.tokens.max().item() >= vocab_size:
        raise ValueError("token size exceeded vocab size")
    
    train_data = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    model_config:ModelConfig = ModelConfig{
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
        
        for x,y in loader:
            x:Tensor
            y:Tensor
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(x, use_cache=True)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            
            if not torch.isinf(loss).item():
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
        
        from cs336_basics.checkpoint import ModelCheckpointDict  # assumes this is where the typed dict is defined for now

        checkpoint: ModelCheckpointDict = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_t": optimizer.t,
        }
        torch.save(checkpoint, "checkpoint.pt")
    
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