from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
import math

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, current lr = {lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)
    
    def step(self, closure:Optional[Callable] = None):
        """
        it takes param groups like this
        self.param_groups = [
            {"params": [weight1, bias1], "lr": 0.01},
            {"params": [weight2, bias2], "lr": 0.001},
        ]
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t + 1) * grad #learning rate decay
                state["t"] = t + 1 #increment count how much time this param changed
        
        return loss

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, )