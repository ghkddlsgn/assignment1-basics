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
    def __init__(self, params, lr=1e-3, beta1:float=0.9, beta2:float=0.999, eps:float=1e-8, weight_decay:float=0.01):

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self.init_raise_check(**defaults)

        super().__init__(params, defaults)
        self.beta1:float = beta1
        self.beta2:float = beta2
        self.m:Tensor = Tensor()
        self.v:Tensor = Tensor()
    
    def init_raise_check(self, lr, beta1, beta2, eps, weight_decay):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1]")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
       
    def init_modify_mv(self):
        pass
    
    @torch.no_grad()
    def step(self):
        loss = None

        for group in self.param_groups:
            params = []
            grads = []
            lrs = []

            #collect Tensor and lr
            for p in group["params"]:
                if p.grad is None: continue
                t = self.state[p].get("t", 0)

                params.append(p)
                grads.append(p.grad)
                lrs.append(group["lr"])
            if not params: continue
    
    def get_alpha(self)