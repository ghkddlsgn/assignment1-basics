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
        self.t = 0
            
    
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
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # One shared step count, regardless of the number of parameter groups.
        self.t += 1
        adjusted_lrs = self.get_new_lr()

        for group, alpha_t in zip(self.param_groups, adjusted_lrs):
            params = []
            grads = []
            ms = []
            vs = []

            # Collect tensors; moments persist in each parameter's state.
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                params.append(p)
                grads.append(p.grad)
                ms.append(state["m"])
                vs.append(state["v"])
            if not params:
                continue

            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]

            # 8: theta <- theta - lr * weight_decay * theta
            torch._foreach_mul_(params, 1 - lr * group["weight_decay"])

            # 9: m <- beta1 * m + (1 - beta1) * grad
            torch._foreach_mul_(ms, beta1)
            torch._foreach_add_(ms, grads, alpha=1 - beta1)

            # 10: v <- beta2 * v + (1 - beta2) * grad**2
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1 - beta2)

            # 11: theta <- theta - alpha_t * m / (sqrt(v) + eps)
            denominators = torch._foreach_sqrt(vs)
            torch._foreach_add_(denominators, group["eps"])
            torch._foreach_addcdiv_(params, ms, denominators, value=-alpha_t)

        return loss

    def get_new_lr(self) -> list[float]:
        """Return each group's bias-corrected lr without changing its base lr."""
        if self.t == 0:
            return [group["lr"] for group in self.param_groups]

        result: list[float] = []
        for group in self.param_groups:
            alpha_t = (
                group["lr"]
                * math.sqrt(1 - group["beta2"] ** self.t)
                / (1 - group["beta1"] ** self.t)
            )
            result.append(alpha_t)
        return result
