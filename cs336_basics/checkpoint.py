import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

import os
import typing

def save_checkpoint(model:nn.Module, optimizer:Optimizer, iteration:int, 
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    savings = []
    model.state_dict()

def load_checkpoint(src, model:nn.Module, optimizer)