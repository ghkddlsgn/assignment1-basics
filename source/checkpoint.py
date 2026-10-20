import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

import os
import typing

from source.model_config.config_data_strucrue import ModelConfigDict


class ModelCheckpointDict(typing.TypedDict):
    epoch: int
    global_step: int
    model_config: ModelConfigDict
    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, typing.Any]
    optimizer_t: int


def save_checkpoint(model:nn.Module, optimizer:Optimizer, iteration:int, 
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    savings = []
    model.state_dict()

def load_checkpoint(src, model:nn.Module, optimizer):
    pass
