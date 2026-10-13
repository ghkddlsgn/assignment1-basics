import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from pathlib import Path


class StoryDataset:
    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob("*.pt"))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        sample = torch.load(
            self.paths[index],
            map_location="cpu",
            weights_only=True
        )
        return sample["x"], sample["y"]