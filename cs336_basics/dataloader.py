import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from pathlib import Path


class StoryDataset:
    def __init__(self,path):
        self.path = path
    
    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            lines = []
            
            for line in f:
                if line.strip() == "<|endoftext|>":
                    story = "".join(lines).strip()
                    lines = []
                    
                    if story:
                        yield story
                else:
                    lines.append(line)
            
                
    
    def __getitem__(self, index):
        sample = torch.load(
            self.path[index],
            map_location="cpu",
            weights_only=True
        )
        return sample["x"], sample["y"]