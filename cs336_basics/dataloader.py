from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from pathlib import Path
from typing import Annotated
from pydantic import Field, validate_call
import multiprocessing as mp

#for multi processing---------------------------
def tokenize_story(story, tokenizer):
    return tokenizer.encode(story)

#-----------------------------------------------

class StoryDataset:
    def __init__(self, path):
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

            story = "".join(lines).strip()
            if story:
                yield story

class StoryDataLoader:
    @validate_call
    def __init__(self, dataset:StoryDataset, context_length:Annotated[int, Field(gt=0)], tokenizer,
                 batch_size:int = 1, device:str="cuda", num_worker:Annotated[int, Field(gt=0)] = 2):
        self.dataset = dataset
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.num_worker = num_worker

    def __iter__(self):
        inputs, targets = [], []
        length = self.context_length
        
        encode = partial(tokenize_story, tokenizer=self.tokenizer)
        ctx = mp.get_context("spawn")
        
        with ctx.Pool(processes=self.num_worker) as pool:
            results = pool.imap(encode, self.dataset, chunksize=1)
            
            for token_ids in results:
                tokens = torch.tensor(token_ids, dtype=torch.long)
                
                for start in range(0, len(tokens) - length, length):
                    end = start + length
                    x = tokens[start: end]
                    y = tokens[start + 1 : end + 1]
                    
                    inputs.append(x)
                    targets.append(y)
                    
                    if len(inputs) == self.batch_size:
                        x_batch = torch.stack(inputs).to(self.device)
                        y_batch = torch.stack(targets).to(self.device)
                        
                        yield x_batch, y_batch