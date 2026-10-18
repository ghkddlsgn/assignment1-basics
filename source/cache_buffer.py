import torch
import torch.nn as nn
from torch import Tensor

class CacheBuffer(nn.Module):
    buffer:Tensor #[batch, heads, seq, embed_dim]
    
    def __init__(self, shape:list[int,...] = [1,1,1,1], buffer_max_len:int = 65536, cache_seq_dim:int = -2, device:str="cpu", dtype=torch.float32):
        super().__init__()
        self.register_buffer("buffer", torch.empty(0, device=device, dtype=dtype), persistent=False)
        self.refresh_to_new_buffer(shape, buffer_max_len, cache_seq_dim)
    
    def refresh_to_new_buffer(self, shape:list[int,...], buffer_max_len:int, cache_seq_dim:int = -2):
        self.buffer_max_len = buffer_max_len
        shape[cache_seq_dim] = buffer_max_len
        self.buffer = self.buffer.new_empty(tuple(shape))
        self.len = 0
        self.cache_seq_dim = cache_seq_dim
    
    def clean_buffer(self):
        self.len = 0        
    
    def append_to_buffer(self, additional_cache:Tensor):
        #overflow check
        expected_shape = list(additional_cache.shape)
        additional_context_len = additional_cache.shape[self.cache_seq_dim]
        if self.len + additional_context_len > self.buffer_max_len:
            raise RuntimeError("Buffer overflow: max len reached")
        
        start = self.len
        end = start + additional_context_len
        
        # Resize only when dimensions other than sequence length change.
        expected_shape[self.cache_seq_dim] = self.buffer_max_len
        if self.len == 0 and tuple(expected_shape) != tuple(self.buffer.shape):
            self.refresh_to_new_buffer(expected_shape, self.buffer_max_len, self.cache_seq_dim)
        
        #find target buffer area to append
        target_dims = self._get_target_dim_buffer_slice(self.cache_seq_dim, start, end)        
        target_buffer_place = self.buffer[tuple(target_dims)] #where to write?

        if target_buffer_place.shape != additional_cache.shape:
            raise ValueError(f"Shape mismatch when appending to buffer: target shape {target_buffer_place.shape}, additional_cache shape {additional_cache.shape}")

        #write cache at buffer
        target_buffer_place.copy_(additional_cache) #attach additional_cache at the end of current cache
        self.len = end
    
    def get_cache(self) -> Tensor:
        target_dims = self._get_target_dim_buffer_slice(self.cache_seq_dim, 0, self.len)
        return self.buffer[target_dims]
    
    def _get_target_dim_buffer_slice(self, target_dim:int, start:int, end:int) -> tuple[slice, ...]:
        result = [slice(None)] * self.buffer.dim()
        result[target_dim] = slice(start, end)
        return tuple(result)