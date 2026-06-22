"""
kv_cache.py
-----------
KV Cache for fast autoregressive generation.

Satisfies: V2 Upgrade (KV Cache)
"""

from dataclasses import dataclass
import torch

@dataclass
class KVCache:
    """Per-layer key/value cache for incremental decoding.
    
    Attributes
    ----------
    key : torch.Tensor
        Cached key tensors of shape (B, T_so_far, head_size) or (B, T_so_far, C)
    value : torch.Tensor
        Cached value tensors of shape (B, T_so_far, head_size) or (B, T_so_far, C)
    """
    key: torch.Tensor
    value: torch.Tensor

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key and value to the cache and return the full updated tensors.
        
        Parameters
        ----------
        new_k : torch.Tensor
            New key tensor for the current timestep (B, T_new, ...)
        new_v : torch.Tensor
            New value tensor for the current timestep (B, T_new, ...)
            
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The full (historical + new) key and value tensors.
        """
        self.key = torch.cat([self.key, new_k], dim=1)
        self.value = torch.cat([self.value, new_v], dim=1)
        return self.key, self.value
