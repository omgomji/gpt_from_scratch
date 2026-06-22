"""
block.py
--------
Transformer Block (Module 8 from SDD §5.1).

Satisfies: FR-11 (Transformer block with residual + LayerNorm)

Overview
--------
A Transformer block composes two sub-layers:
    1. Causal Multi-Head Self-Attention (cross-token communication)
    2. Position-wise Feed-Forward Network (per-token computation)

Each sub-layer is wrapped in a *pre-norm + residual* pattern:

    x = x + Attention( LayerNorm(x) )   # attend, then add back
    x = x + FFN( LayerNorm(x) )         # think,  then add back

Pre-norm (applying LayerNorm before the sub-layer) is the standard in
modern implementations (GPT-2, nanoGPT) because it improves gradient
flow compared to post-norm used in the original 2017 paper.

Why residuals?
--------------
Residual connections allow gradients to flow directly from deep layers
back to early layers, avoiding vanishing gradients. Without them, stacking
more than ~2 blocks becomes practically untrainable.

Why LayerNorm?
--------------
LayerNorm normalises each token's feature vector independently, keeping
activation statistics stable across depth — critical for n_layer > 2.

Usage
-----
    from src.block import Block
    block = Block(n_embd=128, n_head=4, block_size=64, dropout=0.1)
    x_out = block(x)   # x: (B, T, C) → x_out: (B, T, C)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feedforward import FeedForward
from src.kv_cache import KVCache


class Block(nn.Module):
    """A single Transformer decoder block.

    Combines causal multi-head self-attention and a feed-forward network,
    each wrapped with pre-norm LayerNorm and a residual (skip) connection.

    Parameters
    ----------
    n_embd     : int
        Model embedding dimension (C).
    n_head     : int
        Number of parallel attention heads.
    block_size : int
        Maximum sequence length (forwarded to MultiHeadAttention).
    dropout    : float
        Dropout probability (forwarded to sub-layers).

    Attributes
    ----------
    ln1  : nn.LayerNorm  — normalisation before attention
    attn : MultiHeadAttention
    ln2  : nn.LayerNorm  — normalisation before FFN
    ffwd : FeedForward
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, kv_caches: list[KVCache] | None = None) -> torch.Tensor:
        """Apply pre-norm attention and FFN with residual connections.

        Implements the pre-norm residual pattern from SDD §5.1 (Module 8)::

            x = x + self.attn(self.ln1(x))   # attend, then add back
            x = x + self.ffwd(self.ln2(x))   # think,  then add back

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, C)
            Token representations from the previous block (or embeddings).

        Returns
        -------
        torch.Tensor  shape (B, T, C)
            Updated token representations.
        """
        # Attention sub-layer: pre-norm → attend → residual add
        x = x + self.attn(self.ln1(x), kv_caches)
        # FFN sub-layer: pre-norm → feed-forward → residual add
        x = x + self.ffwd(self.ln2(x))
        return x
