"""
attention.py
------------
Causal Self-Attention (Module 5) and Multi-Head Attention (Module 6).

Satisfies: FR-8 (causal self-attention), FR-9 (multi-head attention)

Theory
------
Attention allows tokens to selectively "communicate" with past tokens by
computing similarity scores between queries (Q) and keys (K), then using
those scores to aggregate values (V).

    Attention(Q, K, V) = softmax( QK^T / sqrt(head_size) ) · V

The causal mask sets future positions to -∞ before softmax, ensuring that
position t cannot attend to positions t+1, t+2, … — this is the core of
autoregressive (left-to-right) language modelling.

Multi-head attention runs H independent attention heads in parallel, each
operating on a (C/H)-dimensional subspace. Their outputs are concatenated
and optionally projected back to the full model dimension C, giving the
model capacity to simultaneously capture H different relation types.

Architecture
------------
CausalSelfAttention
    Linear (n_embd → head_size)  ×3  for Q, K, V
    register_buffer 'tril'  — lower-triangular causal mask
    Dropout (on attention weights)

MultiHeadAttention
    ModuleList of n_head × CausalSelfAttention
    Linear projection (n_embd → n_embd)  — combines head outputs
    Dropout (on combined output)

Usage
-----
    from src.attention import CausalSelfAttention, MultiHeadAttention

    head  = CausalSelfAttention(n_embd=128, head_size=32, block_size=64, dropout=0.1)
    mha   = MultiHeadAttention(n_embd=128, n_head=4, block_size=64, dropout=0.1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module 5 — CausalSelfAttention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Single causal self-attention head.

    Parameters
    ----------
    n_embd     : int
        Total embedding dimension (C).
    head_size  : int
        Dimension of Q, K, V projections for this head (C // n_head).
    block_size : int
        Maximum sequence length — used to pre-allocate the causal mask (T).
    dropout    : float
        Dropout probability applied to attention weights after softmax.

    Attributes
    ----------
    key, query, value : nn.Linear (bias=False)
        Linear projections that transform (B, T, C) → (B, T, head_size).
    tril : torch.Tensor  shape (block_size, block_size)
        Lower-triangular boolean buffer used for the causal mask.
        Registered as a buffer so it moves with the model to GPU/CPU.
    attn_dropout : nn.Dropout
        Dropout on attention weights (regularisation in Stage 5).
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_size = head_size

        # Q, K, V projections — no bias (standard in modern Transformers)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: tril[i, j] = 1 iff j <= i  (T-6 validates this)
        # register_buffer: not a parameter, but moves with model.to(device)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scaled dot-product causal self-attention.

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, C)
            Input token representations.

        Returns
        -------
        torch.Tensor  shape (B, T, head_size)
            Context-aggregated output for this head.
        """
        B, T, C = x.shape

        # Project to Q, K, V  →  (B, T, head_size)
        q = self.query(x)   # (B, T, hs)
        k = self.key(x)     # (B, T, hs)
        v = self.value(x)   # (B, T, hs)

        # Attention scores — scaled by 1/√head_size to keep variance stable
        # (B, T, hs) @ (B, hs, T) → (B, T, T)
        scale = self.head_size ** -0.5
        wei = (q @ k.transpose(-2, -1)) * scale   # (B, T, T)

        # Apply causal mask: positions j > i get score → -∞ (future blocked)
        wei = wei.masked_fill(~self.tril[:T, :T], float("-inf"))

        # Softmax → attention weights (B, T, T); dropout for regularisation
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        # Aggregate values
        out = wei @ v   # (B, T, T) @ (B, T, hs) → (B, T, hs)
        return out


# ---------------------------------------------------------------------------
# Module 6 — MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention — H parallel causal attention heads.

    Each head operates on a (C/H)-dimensional subspace, learning different
    relational patterns simultaneously. Their outputs are concatenated to
    restore dimension C and then projected through a linear layer.

    Parameters
    ----------
    n_embd     : int
        Model embedding dimension (C).
    n_head     : int
        Number of parallel attention heads (H).
    block_size : int
        Maximum sequence length (T) — forwarded to each head.
    dropout    : float
        Dropout probability applied independently in each head and on the
        final projection output.

    Notes
    -----
    * ``head_size = n_embd // n_head`` — the SDD requires that head dimensions
      evenly divide the embedding dimension.
    * Output shape is (B, T, C) — same as input, enabling residual addition.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert n_embd % n_head == 0, (
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."
        )
        head_size = n_embd // n_head

        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(n_embd, head_size, block_size, dropout)
                for _ in range(n_head)
            ]
        )
        # Output projection: recombines concatenated head outputs back to C (T-7)
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all heads in parallel and combine their outputs.

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, C)

        Returns
        -------
        torch.Tensor  shape (B, T, C)
            Projected concatenation of all head outputs.
        """
        # Each head: (B, T, head_size) → concat along last dim → (B, T, C)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # Project back to embedding dim and apply dropout
        out = self.dropout(self.proj(out))
        return out
