"""
feedforward.py
--------------
Per-token Feed-Forward Network (Module 7 from SDD §5.1).

Satisfies: FR-10 (per-token feed-forward network)

Role in the Transformer
-----------------------
After the attention mechanism allows tokens to *communicate* across time,
the FFN provides each token with a dedicated "thinking" step — a non-linear
transformation applied *independently* at each position. This is where the
model's capacity to represent complex per-token features lives.

Architecture (following Vaswani et al. §3.3, nanoGPT convention)
-----------------------------------------------------------------
    Linear(n_embd → 4 * n_embd)   — expand to 4× hidden dim
    ReLU                           — non-linearity
    Linear(4 * n_embd → n_embd)   — project back to model dim
    Dropout                        — regularisation (Stage 5)

The 4× expansion is the standard factor from "Attention Is All You Need".

Usage
-----
    from src.feedforward import FeedForward
    ffn = FeedForward(n_embd=128, dropout=0.1)
    out = ffn(x)   # x: (B, T, C)  →  out: (B, T, C)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward sub-layer.

    Each token's representation is transformed independently through a
    two-layer MLP with a 4× inner expansion. This adds non-linear capacity
    *after* the cross-token information exchange performed by attention.

    Parameters
    ----------
    n_embd  : int
        Model embedding dimension (C). Input and output both have this dim.
    dropout : float
        Dropout probability applied after the final linear layer.
        Set to 0.0 (default) for Stages 1–4; enabled in Stage 5 scaling.

    Notes
    -----
    * The operation is strictly *position-wise* — there is no interaction
      across the time (T) dimension here. Sequence mixing only happens in
      the attention sub-layer of :class:`src.block.Block`.
    * The 4× hidden-dimension expansion ratio follows the original Transformer
      paper and is the default in nanoGPT.
    """

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # expand
            nn.ReLU(),                         # non-linearity
            nn.Linear(4 * n_embd, n_embd),   # project back
            nn.Dropout(dropout),               # regularisation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise FFN to every token independently.

        Parameters
        ----------
        x : torch.Tensor  shape (B, T, C)
            Token representations after attention.

        Returns
        -------
        torch.Tensor  shape (B, T, C)
            Transformed token representations (same shape as input).
        """
        return self.net(x)
