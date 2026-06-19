"""
bigram.py
---------
Bigram Language Model baseline (Module 4 from SDD §5.1).

Satisfies: FR-7 (train bigram baseline LM)

The bigram model predicts the next token using *only* the current token —
equivalent to looking up a learned probability distribution from a table.
It is the simplest possible neural language model and serves three purposes:

1. Verifies the entire data → loss → backprop → generation pipeline.
2. Establishes a concrete performance baseline for comparison.
3. Shows, by its poor generation quality, *why* attention (context) is needed.

Architecture
------------
    idx (B, T)
      └─► token_embedding_table (vocab_size, vocab_size)  →  logits (B, T, vocab_size)
                                                             │
                                                             └─► cross_entropy(logits, targets) → loss

Note: The embedding table dimension is (vocab_size × vocab_size) — each row
is directly the logit distribution for the *next* token given the current one,
bypassing any embedding projection. This is the literal bigram factorisation.

Usage
-----
    from src.bigram import BigramLanguageModel
    model = BigramLanguageModel(vocab_size=65)
    logits, loss = model(xb, yb)
    generated = model.generate(idx, max_new_tokens=200)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """Character-level bigram language model.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens (characters) in the vocabulary.

    Attributes
    ----------
    token_embedding_table : nn.Embedding  shape (vocab_size, vocab_size)
        Each row ``[i]`` is the un-normalised distribution (logits) over the
        next token, given that the current token is ``i``.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optionally the cross-entropy loss.

        Parameters
        ----------
        idx     : torch.Tensor  shape (B, T)
            Integer token IDs.
        targets : torch.Tensor  shape (B, T) | None
            Next-token targets. If None, only logits are returned with loss=None.

        Returns
        -------
        logits : torch.Tensor  shape (B, T, vocab_size)
        loss   : torch.Tensor  scalar | None
        """
        # Each token ID indexes a row in the embedding table → logits (FR-7)
        logits: torch.Tensor = self.token_embedding_table(idx)  # (B, T, vocab_size)

        loss: torch.Tensor | None = None
        if targets is not None:
            B, T, C = logits.shape
            # F.cross_entropy expects (N, C) predictions and (N,) targets
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    # ------------------------------------------------------------------
    # Autoregressive generation (FR-17, FR-18 — simplified version)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        """Generate *max_new_tokens* tokens autoregressively.

        The bigram model only needs the *last* token to predict the next one,
        so no context cropping is necessary here. The full GPT generation loop
        (with cropping) lives in :mod:`src.generator`.

        Parameters
        ----------
        idx            : torch.Tensor  shape (B, T)
            Seed token sequence.
        max_new_tokens : int
            Number of new tokens to append.

        Returns
        -------
        torch.Tensor  shape (B, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Score the current sequence → pick last time-step's logits
            logits, _ = self(idx)
            logits = logits[:, -1, :]            # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)    # (B, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)             # (B, T+1)
        return idx
