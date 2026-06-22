"""
model.py
--------
Full GPT-style Decoder-only Transformer (Module 9 from SDD §5.1).

Satisfies: FR-12 (assemble GPT model), FR-13 (cross-entropy loss)

Architecture (SDD §5.2 class relationship diagram)
--------------------------------------------------
    GPTModel
     ├─ token_embedding_table    : nn.Embedding (vocab_size, n_embd)
     ├─ position_embedding_table : nn.Embedding (block_size, n_embd)
     ├─ blocks                   : nn.Sequential  ← n_layer × Block
     ├─ ln_f                     : nn.LayerNorm(n_embd)      (final norm)
     └─ lm_head                  : nn.Linear(n_embd, vocab_size, bias=False)

Forward pass
------------
    tok_emb  = token_embedding_table(idx)                # (B, T, C)
    pos_emb  = position_embedding_table(arange(T))       # (T, C)
    x        = tok_emb + pos_emb                         # (B, T, C)
    x        = blocks(x)                                 # (B, T, C) through N blocks
    x        = ln_f(x)                                   # (B, T, C)
    logits   = lm_head(x)                                # (B, T, vocab_size)
    loss     = cross_entropy(logits, targets)             # scalar (if targets given)

Positional embeddings
---------------------
Learned absolute positional embeddings — each of the T positions gets its own
embedding vector, added to the token embedding. This gives the model awareness
of *where* a token appears in the context window.

Weight tying
------------
lm_head and token_embedding_table share no weights here (kept separate for
clarity), following the nanoGPT tutorial approach. Weight tying is noted as
a future optimisation.

Usage
-----
    from src.config import get_config
    from src.model import GPTModel

    cfg    = get_config("small")
    model  = GPTModel(cfg).to(cfg.resolved_device())
    logits, loss = model(xb, yb)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block import Block
from src.config import Config
from src.kv_cache import KVCache


class GPTModel(nn.Module):
    """Decoder-only Transformer language model.

    Parameters
    ----------
    config : Config
        Hyperparameter bundle (block_size, n_embd, n_head, n_layer,
        dropout, vocab_size is injected externally via set_vocab_size).

    Attributes
    ----------
    token_embedding_table    : nn.Embedding (vocab_size, n_embd)
    position_embedding_table : nn.Embedding (block_size, n_embd)
    blocks                   : nn.Sequential of n_layer Block modules
    ln_f                     : nn.LayerNorm — final normalisation
    lm_head                  : nn.Linear (n_embd → vocab_size, bias=False)
    """

    def __init__(self, config: Config, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f   = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Initialise weights following GPT-2 paper conventions
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply GPT-2-style weight initialisation.

        * Linear layers: N(0, 0.02)
        * Embedding layers: N(0, 0.02)
        * Biases: zero
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[list[KVCache]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[list[KVCache]]]:
        """Compute logits and, optionally, cross-entropy loss.

        Parameters
        ----------
        idx     : torch.Tensor  shape (B, T)
            Integer token IDs. T must be ≤ block_size.
        targets : torch.Tensor  shape (B, T) | None
            Next-token targets. If None, only logits are returned.
        kv_caches : list[list[KVCache]] | None
            Optional pre-existing key-value caches for fast autoregressive generation.
        use_cache : bool
            If True, initialize (if None) and update kv_caches, returning them.

        Returns
        -------
        logits : torch.Tensor  shape (B, T, vocab_size)
        loss   : torch.Tensor scalar | None
        kv_caches : (only if use_cache=True) updated caches.
        """
        B, T = idx.shape
        device = idx.device

        if use_cache and kv_caches is None:
            kv_caches = []
            head_size = self.config.n_embd // self.config.n_head
            for _ in range(self.config.n_layer):
                layer_cache = []
                for _ in range(self.config.n_head):
                    layer_cache.append(KVCache(
                        key=torch.empty((B, 0, head_size), device=device),
                        value=torch.empty((B, 0, head_size), device=device),
                    ))
                kv_caches.append(layer_cache)

        # Determine starting position based on cache length
        start_pos = 0
        if kv_caches is not None:
            start_pos = kv_caches[0][0].key.shape[1]

        # Token + positional embeddings (SDD §9, GPTModel forward)
        tok_emb = self.token_embedding_table(idx)                              # (B, T, C)
        pos_ids = torch.arange(start_pos, start_pos + T, device=device)        # (T,)
        pos_emb = self.position_embedding_table(pos_ids)                       # (T, C)
        x = tok_emb + pos_emb                                                  # (B, T, C)

        # N Transformer blocks
        for i, block in enumerate(self.blocks):
            if kv_caches is not None:
                x = block(x, kv_caches[i])
            else:
                x = block(x)

        # Final layer norm + project to vocabulary
        x = self.ln_f(x)                    # (B, T, C)
        logits = self.lm_head(x)            # (B, T, vocab_size)

        # Compute loss if targets are provided (FR-13)
        loss: torch.Tensor | None = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        if use_cache:
            return logits, loss, kv_caches
        return logits, loss

    # ------------------------------------------------------------------
    # Parameter count utility
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total (trainable) parameter count.

        Parameters
        ----------
        trainable_only : bool
            If True (default), count only parameters with requires_grad=True.

        Returns
        -------
        int
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GPTModel("
            f"vocab_size={self.vocab_size}, "
            f"n_embd={self.config.n_embd}, "
            f"n_head={self.config.n_head}, "
            f"n_layer={self.config.n_layer}, "
            f"block_size={self.config.block_size}, "
            f"params={self.num_parameters():,})"
        )
