"""
generator.py
------------
Autoregressive text generator (Module 11 from SDD §5.1).

Satisfies: FR-17 (generate text autoregressively), FR-18 (crop context to block_size)

Generation loop (SDD §5.4 / §8.2)
----------------------------------
    idx = seed_tokens                              # (1, T_seed)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]            # crop to block_size (FR-18)
        logits, _ = model(idx_cond)                # forward, no grad
        logits = logits[:, -1, :]                  # last time-step: (1, vocab_size)
        probs = softmax(logits, dim=-1)            # probability distribution
        next_id = multinomial(probs, 1)            # sample one token
        idx = cat([idx, next_id], dim=1)           # append (B, T+1)
    return decode(idx[0].tolist())

Temperature
-----------
An optional temperature parameter (default 1.0) scales the logits before
softmax, controlling generation randomness:
    - temperature < 1.0  → more conservative, peaked distribution
    - temperature > 1.0  → more diverse / random
    - temperature = 0.0  → greedy (argmax)

Usage
-----
    from src.generator import Generator

    gen  = Generator(model, tokenizer, block_size=64)
    text = gen.generate(prompt="KING:", max_new_tokens=500)
    print(text)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tokenizer import CharTokenizer


class Generator:
    """Autoregressive text generator wrapping a trained language model.

    Parameters
    ----------
    model      : nn.Module
        A trained GPTModel (or BigramLanguageModel) in eval-capable state.
    tokenizer  : CharTokenizer
        Tokenizer matching the one used during training.
    block_size : int
        Maximum context window (T) — contexts longer than this are cropped
        before each forward pass (FR-18 / T-10).
    device     : torch.device | None
        Target device. Defaults to the device of the first model parameter.

    Notes
    -----
    * The generator always operates in ``model.eval()`` mode with
      ``torch.no_grad()`` to avoid gradient computation and ensure dropout
      is disabled (NFR-5).
    * Temperature = 0.0 triggers greedy (argmax) decoding rather than
      multinomial sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CharTokenizer,
        block_size: int,
        device: torch.device | None = None,
    ) -> None:
        self.model      = model
        self.tokenizer  = tokenizer
        self.block_size = block_size

        if device is None:
            device = next(model.parameters()).device
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 500,
        temperature: float = 1.0,
        use_cache: bool = True,
    ) -> str:
        """Generate text autoregressively starting from *prompt*.

        Parameters
        ----------
        prompt         : str
            Seed text. An empty string starts generation from a zero-context
            (the model may still produce coherent text due to positional
            embeddings).
        max_new_tokens : int
            Number of new characters to generate.
        temperature    : float
            Sampling temperature. 1.0 = neutral, < 1.0 = focused,
            > 1.0 = creative. 0.0 = greedy argmax.
        use_cache      : bool
            If True, use KV caching for fast generation (default True).

        Returns
        -------
        str
            Full generated text (prompt + *max_new_tokens* new characters).
        """
        # Encode prompt (or start from token 0 if empty)
        if prompt:
            seed_ids = self.tokenizer.encode(prompt)
        else:
            seed_ids = [0]

        idx = torch.tensor([seed_ids], dtype=torch.long, device=self.device)

        idx = self._generate_ids(idx, max_new_tokens, temperature, use_cache)
        return self.tokenizer.decode(idx[0].tolist())

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_ids(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Inner generation loop operating on token-ID tensors.

        Parameters
        ----------
        idx            : torch.Tensor  shape (B, T)
        max_new_tokens : int
        temperature    : float

        Returns
        -------
        torch.Tensor  shape (B, T + max_new_tokens)
        """
        self.model.eval()
        
        is_gpt = type(self.model).__name__ == "GPTModel"
        kv_caches = None

        for _ in range(max_new_tokens):
            if use_cache and is_gpt:
                cache_len = kv_caches[0][0].key.shape[1] if kv_caches is not None else 0
                if cache_len >= self.block_size:
                    kv_caches = None  # Flush cache if we hit context limit
                
                if kv_caches is None:
                    idx_cond = idx[:, -self.block_size:]
                    logits, _, kv_caches = self.model(idx_cond, use_cache=True)
                else:
                    next_id = idx[:, -1:]
                    logits, _, kv_caches = self.model(next_id, kv_caches=kv_caches, use_cache=True)
            else:
                # ── Crop context to block_size (FR-18 / T-10) ─────────────
                idx_cond = idx[:, -self.block_size:]
                # ── Forward pass (eval mode, no grad) ─────────────────────
                logits, _ = self.model(idx_cond)

            # ── Focus on last time-step's predictions ─────────────────
            logits = logits[:, -1, :]   # (B, vocab_size)

            # ── Apply temperature scaling ──────────────────────────────
            if temperature == 0.0:
                # Greedy decoding — pick the highest-probability token
                next_id = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                logits = logits / temperature
                probs  = F.softmax(logits, dim=-1)             # (B, vocab_size)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # ── Append sampled token ───────────────────────────────────
            idx = torch.cat([idx, next_id], dim=1)  # (B, T+1)

        return idx
