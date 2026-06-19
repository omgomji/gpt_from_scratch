"""
dataset.py
----------
Data pipeline — Modules 2 (DatasetBuilder) and 3 (BatchSampler) from SDD §5.1.

Satisfies: FR-5 (train/val split 90/10), FR-6 (mini-batch sampling)

Overview
--------
1. DatasetBuilder reads a plain-text corpus, tokenizes it, and produces
   train / validation split tensors.
2. get_batch() samples a random mini-batch of (x, y) pairs where
   y[t] = x[t+1] — the standard next-token prediction target.

Usage
-----
    from src.tokenizer import CharTokenizer
    from src.dataset import DatasetBuilder, get_batch
    import torch

    tokenizer = CharTokenizer()
    with open("data/input.txt") as f:
        text = f.read()
    tokenizer.build_vocab(text)

    builder = DatasetBuilder(text, tokenizer)
    xb, yb = get_batch("train", builder, batch_size=32, block_size=64,
                        device=torch.device("cpu"))
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch

from src.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Module 2 — DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Reads a plain-text corpus and prepares train / validation tensors.

    Attributes
    ----------
    train_data : torch.Tensor  shape (N_train,)
        Integer token IDs for the first 90 % of the corpus (FR-5).
    val_data   : torch.Tensor  shape (N_val,)
        Integer token IDs for the remaining 10 % of the corpus (FR-5).
    tokenizer  : CharTokenizer
        The tokenizer used for encoding — stored for external access.

    Parameters
    ----------
    text      : str
        Raw UTF-8 corpus text.
    tokenizer : CharTokenizer
        A tokenizer that has already had :meth:`build_vocab` called.
    split_ratio : float
        Fraction of corpus used for training (default 0.9 → 90/10 split).
    """

    TRAIN_SPLIT: float = 0.9  # SDD §5.1, Module 2 — "first 90 % train"

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        split_ratio: float = TRAIN_SPLIT,
    ) -> None:
        self.tokenizer = tokenizer

        # Encode the full corpus as a 1-D integer tensor (FR-3)
        ids: list[int] = tokenizer.encode(text)
        data = torch.tensor(ids, dtype=torch.long)

        # 90/10 split (FR-5)
        n = int(split_ratio * len(data))
        self.train_data: torch.Tensor = data[:n]
        self.val_data: torch.Tensor = data[n:]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Convenience access to vocabulary size."""
        return self.tokenizer.vocab_size

    def split_size(self, split: Literal["train", "val"]) -> int:
        """Return the number of tokens in the given split."""
        return len(self._get_split(split))

    def _get_split(self, split: Literal["train", "val"]) -> torch.Tensor:
        if split == "train":
            return self.train_data
        if split == "val":
            return self.val_data
        raise ValueError(f"split must be 'train' or 'val', got '{split}'.")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DatasetBuilder("
            f"train={len(self.train_data):,} tokens, "
            f"val={len(self.val_data):,} tokens, "
            f"vocab_size={self.vocab_size})"
        )


# ---------------------------------------------------------------------------
# Module 3 — BatchSampler (functional)
# ---------------------------------------------------------------------------

def get_batch(
    split: Literal["train", "val"],
    dataset: DatasetBuilder,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random mini-batch of (x, y) pairs from *split* (FR-6).

    For each of the *batch_size* rows, a random start offset ``i`` is chosen
    such that ``i + block_size < len(data)``, then::

        x = data[i     : i + block_size]      # input context
        y = data[i + 1 : i + block_size + 1]  # target — shifted by one (FR-6)

    Parameters
    ----------
    split      : "train" or "val"
    dataset    : DatasetBuilder instance (provides the split tensors)
    batch_size : number of independent sequences per batch (B)
    block_size : maximum context window length (T)
    device     : target torch.device for the output tensors

    Returns
    -------
    xb : torch.Tensor  shape (B, T)   — input token IDs
    yb : torch.Tensor  shape (B, T)   — next-token target IDs
    """
    data = dataset._get_split(split)

    # Random starting offsets — one per row in the batch
    offsets = torch.randint(len(data) - block_size, (batch_size,))

    # Stack into (B, T) tensors (T-3, T-4)
    xb = torch.stack([data[i     : i + block_size    ] for i in offsets])
    yb = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])

    return xb.to(device), yb.to(device)
