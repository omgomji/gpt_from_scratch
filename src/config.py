"""
config.py
---------
Hyperparameter configuration for GPT-from-Scratch.

Provides a dataclass-based Config object and named presets
corresponding to each training stage in the SDD (§7).

Presets
-------
    bigram  — Stage 1: baseline bigram language model
    tiny    — Stage 2-3: single/multi-head attention, fits on CPU
    small   — Stage 4: full Transformer block
    medium  — Stage 5: scaled model, GPU recommended

Usage
-----
    from src.config import get_config
    cfg = get_config("small")
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    """Hyperparameter bundle for a single training run.

    Attributes
    ----------
    preset       : human-readable name of the configuration preset
    block_size   : maximum context length (T) — aka sequence length
    n_embd       : token/positional embedding dimension (C)
    n_head       : number of parallel attention heads
    n_layer      : number of stacked Transformer blocks
    dropout      : dropout probability (0.0 = disabled)
    batch_size   : number of independent sequences per mini-batch (B)
    max_iters    : total training steps
    eval_interval: steps between validation-loss evaluations
    eval_iters   : batches averaged per validation estimate (reduces variance)
    learning_rate: AdamW learning rate
    seed         : random seed for reproducibility (NFR-1)
    device       : PyTorch device string; "auto" auto-selects CUDA/CPU (NFR-2)
    checkpoint_dir: directory where checkpoints are saved (FR-16)
    data_path    : path to plain-text corpus file (FR-1)
    """

    preset: str = "small"

    # Model architecture
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    max_iters: int = 5_000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4

    # System
    seed: int = 1337
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    data_path: str = "data/input.txt"

    # -----------------------------------------------------------------------
    # Resolved device (lazy property)
    # -----------------------------------------------------------------------

    def resolved_device(self) -> torch.device:
        """Return the actual torch.device, resolving "auto" at call time."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def __str__(self) -> str:  # pragma: no cover
        lines = [f"Config(preset='{self.preset}')"]
        for f in dataclasses.fields(self):
            if f.name == "preset":
                continue
            lines.append(f"  {f.name:20s} = {getattr(self, f.name)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Named presets (SDD §7 — Training Strategy)
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    # ── Stage 1: Bigram baseline ──────────────────────────────────────────
    # Proves the data pipeline and loss/generation loop work.
    "bigram": dict(
        preset="bigram",
        block_size=8,
        n_embd=32,          # unused in BigramLM but kept for compatibility
        n_head=1,
        n_layer=1,
        dropout=0.0,
        batch_size=32,
        max_iters=10_000,
        eval_interval=1_000,
        eval_iters=200,
        learning_rate=1e-3,
    ),

    # ── Stage 2-3: Single/Multi-head attention — CPU-friendly ─────────────
    # Introduces context-aware self-attention with a small footprint.
    "tiny": dict(
        preset="tiny",
        block_size=32,
        n_embd=64,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        batch_size=32,
        max_iters=5_000,
        eval_interval=500,
        eval_iters=200,
        learning_rate=1e-3,
    ),

    # ── Stage 4: Full Transformer block — modest GPU or fast CPU ─────────
    # Adds LayerNorm + residuals + FFN; first fully capable model.
    "small": dict(
        preset="small",
        block_size=64,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
        batch_size=32,
        max_iters=5_000,
        eval_interval=500,
        eval_iters=200,
        learning_rate=3e-4,
    ),

    # ── Stage 5: Scaled model — GPU strongly recommended ─────────────────
    # Matches the nanoGPT-style scaling described in the SDD.
    "medium": dict(
        preset="medium",
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.2,
        batch_size=64,
        max_iters=5_000,
        eval_interval=500,
        eval_iters=200,
        learning_rate=3e-4,
    ),
}

PresetName = Literal["bigram", "tiny", "small", "medium"]


def get_config(preset: PresetName = "small", **overrides) -> Config:
    """Return a Config for the named preset, with optional field overrides.

    Parameters
    ----------
    preset   : one of "bigram", "tiny", "small", "medium"
    **overrides : keyword arguments forwarded to Config (override preset defaults)

    Returns
    -------
    Config

    Raises
    ------
    ValueError if preset name is not recognised.

    Examples
    --------
    >>> cfg = get_config("small")
    >>> cfg_gpu = get_config("medium", device="cuda:0", max_iters=10_000)
    """
    if preset not in _PRESETS:
        valid = ", ".join(f'"{k}"' for k in _PRESETS)
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}.")

    kwargs = {**_PRESETS[preset], **overrides}
    return Config(**kwargs)


def list_presets() -> list[str]:
    """Return all available preset names."""
    return list(_PRESETS.keys())
