"""
trainer.py
----------
Training loop, validation, and checkpointing (Module 10 from SDD §5.1).

Satisfies:
    FR-14 — training loop (forward / backward / optimizer step)
    FR-15 — periodic train/val loss reporting
    FR-16 — checkpoint save and load

Design
------
The Trainer owns the outer training loop described in SDD §8.1::

    for step in range(max_iters):
        xb, yb = get_batch("train", ...)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            report(train_loss, val_loss)
            save_checkpoint()

Validation is estimated by averaging the loss over eval_iters random batches
*without gradient tracking* (torch.no_grad) and in eval mode (dropout off),
as specified in SDD §10 / NFR-5.

Checkpoints
-----------
Each checkpoint is a dict saved with torch.save containing:
    - model state dict
    - optimizer state dict
    - current step number
    - training/val loss at that step
    - configuration preset name

Resume
------
Call ``Trainer.load_checkpoint(path)`` before ``train()`` to resume from a
previous run. The step counter and optimizer state are restored.

Usage
-----
    from src.config import get_config
    from src.model import GPTModel
    from src.dataset import DatasetBuilder, get_batch
    from src.trainer import Trainer

    cfg     = get_config("small")
    dataset = DatasetBuilder(text, tokenizer)
    model   = GPTModel(cfg, vocab_size=dataset.vocab_size).to(device)
    trainer = Trainer(model, cfg, dataset)
    trainer.train()
"""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.nn as nn

from src.config import Config
from src.dataset import DatasetBuilder, get_batch
from src.lr_scheduler import get_lr


class Trainer:
    """Manages the full training lifecycle of a GPTModel.

    Parameters
    ----------
    model   : nn.Module
        The model to train (GPTModel or BigramLanguageModel).
    config  : Config
        Hyperparameter bundle.
    dataset : DatasetBuilder
        Provides train and validation token tensors.
    on_log  : Callable[[int, float, float], None] | None
        Optional callback invoked at each evaluation step with
        (step, train_loss, val_loss). Defaults to a stdout logger.

    Attributes
    ----------
    optimizer : torch.optim.AdamW
        AdamW optimizer (standard for GPT-style models).
    device    : torch.device
        Resolved target device.
    step      : int
        Current training step (0-based); updated during training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        dataset: DatasetBuilder,
        on_log: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.model   = model
        self.config  = config
        self.dataset = dataset
        self.device  = config.resolved_device()
        self.step    = 0

        # Setup model compilation
        if config.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Move model to target device immediately
        self.model.to(self.device)

        # Weight decay group separation (GPT-2 convention)
        decay_params    = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in self.model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay_params,    "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate)

        # Setup AMP Scaler
        use_scaler = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        self._on_log = on_log or self._default_log



        # Training history (step → {train_loss, val_loss})
        self.history: list[dict] = []

        # Ensure checkpoint directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main training loop (FR-14)
    # ------------------------------------------------------------------

    def train(self) -> list[dict]:
        """Run the full training loop.

        Returns
        -------
        list[dict]
            Training history — one entry per evaluation step, each dict
            containing keys: ``step``, ``train_loss``, ``val_loss``.
        """
        cfg   = self.config
        model = self.model
        # Model already on device (moved in __init__)

        # ── Initial evaluation before training begins ─────────────────
        train_loss, val_loss = self._estimate_loss()
        self._on_log(0, train_loss, val_loss)
        self.history.append({"step": 0, "train_loss": train_loss, "val_loss": val_loss})

        t_start = time.perf_counter()

        for step in range(1, cfg.max_iters + 1):
            self.step = step

            # ── LR Scheduler ──────────────────────────────────────────
            lr = get_lr(step, cfg.max_iters, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # ── Training step (with Gradient Accumulation & AMP) ──────
            model.train()
            self.optimizer.zero_grad(set_to_none=True)
            
            for _ in range(cfg.grad_accum_steps):
                xb, yb = get_batch("train", self.dataset, cfg.batch_size, cfg.block_size, self.device)
                
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.autocast(device_type=device_type, enabled=cfg.use_amp):
                    _, loss = model(xb, yb)
                    loss = loss / cfg.grad_accum_steps
                    
                self.scaler.scale(loss).backward()
                
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Periodic evaluation (FR-15) ───────────────────────────
            if step % cfg.eval_interval == 0:
                train_loss, val_loss = self._estimate_loss()
                elapsed = time.perf_counter() - t_start
                self._on_log(step, train_loss, val_loss)
                self.history.append(
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "elapsed_s": round(elapsed, 1),
                    }
                )
                # Save checkpoint at every eval point (FR-16)
                self.save_checkpoint(tag=f"step{step:06d}")

        # Final checkpoint
        self.save_checkpoint(tag="final")
        return self.history

    # ------------------------------------------------------------------
    # Loss estimation (FR-15)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _estimate_loss(self) -> tuple[float, float]:
        """Estimate train and validation loss without updating gradients.

        Averages loss over ``config.eval_iters`` random batches for each split.
        Uses ``model.eval()`` so that dropout is disabled (NFR-5 / T-9).

        Returns
        -------
        (train_loss, val_loss) : tuple[float, float]
        """
        cfg   = self.config
        model = self.model
        model.eval()   # Dropout OFF during evaluation (T-9)

        results: dict[str, float] = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                xb, yb = get_batch(split, self.dataset, cfg.batch_size, cfg.block_size, self.device)
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                with torch.autocast(device_type=device_type, enabled=cfg.use_amp):
                    _, loss = model(xb, yb)
                losses[k] = loss.item()
            results[split] = losses.mean().item()

        model.train()  # Restore training mode
        return results["train"], results["val"]

    # ------------------------------------------------------------------
    # Checkpointing (FR-16)
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "checkpoint") -> str:
        """Save model + optimizer state to disk.

        Parameters
        ----------
        tag : str
            Filename suffix, e.g. ``"step001000"`` or ``"final"``.

        Returns
        -------
        str
            Absolute path of the saved checkpoint file.
        """
        path = os.path.join(self.config.checkpoint_dir, f"{tag}.pt")
        state = {
            "step":           self.step,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config_preset":  self.config.preset,
            "history":        self.history,
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        """Resume training from a previously saved checkpoint.

        Restores model weights, optimizer state, and the step counter.

        Parameters
        ----------
        path : str
            Path to a ``.pt`` checkpoint file.

        Returns
        -------
        int
            The step number at which the checkpoint was saved.
        """
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.step = state["step"]
        self.history = state.get("history", [])
        return self.step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_log(step: int, train_loss: float, val_loss: float) -> None:
        print(f"step {step:6d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
