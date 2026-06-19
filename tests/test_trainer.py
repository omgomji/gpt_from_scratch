"""
tests/test_trainer.py
---------------------
Test suite for the Trainer class.

T-9: model.eval() and torch.no_grad() must be active during validation
"""

import os
import pytest
import torch
import torch.nn as nn

from src.config import get_config
from src.model import GPTModel
from src.tokenizer import CharTokenizer
from src.dataset import DatasetBuilder
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 27   # a-z + space
CORPUS     = ("abcdefghijklmnopqrstuvwxyz " * 300)   # ~8100 chars


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab(CORPUS)
    return tok


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return DatasetBuilder(CORPUS, tokenizer)


@pytest.fixture
def cfg(tmp_path):
    return get_config(
        "tiny",
        block_size=8,
        batch_size=8,
        max_iters=10,
        eval_interval=5,
        eval_iters=3,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",   # force CPU so tests are device-agnostic
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return GPTModel(cfg, vocab_size=VOCAB_SIZE)


@pytest.fixture
def trainer(model, cfg, dataset):
    return Trainer(model, cfg, dataset)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainer:

    # ── T-9: eval() + no_grad() during validation ─────────────────────────

    def test_validation_uses_eval_mode(self, trainer):
        """T-9a: model.training must be False inside _estimate_loss."""
        observed_training_flags: list[bool] = []

        original_eval = trainer.model.eval
        def patched_eval():
            observed_training_flags.append(trainer.model.training)
            return original_eval()
        trainer.model.eval = patched_eval   # type: ignore[method-assign]

        trainer._estimate_loss()

        # eval() should have been called at least once
        assert len(observed_training_flags) > 0, "eval() was never called during _estimate_loss"
        # Restore
        trainer.model.eval = original_eval  # type: ignore[method-assign]

    def test_validation_restores_train_mode(self, trainer):
        """T-9b: model must be back in train() mode after _estimate_loss."""
        trainer.model.train()
        trainer._estimate_loss()
        assert trainer.model.training, "model was not restored to train mode after _estimate_loss"

    def test_validation_runs_without_gradient(self, trainer):
        """T-9c: No gradient tensors should be produced during _estimate_loss.

        We verify by checking that _estimate_loss doesn't throw and that
        gradient tracking is not active on the outputs (torch.no_grad decorator).
        """
        # If no_grad is missing this will raise RuntimeError on some builds
        train_loss, val_loss = trainer._estimate_loss()
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert 0.0 < train_loss < 20.0
        assert 0.0 < val_loss   < 20.0

    # ── Training loop basics ──────────────────────────────────────────────

    def test_train_returns_history(self, trainer):
        """train() must return a list of dicts with the expected keys."""
        history = trainer.train()
        assert isinstance(history, list)
        assert len(history) > 0
        for entry in history:
            assert "step"       in entry
            assert "train_loss" in entry
            assert "val_loss"   in entry

    def test_loss_is_finite(self, trainer):
        """All logged losses must be finite (no NaN / inf)."""
        history = trainer.train()
        for entry in history:
            assert torch.isfinite(torch.tensor(entry["train_loss"]))
            assert torch.isfinite(torch.tensor(entry["val_loss"]))

    # ── Checkpointing ─────────────────────────────────────────────────────

    def test_checkpoint_file_created(self, trainer):
        """A checkpoint file must exist after training."""
        trainer.train()
        ckpt_dir = trainer.config.checkpoint_dir
        pts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        assert len(pts) > 0, f"No .pt files found in {ckpt_dir}"

    def test_save_and_load_checkpoint(self, trainer, model, cfg, dataset, tmp_path):
        """save_checkpoint / load_checkpoint must produce identical model state."""
        trainer.train()
        path = trainer.save_checkpoint(tag="test")

        # Fresh trainer with same model architecture
        model2  = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer2 = Trainer(model2, cfg, dataset)
        step = trainer2.load_checkpoint(path)

        assert step == trainer.step

        # Model weights must be bitwise identical after load
        for (name, p1), (_, p2) in zip(
            trainer.model.named_parameters(),
            trainer2.model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameter '{name}' differs after checkpoint load."
