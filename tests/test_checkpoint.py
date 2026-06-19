"""
tests/test_checkpoint.py
------------------------
Test suite for checkpoint save/load round-trips.

T-11: save → load must reproduce identical logits
"""

import os
import pytest
import torch

from src.config import get_config
from src.model import GPTModel
from src.tokenizer import CharTokenizer
from src.dataset import DatasetBuilder
from src.trainer import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 27
CORPUS     = "abcdefghijklmnopqrstuvwxyz " * 200
SEED       = 99


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
        max_iters=5,
        eval_interval=5,
        eval_iters=2,
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu",   # force CPU so tests are device-agnostic
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpoint:

    # ── T-11: Save → load → identical logits ─────────────────────────────

    def test_save_load_identical_logits(self, cfg, dataset, tmp_path):
        """T-11: After loading a checkpoint the model must produce the same logits.

        Steps:
        1. Train for a few steps (moves weights away from init).
        2. Save checkpoint.
        3. Build a *fresh* model with the same architecture.
        4. Load the checkpoint into the fresh model.
        5. Assert that forward passes on the same input yield identical logits.
        """
        torch.manual_seed(SEED)
        model1  = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer = Trainer(model1, cfg, dataset)
        trainer.train()

        path = trainer.save_checkpoint(tag="t11")
        assert os.path.isfile(path), f"Checkpoint file not created: {path}"

        # Fresh model — random weights
        torch.manual_seed(SEED + 1)
        model2 = GPTModel(cfg, vocab_size=VOCAB_SIZE)

        # Sanity: before loading, logits differ
        # Use an input idx on the same device as the models
        device = cfg.resolved_device()
        idx = torch.zeros((1, cfg.block_size), dtype=torch.long, device=device)
        with torch.no_grad():
            logits1_pre, _ = model1(idx)
            logits2_pre, _ = model2(idx)
        # They should differ (different weights)
        assert not torch.allclose(logits1_pre, logits2_pre), (
            "Pre-load logits are identical — test setup is wrong."
        )

        # Load checkpoint into model2
        state = torch.load(path, map_location="cpu")
        model2.load_state_dict(state["model_state"])

        # Post-load logits must be bitwise identical
        with torch.no_grad():
            logits1_post, _ = model1(idx)
            logits2_post, _ = model2(idx)

        assert torch.allclose(logits1_post, logits2_post, atol=1e-6), (
            "Logits after checkpoint load do not match original model."
        )

    def test_checkpoint_contains_required_keys(self, cfg, dataset, tmp_path):
        """Checkpoint dict must contain model_state, optimizer_state, step, config_preset."""
        torch.manual_seed(SEED)
        model   = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer = Trainer(model, cfg, dataset)
        trainer.train()
        path  = trainer.save_checkpoint(tag="keys")
        state = torch.load(path, map_location="cpu")

        for key in ("model_state", "optimizer_state", "step", "config_preset"):
            assert key in state, f"Missing key '{key}' in checkpoint."

    def test_checkpoint_step_matches_trainer(self, cfg, dataset, tmp_path):
        """Saved step number must equal trainer.step at save time."""
        torch.manual_seed(SEED)
        model   = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer = Trainer(model, cfg, dataset)
        trainer.train()
        path  = trainer.save_checkpoint(tag="step_check")
        state = torch.load(path, map_location="cpu")
        assert state["step"] == trainer.step

    def test_load_checkpoint_restores_step(self, cfg, dataset, tmp_path):
        """load_checkpoint() must restore the step counter correctly."""
        torch.manual_seed(SEED)
        model1  = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer1 = Trainer(model1, cfg, dataset)
        trainer1.train()
        path = trainer1.save_checkpoint(tag="restore_step")

        model2   = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        trainer2 = Trainer(model2, cfg, dataset)
        loaded_step = trainer2.load_checkpoint(path)

        assert loaded_step == trainer1.step
        assert trainer2.step == trainer1.step
