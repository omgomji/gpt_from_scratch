"""
tests/test_model.py
-------------------
Test suite for BigramLanguageModel and GPTModel.

T-5: Bigram initial loss ≈ -ln(1/vocab_size)  (random-baseline entropy)
T-8: GPTModel can overfit a tiny batch → loss approaches 0
"""

import math

import pytest
import torch

from src.bigram import BigramLanguageModel
from src.config import get_config
from src.model import GPTModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 65      # Tiny Shakespeare char count
BATCH_SIZE = 4
BLOCK_SIZE = 8
SEED       = 1337


# ---------------------------------------------------------------------------
# Tests: BigramLanguageModel
# ---------------------------------------------------------------------------

class TestBigramLanguageModel:

    @pytest.fixture
    def bigram(self):
        torch.manual_seed(SEED)
        return BigramLanguageModel(vocab_size=VOCAB_SIZE)

    # ── T-5: Initial loss ≈ -ln(1/V) ─────────────────────────────────────

    def test_initial_loss_near_random_baseline(self, bigram):
        """T-5: Untrained bigram loss must ≈ -ln(1/vocab_size).

        Before any gradient updates, the model has near-uniform weights,
        so the loss should be close to the entropy of a uniform distribution:
            H = -ln(1/vocab_size) = ln(vocab_size)
        """
        torch.manual_seed(SEED)
        idx     = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        _, loss = bigram(idx, targets)

        expected = math.log(VOCAB_SIZE)   # ln(65) ≈ 4.17
        # Default nn.Embedding uses N(0,1) which pushes loss slightly above
        # the theoretical uniform-distribution baseline; tolerance of 1.0
        # still catches any badly-broken initialisation while being realistic.
        assert abs(loss.item() - expected) < 1.0, (
            f"Initial loss {loss.item():.3f} is far from expected {expected:.3f}. "
            f"Check that weights are near-uniform at initialisation."
        )

    def test_forward_output_shapes(self, bigram):
        """forward() must return logits (B, T, V) and a scalar loss."""
        idx     = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        logits, loss = bigram(idx, targets)
        assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
        assert loss.ndim == 0   # scalar

    def test_forward_no_targets_returns_none_loss(self, bigram):
        idx = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        logits, loss = bigram(idx)
        assert loss is None
        assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)

    def test_generate_returns_extended_sequence(self, bigram):
        """generate() must return a tensor with T + max_new_tokens columns."""
        idx = torch.zeros((1, 1), dtype=torch.long)
        new_tokens = 10
        out = bigram.generate(idx, max_new_tokens=new_tokens)
        assert out.shape == (1, 1 + new_tokens)

    def test_generate_values_in_vocab_range(self, bigram):
        idx = torch.zeros((1, 1), dtype=torch.long)
        out = bigram.generate(idx, max_new_tokens=20)
        assert out.min().item() >= 0
        assert out.max().item() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# Tests: GPTModel
# ---------------------------------------------------------------------------

class TestGPTModel:

    @pytest.fixture
    def cfg(self):
        return get_config("tiny", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    @pytest.fixture
    def model(self, cfg):
        torch.manual_seed(SEED)
        return GPTModel(cfg, vocab_size=VOCAB_SIZE)

    def test_forward_output_shapes(self, model):
        """GPTModel.forward must return (B, T, V) logits and scalar loss."""
        idx     = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        targets = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        logits, loss = model(idx, targets)
        assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
        assert loss.ndim == 0

    def test_forward_no_targets_returns_none_loss(self, model):
        idx = torch.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=torch.long)
        logits, loss = model(idx)
        assert loss is None

    def test_num_parameters_positive(self, model):
        assert model.num_parameters() > 0

    def test_num_parameters_all_greater_than_trainable(self, model):
        assert model.num_parameters(trainable_only=False) >= model.num_parameters()

    # ── T-8: Overfit tiny batch ───────────────────────────────────────────

    def test_overfit_tiny_batch(self, cfg):
        """T-8: GPTModel must drive loss close to 0 on a fixed 8-token batch.

        This validates that the full forward + backward + optimizer wiring is
        correct. A model that cannot overfit a trivially small dataset has a
        bug somewhere in the gradient path.
        """
        torch.manual_seed(SEED)
        model     = GPTModel(cfg, vocab_size=VOCAB_SIZE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Tiny deterministic batch — 2 sequences of 8 tokens
        idx     = torch.randint(VOCAB_SIZE, (2, BLOCK_SIZE))
        targets = torch.randint(VOCAB_SIZE, (2, BLOCK_SIZE))

        initial_loss = None
        for step in range(500):
            model.train()
            _, loss = model(idx, targets)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 0.5, (
            f"Model failed to overfit tiny batch: loss={final_loss:.4f} "
            f"(initial={initial_loss:.4f}). Check backprop wiring."
        )

    def test_initial_gpt_loss_near_random_baseline(self, model):
        """GPTModel initial loss should also be near ln(vocab_size)."""
        torch.manual_seed(SEED)
        idx     = torch.randint(VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        targets = torch.randint(VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
        _, loss = model(idx, targets)

        expected = math.log(VOCAB_SIZE)
        # Larger tolerance than bigram because weight init std=0.02 shifts it slightly
        assert abs(loss.item() - expected) < 1.5, (
            f"GPTModel initial loss {loss.item():.3f} far from expected {expected:.3f}"
        )
