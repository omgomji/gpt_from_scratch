"""
tests/test_generator.py
-----------------------
Test suite for the Generator class.

T-10: len(idx_cond) ≤ block_size at every generation step (context crop)
"""

import pytest
import torch

from src.config import get_config
from src.model import GPTModel
from src.tokenizer import CharTokenizer
from src.generator import Generator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 65
BLOCK_SIZE = 16
SEED       = 42


@pytest.fixture
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;")
    return tok


@pytest.fixture
def model():
    cfg = get_config("tiny", block_size=BLOCK_SIZE)
    torch.manual_seed(SEED)
    m = GPTModel(cfg, vocab_size=VOCAB_SIZE)
    m.eval()
    return m


@pytest.fixture
def generator(model, tokenizer):
    return Generator(model, tokenizer, block_size=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerator:

    # ── T-10: Context crop ────────────────────────────────────────────────

    def test_context_never_exceeds_block_size(self, model, tokenizer):
        """T-10: idx_cond passed to model must never be longer than block_size.

        We instrument model.forward to capture the length of each call's idx.
        """
        seen_lengths: list[int] = []
        original_forward = model.forward

        def instrumented_forward(idx, targets=None):
            seen_lengths.append(idx.shape[1])   # T dimension
            return original_forward(idx, targets)

        model.forward = instrumented_forward   # type: ignore[method-assign]

        gen = Generator(model, tokenizer, block_size=BLOCK_SIZE)
        # Use a seed longer than block_size to force cropping
        long_prompt = "a" * (BLOCK_SIZE + 5)
        gen.generate(prompt=long_prompt, max_new_tokens=10)

        model.forward = original_forward       # type: ignore[method-assign]

        assert len(seen_lengths) > 0
        for length in seen_lengths:
            assert length <= BLOCK_SIZE, (
                f"Generator passed idx of length {length} to model, "
                f"which exceeds block_size={BLOCK_SIZE}."
            )

    def test_generate_returns_string(self, generator):
        """generate() must return a str."""
        out = generator.generate(max_new_tokens=20)
        assert isinstance(out, str)

    def test_generate_length_at_least_prompt_plus_new(self, generator):
        """Output must contain at least the new tokens (may include prompt)."""
        max_new = 15
        out = generator.generate(prompt="a", max_new_tokens=max_new)
        # Decoded output has at least max_new characters (+ possible prompt chars)
        assert len(out) >= max_new

    def test_generate_empty_prompt(self, generator):
        """Generating from an empty prompt should not raise."""
        out = generator.generate(prompt="", max_new_tokens=5)
        assert isinstance(out, str)

    def test_temperature_zero_is_deterministic(self, generator):
        """Greedy decoding (temperature=0) must be deterministic."""
        out1 = generator.generate(max_new_tokens=20, temperature=0.0)
        out2 = generator.generate(max_new_tokens=20, temperature=0.0)
        assert out1 == out2

    def test_temperature_one_produces_valid_chars(self, generator, tokenizer):
        """All generated characters must be in the tokenizer vocabulary."""
        out = generator.generate(max_new_tokens=50, temperature=1.0)
        for ch in out:
            assert ch in tokenizer.stoi, f"'{ch}' not in vocabulary"

    def test_generate_zero_new_tokens(self, generator, tokenizer):
        """Generating 0 new tokens should return only the prompt (decoded seed)."""
        prompt = "abc"
        out = generator.generate(prompt=prompt, max_new_tokens=0)
        # Output should at least start with the prompt
        assert out.startswith(prompt)

    def test_max_new_tokens_respected(self, generator):
        """Output must have exactly max_new_tokens more chars than the single seed."""
        max_new = 30
        # Empty prompt → seed is token [0], 1 char decoded
        out = generator.generate(prompt="", max_new_tokens=max_new)
        # seed length=1 + max_new tokens
        assert len(out) == 1 + max_new
