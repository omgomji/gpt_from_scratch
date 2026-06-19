"""
tests/test_dataset.py
---------------------
Test suite for DatasetBuilder and get_batch — corresponds to T-2, T-3, T-4.

T-2: Split sizes ≈ 90/10
T-3: shape(xb) == shape(yb) == [B, T]
T-4: y[i] == x[i+1] (next-token alignment)
"""

import pytest
import torch

from src.tokenizer import CharTokenizer
from src.dataset import DatasetBuilder, get_batch


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORPUS = "abcdefghijklmnopqrstuvwxyz " * 200   # 5 400 characters


@pytest.fixture(scope="module")
def tokenizer():
    tok = CharTokenizer()
    tok.build_vocab(CORPUS)
    return tok


@pytest.fixture(scope="module")
def dataset(tokenizer):
    return DatasetBuilder(CORPUS, tokenizer)


BATCH_SIZE  = 16
BLOCK_SIZE  = 8
DEVICE      = torch.device("cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetBuilder:

    # ── T-2: 90/10 split ─────────────────────────────────────────────────

    def test_split_total_equals_corpus_length(self, dataset, tokenizer):
        """train_data + val_data must cover the full encoded corpus."""
        n_train = len(dataset.train_data)
        n_val   = len(dataset.val_data)
        full    = len(tokenizer.encode(CORPUS))
        assert n_train + n_val == full

    def test_train_fraction_is_ninety_percent(self, dataset, tokenizer):
        """Train split must be exactly 90% of corpus (floor)."""
        full    = len(tokenizer.encode(CORPUS))
        n_train = len(dataset.train_data)
        expected_n = int(0.9 * full)
        assert n_train == expected_n

    def test_val_fraction_is_ten_percent(self, dataset, tokenizer):
        """Val split must be the remaining 10% of the corpus."""
        full  = len(tokenizer.encode(CORPUS))
        n_val = len(dataset.val_data)
        expected_n = full - int(0.9 * full)
        assert n_val == expected_n

    def test_train_data_is_long_tensor(self, dataset):
        assert dataset.train_data.dtype == torch.long

    def test_val_data_is_long_tensor(self, dataset):
        assert dataset.val_data.dtype == torch.long

    def test_vocab_size_property(self, dataset, tokenizer):
        assert dataset.vocab_size == tokenizer.vocab_size

    def test_split_size_train(self, dataset):
        assert dataset.split_size("train") == len(dataset.train_data)

    def test_split_size_val(self, dataset):
        assert dataset.split_size("val") == len(dataset.val_data)

    def test_invalid_split_raises(self, dataset):
        with pytest.raises(ValueError, match="'train' or 'val'"):
            dataset._get_split("test")


class TestGetBatch:

    # ── T-3: Batch shape ─────────────────────────────────────────────────

    def test_xb_shape(self, dataset):
        """T-3: shape(xb) must be (B, T)."""
        xb, _ = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert xb.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_yb_shape(self, dataset):
        """T-3: shape(yb) must be (B, T)."""
        _, yb = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert yb.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_xb_and_yb_same_shape(self, dataset):
        """T-3: xb and yb must have identical shapes."""
        xb, yb = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert xb.shape == yb.shape

    # ── T-4: Next-token alignment ─────────────────────────────────────────

    def test_y_is_x_shifted_by_one(self, dataset):
        """T-4: yb[b, t] must equal xb[b, t+1] for every t < T-1."""
        xb, yb = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        # For every row and every position except the last in x:
        assert torch.all(yb[:, :-1] == xb[:, 1:]), (
            "yb[b, t] should equal xb[b, t+1] for all t < T-1"
        )

    def test_batch_elements_are_long_tensors(self, dataset):
        """Batch tensors must use torch.long dtype."""
        xb, yb = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert xb.dtype == torch.long
        assert yb.dtype == torch.long

    def test_get_batch_val_split(self, dataset):
        """get_batch should work for the val split too."""
        xb, yb = get_batch("val", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert xb.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_values_within_vocab_range(self, dataset):
        """All token IDs in the batch must be valid vocabulary indices."""
        xb, yb = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        assert xb.min().item() >= 0
        assert xb.max().item() < dataset.vocab_size
        assert yb.min().item() >= 0
        assert yb.max().item() < dataset.vocab_size

    def test_get_batch_nondeterministic(self, dataset):
        """Two successive batches should (very likely) differ."""
        xb1, _ = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        xb2, _ = get_batch("train", dataset, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        # With 5 400-token corpus this will not be equal except with negligible probability
        assert not torch.all(xb1 == xb2)
