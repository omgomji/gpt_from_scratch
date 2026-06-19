"""
tests/test_tokenizer.py
-----------------------
Test suite for CharTokenizer — corresponds to T-1 in the SDD test plan.

T-1: decode(encode(x)) == x  (exact round-trip for all corpus text)
"""

import pytest
from src.tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = "Hello, World! 123\nShakespeare wrote: 'To be or not to be.'"


@pytest.fixture
def tokenizer() -> CharTokenizer:
    tok = CharTokenizer()
    tok.build_vocab(SAMPLE_TEXT)
    return tok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCharTokenizer:

    def test_build_vocab_populates_stoi_and_itos(self, tokenizer):
        """Vocabulary must have stoi and itos mappings populated."""
        assert len(tokenizer.stoi) > 0
        assert len(tokenizer.itos) > 0
        assert tokenizer.vocab_size == len(tokenizer.stoi)

    def test_vocab_size_equals_unique_chars(self, tokenizer):
        """vocab_size must equal the number of unique characters in corpus."""
        expected = len(set(SAMPLE_TEXT))
        assert tokenizer.vocab_size == expected

    def test_stoi_and_itos_are_inverses(self, tokenizer):
        """stoi[ch] → id, itos[id] → ch must be mutually inverse."""
        for ch, idx in tokenizer.stoi.items():
            assert tokenizer.itos[idx] == ch

    def test_vocab_is_sorted_by_unicode(self, tokenizer):
        """Characters should be sorted by code-point for reproducibility (NFR-1)."""
        chars_in_order = [tokenizer.itos[i] for i in range(tokenizer.vocab_size)]
        assert chars_in_order == sorted(chars_in_order)

    # ── T-1: Round-trip ───────────────────────────────────────────────────

    def test_round_trip_encode_decode(self, tokenizer):
        """T-1: decode(encode(text)) must equal the original text exactly."""
        encoded = tokenizer.encode(SAMPLE_TEXT)
        decoded = tokenizer.decode(encoded)
        assert decoded == SAMPLE_TEXT

    def test_round_trip_single_char(self, tokenizer):
        """Round-trip must work for every individual character in vocabulary."""
        for ch in tokenizer.stoi:
            assert tokenizer.decode(tokenizer.encode(ch)) == ch

    def test_encode_returns_list_of_ints(self, tokenizer):
        """encode() must return List[int]."""
        result = tokenizer.encode("Hello")
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_decode_returns_str(self, tokenizer):
        """decode() must return str."""
        ids = tokenizer.encode("Hello")
        result = tokenizer.decode(ids)
        assert isinstance(result, str)

    def test_encode_length_equals_text_length(self, tokenizer):
        """Character-level: len(encode(text)) == len(text)."""
        # Use chars that are guaranteed to be in the fixture vocab
        # (SAMPLE_TEXT contains 'H', 'e', 'l', 'o')
        text = "Hello"
        assert len(tokenizer.encode(text)) == len(text)

    def test_empty_string_encodes_to_empty_list(self, tokenizer):
        """encode('') should return []."""
        assert tokenizer.encode("") == []

    def test_empty_list_decodes_to_empty_string(self, tokenizer):
        """decode([]) should return ''."""
        assert tokenizer.decode([]) == ""

    # ── Error conditions ──────────────────────────────────────────────────

    def test_encode_raises_before_build_vocab(self):
        """Calling encode before build_vocab should raise RuntimeError."""
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            tok.encode("test")

    def test_decode_raises_before_build_vocab(self):
        """Calling decode before build_vocab should raise RuntimeError."""
        tok = CharTokenizer()
        with pytest.raises(RuntimeError, match="build_vocab"):
            tok.decode([0])

    def test_encode_unknown_char_raises_key_error(self, tokenizer):
        """A character absent from the vocabulary should raise KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode("\x00")  # null byte not in SAMPLE_TEXT

    def test_repr(self):
        tok = CharTokenizer()
        tok.build_vocab("abc")
        assert "CharTokenizer" in repr(tok)
        assert "3" in repr(tok)
