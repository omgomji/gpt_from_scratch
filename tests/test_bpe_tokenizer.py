"""
tests/test_bpe_tokenizer.py
---------------------------
Test suite for the BPE tokenizer.
"""

import os
import pytest
from src.bpe_tokenizer import BPETokenizer

# Sample text for testing BPE merges
SAMPLE_TEXT = (
    "hello world hello world hello world "
    "this is a test this is a test "
    "byte pair encoding byte pair encoding"
)

@pytest.fixture
def tokenizer():
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=256 + 10)  # train 10 merges
    return tok

class TestBPETokenizer:
    def test_vocab_size(self, tokenizer):
        """Vocabulary size must equal 256 + number of merges."""
        assert tokenizer.vocab_size == 266
        assert len(tokenizer.merges) == 10

    def test_round_trip_encode_decode(self, tokenizer):
        """decode(encode(text)) == text for any text."""
        test_string = "hello this is a test of byte pair encoding!"
        ids = tokenizer.encode(test_string)
        decoded = tokenizer.decode(ids)
        assert decoded == test_string

    def test_encode_empty_string(self, tokenizer):
        """Empty string encodes to empty list."""
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_encode_single_char(self, tokenizer):
        """Single character encodes properly without merges."""
        ids = tokenizer.encode("a")
        assert len(ids) == 1
        assert tokenizer.decode(ids) == "a"

    def test_encode_unicode(self, tokenizer):
        """Unicode characters outside ascii should be preserved via utf-8 bytes."""
        text = "Hello 🌍! 안녕"
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text

    def test_save_and_load(self, tokenizer, tmp_path):
        """Tokenizer state can be saved and loaded accurately."""
        save_path = str(tmp_path / "bpe.json")
        tokenizer.save(save_path)
        assert os.path.exists(save_path)
        
        # New tokenizer instance
        tok_new = BPETokenizer()
        tok_new.load(save_path)
        
        assert tok_new.vocab_size == tokenizer.vocab_size
        assert tok_new.merges == tokenizer.merges
        assert tok_new.vocab == tokenizer.vocab
        
        text = "testing save and load functionality"
        assert tok_new.encode(text) == tokenizer.encode(text)
        assert tok_new.decode(tok_new.encode(text)) == text
