"""
tests/test_kv_cache.py
----------------------
Test suite for KV Cache generation equivalence.
"""

import pytest
import torch

from src.config import get_config
from src.model import GPTModel

@pytest.fixture
def model():
    cfg = get_config("tiny")
    cfg.n_layer = 1
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.block_size = 16
    
    m = GPTModel(cfg, vocab_size=64)
    m.eval()
    return m

class TestKVCache:
    def test_cache_equivalence(self, model):
        """Autoregressive generation with and without cache must yield identical logits."""
        B, T_init = 1, 4
        idx = torch.randint(0, 64, (B, T_init))
        
        # Uncached forward
        logits_uncached, _ = model(idx)
        
        # Cached forward: first pass builds cache
        logits_cached, _, kv_caches = model(idx, use_cache=True)
        
        # Logits should be identical
        torch.testing.assert_close(logits_uncached, logits_cached)
        
        # Generate one more token
        next_id = torch.randint(0, 64, (B, 1))
        idx_next = torch.cat([idx, next_id], dim=1)
        
        # Uncached forward for next token (T=5)
        logits_uncached_next, _ = model(idx_next)
        
        # Cached forward for next token (T=1, using cache)
        logits_cached_next, _, kv_caches_updated = model(next_id, kv_caches=kv_caches, use_cache=True)
        
        # The logits for the *new* token must match
        # uncached returns shape (B, 5, vocab), cached returns (B, 1, vocab)
        torch.testing.assert_close(logits_uncached_next[:, -1:, :], logits_cached_next)
        
        # Verify cache sizes
        cache_len = kv_caches_updated[0][0].key.shape[1]
        assert cache_len == 5  # 4 from prompt + 1 from next_id
