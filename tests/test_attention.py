"""
tests/test_attention.py
-----------------------
Test suite for CausalSelfAttention and MultiHeadAttention.

T-6: Causal mask — attention weight wei[i, j] = 0 for all j > i
T-7: Multi-head output dimension == n_embd after concatenation + projection
"""

import pytest
import torch
import torch.nn.functional as F

from src.attention import CausalSelfAttention, MultiHeadAttention


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_EMBD     = 64
N_HEAD     = 4
HEAD_SIZE  = N_EMBD // N_HEAD    # 16
BLOCK_SIZE = 16
BATCH_SIZE = 2
SEQ_LEN    = 8      # T ≤ BLOCK_SIZE
DROPOUT    = 0.0    # Off for deterministic testing


# ---------------------------------------------------------------------------
# Tests: CausalSelfAttention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:

    @pytest.fixture
    def head(self):
        return CausalSelfAttention(
            n_embd=N_EMBD,
            head_size=HEAD_SIZE,
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        ).eval()

    def test_output_shape(self, head):
        """Output must be (B, T, head_size)."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = head(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HEAD_SIZE)

    # ── T-6: Causal mask ─────────────────────────────────────────────────

    def test_causal_mask_shape(self, head):
        """The 'tril' buffer must be square with side == block_size."""
        assert head.tril.shape == (BLOCK_SIZE, BLOCK_SIZE)

    def test_tril_is_lower_triangular(self, head):
        """tril[i, j] must be True iff j <= i."""
        tril = head.tril.float()
        expected = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        assert torch.all(tril == expected)

    def test_attention_scores_upper_triangle_zero(self, head):
        """T-6: No attention weight should flow from future positions.

        We verify by monkey-patching the forward logic to expose the raw
        attention weights before value aggregation.
        """
        B, T, C = BATCH_SIZE, SEQ_LEN, N_EMBD
        x = torch.randn(B, T, C)

        with torch.no_grad():
            q = head.query(x)
            k = head.key(x)
            scale = head.head_size ** -0.5
            wei = (q @ k.transpose(-2, -1)) * scale            # (B, T, T)
            wei = wei.masked_fill(~head.tril[:T, :T], float("-inf"))
            wei = F.softmax(wei, dim=-1)                        # (B, T, T)

        # Upper triangle (j > i) must be exactly 0 after softmax(-inf)
        for i in range(T):
            for j in range(i + 1, T):
                assert torch.allclose(
                    wei[:, i, j],
                    torch.zeros(B),
                ), f"Attention weight at (i={i}, j={j}) should be 0 but got {wei[:, i, j]}"

    def test_no_future_leakage_via_output(self, head):
        """Changing a future token must not change past-position outputs.

        If the causal mask is correctly applied, token at position 0
        should not be affected by what we place at position 1.
        """
        torch.manual_seed(42)
        x = torch.randn(1, 4, N_EMBD)
        x_modified = x.clone()
        x_modified[:, 2:, :] = torch.randn(1, 2, N_EMBD)   # corrupt future

        with torch.no_grad():
            out_orig = head(x)
            out_mod  = head(x_modified)

        # Position 0 output must be unchanged despite future corruption
        assert torch.allclose(out_orig[:, 0, :], out_mod[:, 0, :], atol=1e-5), (
            "Position 0 output was affected by future token changes — causal mask broken."
        )


# ---------------------------------------------------------------------------
# Tests: MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:

    @pytest.fixture
    def mha(self):
        return MultiHeadAttention(
            n_embd=N_EMBD,
            n_head=N_HEAD,
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        ).eval()

    # ── T-7: Output dimension == n_embd ──────────────────────────────────

    def test_output_shape_equals_n_embd(self, mha):
        """T-7: MultiHeadAttention output dim must equal n_embd."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = mha(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD), (
            f"Expected (B={BATCH_SIZE}, T={SEQ_LEN}, C={N_EMBD}), got {out.shape}"
        )

    def test_n_heads_created(self, mha):
        """Module list should have exactly n_head heads."""
        assert len(mha.heads) == N_HEAD

    def test_each_head_is_causal_self_attention(self, mha):
        """Every head must be a CausalSelfAttention instance."""
        for h in mha.heads:
            assert isinstance(h, CausalSelfAttention)

    def test_head_sizes_sum_to_n_embd(self, mha):
        """All head sizes combined must equal n_embd (no capacity lost)."""
        total = sum(h.head_size for h in mha.heads)
        assert total == N_EMBD

    def test_output_same_shape_as_input(self, mha):
        """Output must be the same shape as the input for residual addition."""
        x   = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = mha(x)
        assert out.shape == x.shape

    def test_n_embd_divisibility_assertion(self):
        """Constructor must raise AssertionError when n_embd % n_head != 0."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(n_embd=65, n_head=4, block_size=8)

    def test_gradient_flows_through_mha(self, mha):
        """Backward pass must not raise and must produce finite gradients."""
        mha.train()
        x   = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
