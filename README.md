# GPT from Scratch

> A character-level decoder-only Transformer built from first principles — following the architecture of GPT, trained on Tiny Shakespeare.

**B.E. CSE (AIML) · Chandigarh University**  
Inspired by Andrej Karpathy's *Let's Build GPT* & *nanoGPT*

---

## Overview

This project implements a GPT-style language model **entirely from scratch** using PyTorch — no pre-built Transformer libraries. The model is trained on the Tiny Shakespeare corpus (~1M characters) and generates new Shakespeare-like text character by character.

The implementation follows a **5-stage progressive architecture**, starting from a trivial bigram model and culminating in a full multi-layer Transformer with causal self-attention, positional embeddings, residual connections, and layer normalisation.

```
Raw Text Corpus
    │
    ▼
[CharTokenizer] ── build vocab · encode · decode
    │
    ▼
[DatasetBuilder] ── 90/10 train/val split
    │
    ├──► Train ──► [BatchSampler] ──► (xb, yb) ──► [GPTModel.forward]
    │                                                        │
    │                                               Logits + CE Loss
    │                                                        │
    │                                          Backprop ──► Optimizer.step()
    │
    └──► Val ──► [BatchSampler] ──► (xb, yb) ──► [GPTModel.forward, no_grad] ──► Val Loss

Seed Context ──► [Generator] ──► crop(block_size) ──► [GPTModel.forward] ──► softmax
                     ▲                                                              │
                     └─────────────────── append sampled token ◄── multinomial ◄───┘
```

---

## Project Structure

```
gpt_from_scratch/
├── Documentations/             # SRS, SDD, combined spec
├── data/
│   └── input.txt               # Tiny Shakespeare corpus
├── checkpoints/                # saved model checkpoints (.pt)
├── src/
│   ├── config.py               # hyperparameter configs & training-stage presets
│   ├── tokenizer.py            # Module 1: CharTokenizer (encode/decode)
│   ├── dataset.py              # Module 2+3: DatasetBuilder + get_batch
│   ├── bigram.py               # Module 4: BigramLanguageModel (baseline)
│   ├── attention.py            # Module 5+6: CausalSelfAttention + MultiHeadAttention
│   ├── feedforward.py          # Module 7: FeedForward (per-token MLP)
│   ├── block.py                # Module 8: Transformer Block (pre-norm + residual)
│   ├── model.py                # Module 9: GPTModel (full stack)
│   ├── trainer.py              # Module 10: Trainer (loop, eval, checkpointing)
│   └── generator.py            # Module 11: Generator (autoregressive sampling)
├── tests/                      # pytest test suite (T-1 through T-11)
├── download_data.py            # fetch Tiny Shakespeare corpus
├── train.py                    # CLI training entry point
├── generate.py                 # CLI generation entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Corpus

```bash
python download_data.py
```

This fetches ~1 MB of Tiny Shakespeare text from Karpathy's char-rnn repository and saves it to `data/input.txt`.

### 3. Train

```bash
# Stage 1 — Bigram baseline (~10 000 steps, runs on CPU in ~2 min)
python train.py --config bigram

# Stage 4 — Full Transformer (small preset, ~5 000 steps)
python train.py --config small

# Stage 5 — Scaled model (GPU recommended)
python train.py --config medium

# Resume from a checkpoint
python train.py --config small --resume checkpoints/step002500.pt
```

### 4. Generate Text

```bash
# Generate 500 characters from the final checkpoint
python generate.py --checkpoint checkpoints/final.pt --max-tokens 500

# With a seed prompt
python generate.py --checkpoint checkpoints/final.pt --prompt "KING LEAR:" --max-tokens 300

# Greedy (deterministic) decoding
python generate.py --checkpoint checkpoints/final.pt --temperature 0
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Architecture

### Module Overview

| # | Module | File | Purpose | Satisfies |
|---|--------|------|---------|-----------|
| 1 | CharTokenizer | `src/tokenizer.py` | Build vocab · encode · decode | FR-1 to FR-4 |
| 2 | DatasetBuilder | `src/dataset.py` | 90/10 train/val split | FR-5 |
| 3 | BatchSampler | `src/dataset.py` | Random (x,y) mini-batches | FR-6 |
| 4 | BigramLanguageModel | `src/bigram.py` | Next-token lookup baseline | FR-7 |
| 5 | CausalSelfAttention | `src/attention.py` | QKV · scaled dot-product · causal mask | FR-8 |
| 6 | MultiHeadAttention | `src/attention.py` | H parallel heads · concat · project | FR-9 |
| 7 | FeedForward | `src/feedforward.py` | Per-token 2-layer MLP (4× expand) | FR-10 |
| 8 | Block | `src/block.py` | Pre-norm + residual around Attn & FFN | FR-11 |
| 9 | GPTModel | `src/model.py` | tok-emb + pos-emb → N×Block → lm_head | FR-12, FR-13 |
| 10 | Trainer | `src/trainer.py` | Training loop · eval · checkpointing | FR-14 to FR-16 |
| 11 | Generator | `src/generator.py` | Autoregressive sampling (with crop) | FR-17, FR-18 |

### Transformer Block (Pre-norm Architecture)

```
Input x  (B, T, C)
    │
    ├──► LayerNorm ──► MultiHeadAttention ──► + ──► x  (residual add)
    │                                          ▲
    │                                          │
    └─────────────────────────────────────────┘
    │
    ├──► LayerNorm ──► FeedForward ──────────► + ──► x  (residual add)
    │                                          ▲
    │                                          │
    └─────────────────────────────────────────┘
    │
    ▼
Output x  (B, T, C)
```

### Class Hierarchy

```
GPTModel
 ├─ TokenEmbeddingTable      nn.Embedding (vocab_size, n_embd)
 ├─ PositionalEmbeddingTable nn.Embedding (block_size, n_embd)
 ├─ Block[ ]  ×  n_layer
 │    ├─ LayerNorm (ln1)
 │    ├─ MultiHeadAttention
 │    │     └─ CausalSelfAttention  ×  n_head
 │    ├─ LayerNorm (ln2)
 │    └─ FeedForward
 ├─ LayerNorm (ln_f)
 └─ Linear (lm_head, vocab_size)
```

---

## Training Strategy (5 Stages)

| Stage | Config Preset | Architecture | Goal |
|-------|--------------|--------------|------|
| 1 | `bigram` | Token lookup → logits | Verify pipeline; establish baseline |
| 2–3 | `tiny` | + Single/Multi-head causal self-attention | Context-aware predictions |
| 4 | `small` | + LayerNorm + Residuals + FFN | Stable deep stack |
| 5 | `medium` | ↑ block_size, n_embd, n_head, n_layer + Dropout | Near nanoGPT scale |

---

## Hyperparameter Presets

| Preset | block_size | n_embd | n_head | n_layer | dropout | batch_size | max_iters | lr |
|--------|-----------|--------|--------|---------|---------|------------|-----------|-----|
| `bigram` | 8 | 32 | 1 | 1 | 0.0 | 32 | 10 000 | 1e-3 |
| `tiny` | 32 | 64 | 4 | 2 | 0.0 | 32 | 5 000 | 1e-3 |
| `small` | 64 | 128 | 4 | 4 | 0.1 | 32 | 5 000 | 3e-4 |
| `medium` | 256 | 384 | 6 | 6 | 0.2 | 64 | 5 000 | 3e-4 |

---

## Complexity Analysis

Let B = batch size, T = block\_size, C = n\_embd, H = n\_head, L = n\_layer, V = vocab\_size.

| Component | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Bigram | O(B·T·V) | O(V·C) |
| Single-head attention | O(B·T²·C) | O(B·T²) |
| Multi-head attention | O(B·T²·C) × H constant | O(B·T²) |
| Transformer (per layer) | O(B·T²·C + B·T·C²) | — |
| Transformer (L layers) | O(L·(B·T²·C + B·T·C²)) | — |
| Generation (N tokens) | O(N·L·(T²·C + T·C²)) | — |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Decoder-only** | Unconditional text imitation — no conditioning source to encode |
| **Character-level tokenizer** | Simplest path to learning the architecture; BPE as future work |
| **Causal mask** | Autoregressive LM requires no future-token leakage |
| **Pre-norm (LayerNorm before sub-layer)** | Better gradient flow than post-norm; required for n\_layer > 2 |
| **Residual connections** | Prevent vanishing gradients in deeper stacks |
| **Dropout** | Regularisation for larger configs (Stage 5) |
| **90/10 train/val split** | Minimal viable split to monitor overfitting on small corpus |
| **AdamW optimizer** | Standard for Transformer training; decoupled weight decay |

---

## Testing

All 11 test cases from the SDD test plan are implemented in `tests/`:

| Test ID | File | Assertion |
|---------|------|-----------|
| T-1 | `test_tokenizer.py` | `decode(encode(x)) == x` |
| T-2 | `test_dataset.py` | Split sizes ≈ 90/10 |
| T-3 | `test_dataset.py` | `shape(xb) == shape(yb) == [B, T]` |
| T-4 | `test_dataset.py` | `y[i] == x[i+1]` |
| T-5 | `test_model.py` | Bigram init loss ≈ `ln(vocab_size)` |
| T-6 | `test_attention.py` | Causal mask: `wei[i,j] = 0` for `j > i` |
| T-7 | `test_attention.py` | MHA output dim == n\_embd |
| T-8 | `test_model.py` | Overfit tiny batch → loss → ~0 |
| T-9 | `test_trainer.py` | `eval()` + `no_grad()` used for validation |
| T-10 | `test_generator.py` | Context crop ≤ block\_size always |
| T-11 | `test_checkpoint.py` | Save → load reproduces identical logits |

---

## Limitations & Future Scope

**Current Limitations**
- Character-level modelling is statistically inefficient for semantics
- Tiny Shakespeare is too small for general language competence
- No distributed/multi-node training
- No alignment/RLHF stage — output is a document-completer, not an assistant

**Future Scope**
- Subword tokenizer (BPE / SentencePiece)
- RoPE / ALiBi positional encodings
- Flash-Attention / fused kernels
- Mixed precision (fp16 / bf16) training
- Longer context windows
- Instruction-tuning stage

---

## References

1. Vaswani et al., *Attention Is All You Need* (2017)
2. Karpathy, *nanoGPT* — https://github.com/karpathy/nanoGPT
3. Karpathy, *Let's Build GPT from Scratch* (YouTube)
4. Tiny Shakespeare dataset — https://github.com/karpathy/char-rnn

---

## License

MIT License — Educational use.
