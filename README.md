# GPT from Scratch V2

> A character-level and subword-level decoder-only Transformer built from first principles — following the architecture of GPT, trained on Tiny Shakespeare, WikiText, and more.

**B.E. CSE (AIML) · Chandigarh University**  
Inspired by Andrej Karpathy's *Let's Build GPT* & *nanoGPT*

---

## Overview

This project implements a GPT-style language model **entirely from scratch** using PyTorch. 

Version 2 introduces significant upgrades for scale, speed, and usability:
- **BPE Tokenization** (Byte-Pair Encoding for semantic compression)
- **KV Caching** (O(1) generation step instead of O(T))
- **Advanced Training** (Gradient Accumulation, Mixed Precision, Cosine LR)
- **Attention Visualization** (Extracting and plotting attention heatmaps)
- **Interactive Web Dashboard** (Gradio interface for generation, monitoring, and visualization)

The architecture supports multiple configs from a trivial bigram to a full multi-layer Transformer with causal self-attention, positional embeddings, residual connections, and layer normalisation.

---

## Project Structure

```
gpt_from_scratch/
├── data/                       # Downloaded corpora (Shakespeare, WikiText, Gutenberg)
├── checkpoints/                # Saved model checkpoints (.pt) and BPE merges
├── src/
│   ├── config.py               # Hyperparameter configs & training presets
│   ├── tokenizer.py            # CharTokenizer & BaseTokenizer interface
│   ├── bpe_tokenizer.py        # BPE implementation
│   ├── dataset.py              # DatasetBuilder + get_batch
│   ├── bigram.py               # BigramLanguageModel
│   ├── attention.py            # CausalSelfAttention + MultiHeadAttention
│   ├── feedforward.py          # FeedForward network
│   ├── block.py                # Transformer Block
│   ├── model.py                # GPTModel (full stack)
│   ├── kv_cache.py             # Key-Value Cache structures for fast generation
│   ├── trainer.py              # Trainer (loop, eval, checkpointing)
│   ├── lr_scheduler.py         # Cosine annealing learning rate
│   ├── generator.py            # Autoregressive generation
│   └── attention_viz.py        # Attention weights extraction and visualization
├── tests/                      # Pytest suite
├── download_data.py            # Fetch various text corpora
├── train.py                    # CLI training entry point
├── generate.py                 # CLI generation entry point
├── app.py                      # Gradio Interactive Web Dashboard
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Corpora

```bash
python download_data.py --corpus all
```
*Options: shakespeare, wikitext, gutenberg, all.*

### 3. Interactive Web Dashboard

Launch the Gradio web interface to play with trained models:
```bash
python app.py
```
Open `http://localhost:7860` in your browser.

### 4. Train via CLI

```bash
# Stage 4 — Full Transformer (small preset, CharTokenizer)
python train.py --config small

# Stage 5 — BPE Tokenizer, Mixed Precision, Gradient Accumulation
python train.py --config large --tokenizer bpe --use-amp --grad-accum 4
```

### 5. Generate via CLI

```bash
# Generate text using cache
python generate.py --checkpoint checkpoints/final.pt --prompt "KING LEAR:" --max-tokens 300
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

---

## V2 Upgrades Detail

### 1. Byte-Pair Encoding (BPE)
Subword tokenization compresses sequences, effectively increasing the semantic length the model can process within its context window. Integrated seamlessly via a common `BaseTokenizer` protocol.

### 2. Key-Value Cache (KV Cache)
During generation, attention keys and values from previous tokens are cached. This drops the time complexity of generating the Nth token from O(N) to O(1), making autoregressive generation vastly faster.

### 3. Training Optimizations
- **Automatic Mixed Precision (AMP):** Utilizes Tensor Cores for massive speedups on modern GPUs.
- **Gradient Accumulation:** Allows training with large effective batch sizes even on low-VRAM GPUs.
- **Cosine Learning Rate Scheduler:** Includes a linear warmup to stabilize early training.
- **`torch.compile`**: JIT kernel fusion for execution speedups (PyTorch 2.0+).

### 4. Visualizations & Interactivity
The Gradio dashboard provides a user-friendly interface. A dedicated visualization module extracts the internal attention matrices to show exactly how tokens attend to each other across layers and heads.

---

## Testing

Comprehensive tests cover every component, ensuring mathematically correct implementations of components such as attention masks, loss calculations, cache updating equivalence, and learning rate scheduling.

| Component | Assertions |
|-----------|-----------|
| Tokenizers | Encode/decode roundtrips, BPE pair merging |
| KV Cache | Uncached output == Cached output |
| Generator | Context window constraints |
| Attention | Causality (upper triangle == 0) |

---

## License

MIT License — Educational use.
