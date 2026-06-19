"""
train.py
--------
Command-line training entry point for GPT-from-Scratch.

Orchestrates the full pipeline:
    1. Load corpus from disk (FR-1)
    2. Build character vocabulary (FR-2)
    3. Build dataset with 90/10 split (FR-5)
    4. Initialise model (bigram or GPT depending on config preset)
    5. Train with periodic validation and checkpointing (FR-14, FR-15, FR-16)
    6. Generate a short sample after training (FR-17)

Usage
-----
    # Stage 1 — bigram baseline (~10 000 steps, fast on CPU)
    python train.py --config bigram

    # Stage 4 — full Transformer, small preset
    python train.py --config small

    # Resume from a previous checkpoint
    python train.py --config small --resume checkpoints/step002000.pt

    # Override any hyperparameter
    python train.py --config medium --max-iters 10000 --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from src.config import get_config, list_presets
from src.dataset import DatasetBuilder
from src.bigram import BigramLanguageModel
from src.model import GPTModel
from src.tokenizer import CharTokenizer
from src.trainer import Trainer
from src.generator import Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_corpus(path: str) -> str:
    if not os.path.exists(path):
        print(
            f"[ERROR] Corpus not found at '{path}'.\n"
            f"        Run: python download_data.py"
        )
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-from-Scratch — Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        choices=list_presets(),
        default="small",
        help="Named hyperparameter preset (training stage).",
    )
    parser.add_argument("--data-path",      default="data/input.txt",  help="Path to corpus text file.")
    parser.add_argument("--checkpoint-dir", default="checkpoints",     help="Directory for checkpoint files.")
    parser.add_argument("--resume",         default=None,              help="Path to checkpoint to resume from.")
    parser.add_argument("--max-iters",      type=int, default=None,    help="Override max training iterations.")
    parser.add_argument("--batch-size",     type=int, default=None,    help="Override batch size.")
    parser.add_argument("--seed",           type=int, default=None,    help="Override random seed.")
    parser.add_argument("--device",         default=None,              help='Override device, e.g. "cpu", "cuda".')
    parser.add_argument("--gen-tokens",     type=int, default=300,     help="Tokens to generate after training.")
    parser.add_argument("--gen-prompt",     default="",               help="Seed text for post-training generation.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Build config, applying any CLI overrides
    overrides: dict = {}
    if args.max_iters   is not None: overrides["max_iters"]      = args.max_iters
    if args.batch_size  is not None: overrides["batch_size"]     = args.batch_size
    if args.seed        is not None: overrides["seed"]           = args.seed
    if args.device      is not None: overrides["device"]         = args.device
    if args.data_path   != "data/input.txt": overrides["data_path"] = args.data_path
    overrides["checkpoint_dir"] = args.checkpoint_dir

    cfg = get_config(args.config, **overrides)
    device = cfg.resolved_device()

    _banner(f"GPT-from-Scratch  |  preset='{cfg.preset}'  |  device={device}")
    print(cfg)
    print()

    # Reproducibility (NFR-1)
    torch.manual_seed(cfg.seed)

    # ── Data pipeline ─────────────────────────────────────────────────────
    print("[1/5] Loading corpus...")
    text = _load_corpus(cfg.data_path)
    print(f"      {len(text):,} characters read from '{cfg.data_path}'")

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    print(f"      Vocabulary: {tokenizer.vocab_size} unique characters")

    dataset = DatasetBuilder(text, tokenizer)
    print(f"      Train tokens : {len(dataset.train_data):,}")
    print(f"      Val   tokens : {len(dataset.val_data):,}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    print("[2/5] Building model...")
    if cfg.preset == "bigram":
        model: torch.nn.Module = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
        print(f"      BigramLanguageModel  |  vocab_size={tokenizer.vocab_size}")
    else:
        model = GPTModel(cfg, vocab_size=tokenizer.vocab_size)
        print(f"      {model}")
    model.to(device)
    print()

    # ── Trainer ───────────────────────────────────────────────────────────
    print("[3/5] Initialising trainer...")
    trainer = Trainer(model, cfg, dataset)

    if args.resume:
        step = trainer.load_checkpoint(args.resume)
        print(f"      Resumed from '{args.resume}' at step {step}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[4/5] Training for {cfg.max_iters} steps "
          f"(eval every {cfg.eval_interval})...")
    print("-" * 60)
    t0 = time.perf_counter()
    history = trainer.train()
    elapsed = time.perf_counter() - t0
    print("-" * 60)
    print(f"\n      Training complete in {elapsed:.1f}s")
    final = history[-1]
    print(f"      Final  train_loss={final['train_loss']:.4f}  "
          f"val_loss={final['val_loss']:.4f}\n")

    # ── Generate ──────────────────────────────────────────────────────────
    print("[5/5] Generating sample text...")
    gen  = Generator(model, tokenizer, block_size=cfg.block_size, device=device)
    text_out = gen.generate(
        prompt=args.gen_prompt,
        max_new_tokens=args.gen_tokens,
        temperature=1.0,
    )
    print("-" * 60)
    print(text_out)
    print("-" * 60)
    print(f"\nDone. Checkpoints saved to '{cfg.checkpoint_dir}/'")


if __name__ == "__main__":
    main()
