"""
generate.py
-----------
Standalone text generation entry point — load a checkpoint and generate.

Usage
-----
    # Generate 500 characters from a saved checkpoint
    python generate.py --checkpoint checkpoints/final.pt --max-tokens 500

    # Provide a seed prompt
    python generate.py --checkpoint checkpoints/final.pt --prompt "KING LEAR:"

    # Greedy decoding (deterministic)
    python generate.py --checkpoint checkpoints/final.pt --temperature 0

    # Creative / high-temperature sampling
    python generate.py --checkpoint checkpoints/final.pt --temperature 1.5
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from src.config import get_config
from src.model import GPTModel
from src.bigram import BigramLanguageModel
from src.tokenizer import CharTokenizer
from src.generator import Generator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-from-Scratch — Text Generation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,               help="Path to .pt checkpoint file.")
    parser.add_argument("--config",     default="small",             help="Preset used when the model was trained.")
    parser.add_argument("--data-path",  default="data/input.txt",   help="Corpus used to rebuild vocabulary.")
    parser.add_argument("--prompt",     default="",                  help="Seed text for generation.")
    parser.add_argument("--max-tokens", type=int, default=500,       help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0,   help="Sampling temperature (0 = greedy).")
    parser.add_argument("--device",     default="auto",              help='Target device: "cpu", "cuda", or "auto".')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: '{args.checkpoint}'")
        sys.exit(1)

    # ── Rebuild vocabulary from corpus ────────────────────────────────────
    if not os.path.exists(args.data_path):
        print(f"[ERROR] Corpus not found at '{args.data_path}'. Run download_data.py first.")
        sys.exit(1)
    with open(args.data_path, encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)

    # ── Load config & build model skeleton ───────────────────────────────
    cfg = get_config(args.config)
    if args.device != "auto":
        cfg.device = args.device
    device = cfg.resolved_device()

    if cfg.preset == "bigram":
        model: torch.nn.Module = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    else:
        model = GPTModel(cfg, vocab_size=tokenizer.vocab_size)

    # ── Load checkpoint weights ───────────────────────────────────────────
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from '{args.checkpoint}' (step {state.get('step', '?')})")
    print(f"Generating {args.max_tokens} tokens  |  temperature={args.temperature}\n")
    print("-" * 60)

    # ── Generate ──────────────────────────────────────────────────────────
    gen = Generator(model, tokenizer, block_size=cfg.block_size, device=device)
    output = gen.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
