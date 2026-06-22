"""
app.py
------
Interactive Web Dashboard for GPT-from-Scratch V2.

Features:
- Tab 1: Text Generation Playground (streaming, KV cache toggle)
- Tab 2: Training Monitor (start/stop training, live loss plot, tokens/sec)
- Tab 3: Attention Visualizer (per-layer, per-head heatmaps)
"""

import os
import glob
import threading
import time
import math

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for threads
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gradio as gr

from src.config import get_config, list_presets
from src.model import GPTModel
from src.bigram import BigramLanguageModel
from src.tokenizer import CharTokenizer, BaseTokenizer
from src.bpe_tokenizer import BPETokenizer
from src.attention_viz import capture_attention_weights, plot_attention_heatmap
from src.dataset import DatasetBuilder, get_batch
from src.lr_scheduler import get_lr

# ---------------------------------------------------------------------------
# Shared globals
# ---------------------------------------------------------------------------

# ── Inference state ─────────────────────────────────────────────────────────
current_model = None
current_tokenizer = None
current_checkpoint = None
current_device = "cpu"

# ── Training monitor state ───────────────────────────────────────────────────
_train_lock = threading.Lock()
train_state = {
    "running": False,
    "stop_requested": False,
    "step": 0,
    "max_iters": 0,
    "history": [],          # list of {step, train_loss, val_loss}
    "status": "Idle",
    "start_time": None,
    "last_sample": "",
    "thread": None,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_checkpoints():
    """Return a sorted list of checkpoint paths (newest first)."""
    os.makedirs("checkpoints", exist_ok=True)
    ckpts = sorted(glob.glob("checkpoints/*.pt"), key=os.path.getmtime, reverse=True)
    return ckpts if ckpts else ["No checkpoints found"]


def _load_model_from_path(checkpoint_path):
    """Load model + tokenizer from checkpoint. Returns (model, tokenizer, device, msg)."""
    if not os.path.exists(checkpoint_path) or checkpoint_path == "No checkpoints found":
        return None, None, None, "Invalid checkpoint path."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    preset = state.get("config_preset", "small")
    cfg = get_config(preset)

    if cfg.tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
        bpe_path = os.path.join(os.path.dirname(checkpoint_path), "bpe.json")
        if os.path.exists(bpe_path):
            tokenizer.load(bpe_path)
        else:
            return None, None, None, f"BPE merges not found at '{bpe_path}'."
    else:
        tokenizer = CharTokenizer()
        corpus_path = "data/input.txt"
        if not os.path.exists(corpus_path):
            return None, None, None, f"Corpus not found at '{corpus_path}'."
        with open(corpus_path, encoding="utf-8") as f:
            tokenizer.build_vocab(f.read())

    if cfg.preset == "bigram":
        model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    else:
        model = GPTModel(cfg, vocab_size=tokenizer.vocab_size)

    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    msg = f"Loaded {os.path.basename(checkpoint_path)} | preset={preset} | vocab={tokenizer.vocab_size}"
    return model, tokenizer, device, msg


def _ensure_model_loaded(checkpoint_path):
    """Load model into globals if not already loaded from this path."""
    global current_model, current_tokenizer, current_checkpoint, current_device
    if checkpoint_path == current_checkpoint and current_model is not None:
        return f"Model already loaded: {os.path.basename(checkpoint_path)}"
    model, tokenizer, device, msg = _load_model_from_path(checkpoint_path)
    if model is None:
        return msg
    current_model, current_tokenizer = model, tokenizer
    current_device = device
    current_checkpoint = checkpoint_path
    return msg


# ---------------------------------------------------------------------------
# Tab 1 — Generation Playground
# ---------------------------------------------------------------------------

def generate_text(checkpoint, prompt, max_tokens, temperature, use_cache):
    status = _ensure_model_loaded(checkpoint)
    if current_model is None:
        yield status
        return

    is_gpt = isinstance(current_model, GPTModel)
    block_size = current_model.config.block_size if is_gpt else 8

    seed_ids = current_tokenizer.encode(prompt) if prompt else [0]
    idx = torch.tensor([seed_ids], dtype=torch.long, device=current_device)

    out_text = prompt
    yield out_text

    kv_caches = None
    current_model.eval()

    for _ in range(int(max_tokens)):
        with torch.no_grad():
            if use_cache and is_gpt:
                cache_len = kv_caches[0][0].key.shape[1] if kv_caches is not None else 0
                if cache_len >= block_size:
                    kv_caches = None
                if kv_caches is None:
                    idx_cond = idx[:, -block_size:]
                    logits, _, kv_caches = current_model(idx_cond, use_cache=True)
                else:
                    logits, _, kv_caches = current_model(idx[:, -1:], kv_caches=kv_caches, use_cache=True)
            else:
                idx_cond = idx[:, -block_size:]
                if is_gpt:
                    logits, _ = current_model(idx_cond)
                else:
                    logits, _ = current_model(idx_cond)

            logits = logits[:, -1, :]
            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            out_text += current_tokenizer.decode(next_id[0].tolist())
            yield out_text


# ---------------------------------------------------------------------------
# Tab 2 — Training Monitor
# ---------------------------------------------------------------------------

@torch.no_grad()
def _estimate_loss_quick(model, dataset, cfg, n_batches=50):
    """Quick loss estimate over n_batches for the monitor."""
    device = next(model.parameters()).device
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(n_batches):
            xb, yb = get_batch(split, dataset, cfg.batch_size, cfg.block_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results["train"], results["val"]


def _make_loss_plot(history):
    """Create a matplotlib Figure of the training curves."""
    if not history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Loss Curves")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        plt.tight_layout()
        return fig

    steps = [h["step"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, train_losses, label="Train Loss", color="#4f8ef7", linewidth=2)
    ax.plot(steps, val_losses,  label="Val Loss",   color="#f7784f", linewidth=2, linestyle="--")
    ax.set_title("Training Progress", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if steps:
        ax.set_xlim(left=0, right=max(steps) * 1.05)
    plt.tight_layout()
    return fig


def _training_worker(cfg, corpus_path, tokenizer_type):
    """Background thread: builds dataset + model and runs the training loop."""
    global train_state

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with _train_lock:
            train_state["status"] = "Loading corpus and tokenizer..."

        with open(corpus_path, encoding="utf-8") as f:
            text = f.read()

        if tokenizer_type == "bpe":
            tokenizer = BPETokenizer()
            tokenizer.train(text, vocab_size=cfg.bpe_vocab_size, verbose=False)
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            tokenizer.save(os.path.join(cfg.checkpoint_dir, "bpe.json"))
        else:
            tokenizer = CharTokenizer()
            tokenizer.build_vocab(text)

        dataset = DatasetBuilder(text, tokenizer)

        with _train_lock:
            train_state["status"] = f"Building model (vocab={tokenizer.vocab_size})..."

        model = GPTModel(cfg, vocab_size=tokenizer.vocab_size).to(device)

        # Optimiser with weight decay groups
        decay_p    = [p for p in model.parameters() if p.dim() >= 2]
        no_decay_p = [p for p in model.parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {"params": decay_p,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ], lr=cfg.learning_rate)

        use_scaler = cfg.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        with _train_lock:
            train_state["status"] = "Training..."
            train_state["start_time"] = time.perf_counter()

        for step in range(1, cfg.max_iters + 1):
            # Check stop flag
            with _train_lock:
                if train_state["stop_requested"]:
                    train_state["status"] = f"Stopped at step {step:,} / {cfg.max_iters:,}"
                    train_state["running"] = False
                    return

            # LR schedule
            lr = get_lr(step, cfg.max_iters, cfg.learning_rate, cfg.min_lr, cfg.warmup_iters)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation step
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum_steps):
                xb, yb = get_batch("train", dataset, cfg.batch_size, cfg.block_size, device)
                with torch.autocast(device_type=device.type, enabled=cfg.use_amp):
                    _, loss = model(xb, yb)
                    loss = loss / cfg.grad_accum_steps
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Periodic evaluation
            if step % cfg.eval_interval == 0:
                t_loss, v_loss = _estimate_loss_quick(model, dataset, cfg)
                elapsed = time.perf_counter() - train_state["start_time"]
                tok_per_sec = (step * cfg.batch_size * cfg.block_size * cfg.grad_accum_steps) / elapsed

                with _train_lock:
                    train_state["step"] = step
                    train_state["history"].append({
                        "step": step, "train_loss": t_loss, "val_loss": v_loss
                    })
                    train_state["status"] = (
                        f"Step {step:,}/{cfg.max_iters:,} | "
                        f"train={t_loss:.4f} val={v_loss:.4f} | "
                        f"lr={lr:.2e} | {tok_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed"
                    )

                # Save checkpoint
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"monitor_step{step:06d}.pt")
                torch.save({
                    "step": step, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config_preset": cfg.preset, "history": train_state["history"],
                }, ckpt_path)

        with _train_lock:
            train_state["status"] = f"✅ Training complete! Final val_loss={train_state['history'][-1]['val_loss']:.4f}"
            train_state["running"] = False

    except Exception as e:
        with _train_lock:
            train_state["status"] = f"❌ Error: {e}"
            train_state["running"] = False


def start_training(preset, corpus_path, tok_type, max_iters_override, batch_size_override, use_amp, grad_accum):
    global train_state
    with _train_lock:
        if train_state["running"]:
            return "Training is already running.", _make_loss_plot(train_state["history"])

    # Build config
    try:
        cfg = get_config(preset)
    except Exception as e:
        return f"Config error: {e}", _make_loss_plot([])

    if max_iters_override and int(max_iters_override) > 0:
        cfg.max_iters = int(max_iters_override)
    if batch_size_override and int(batch_size_override) > 0:
        cfg.batch_size = int(batch_size_override)
    cfg.use_amp = use_amp
    cfg.grad_accum_steps = int(grad_accum)
    cfg.eval_interval = max(50, cfg.max_iters // 20)

    corpus = corpus_path if corpus_path and os.path.exists(corpus_path) else "data/input.txt"

    with _train_lock:
        train_state.update({
            "running": True, "stop_requested": False,
            "step": 0, "max_iters": cfg.max_iters,
            "history": [], "status": "Starting...",
            "start_time": None, "last_sample": "",
        })

    t = threading.Thread(target=_training_worker, args=(cfg, corpus, tok_type), daemon=True)
    train_state["thread"] = t
    t.start()
    return "Training started!", _make_loss_plot([])


def stop_training():
    with _train_lock:
        if not train_state["running"]:
            return "No training is currently running."
        train_state["stop_requested"] = True
    return "Stop requested — will halt after next eval checkpoint."


def poll_training():
    """Called by gr.Timer to refresh the monitor UI."""
    with _train_lock:
        status = train_state["status"]
        history = list(train_state["history"])
        step = train_state["step"]
        max_iters = train_state["max_iters"]
        running = train_state["running"]

    progress = f"{step:,} / {max_iters:,}" if max_iters else "—"
    pct = (step / max_iters * 100) if max_iters else 0
    header = f"**{'🟢 Running' if running else '⚪ Idle'}**  |  Step: {progress}  |  {pct:.1f}%\n\n{status}"

    return header, _make_loss_plot(history)


# ---------------------------------------------------------------------------
# Tab 3 — Attention Visualizer
# ---------------------------------------------------------------------------

def viz_attention(checkpoint, text, layer_idx, head_idx):
    status = _ensure_model_loaded(checkpoint)
    if current_model is None:
        return None, status
    if not hasattr(current_model, "blocks"):
        return None, "Model does not support attention visualization (BigramLM has no attention)."
    if not text:
        text = "Hello, world!"

    try:
        ids = current_tokenizer.encode(text)
    except Exception as e:
        return None, f"Encoding error: {e}"

    idx_tensor = torch.tensor([ids], dtype=torch.long, device=current_device)
    weights = capture_attention_weights(current_model, idx_tensor)

    if not weights or not weights[0]:
        return None, "No attention weights captured. Check the checkpoint is a GPT model."

    layer_i, head_i = int(layer_idx), int(head_idx)
    n_layers = len(weights)
    n_heads = len(weights[0])

    if layer_i >= n_layers or head_i >= n_heads:
        return None, f"Out of range. Model has {n_layers} layer(s) and {n_heads} head(s) per layer."

    head_weights = weights[layer_i][head_i]
    tokens = [current_tokenizer.decode([i]) for i in ids]

    fig = plot_attention_heatmap(
        head_weights, tokens,
        title=f"Attention  |  Layer {layer_i}  ·  Head {head_i}"
    )
    return fig, f"✅ Showing layer {layer_i}, head {head_i}. Model has {n_layers}L × {n_heads}H."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="GPT-from-Scratch V2 Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 GPT-from-Scratch V2 Dashboard")
    gr.Markdown("*Interactive playground for your trained GPT model.*")

    # ── Tab 1: Generation ────────────────────────────────────────────────────
    with gr.Tab("🎯 Generation Playground"):
        with gr.Row():
            with gr.Column(scale=1):
                gen_ckpt = gr.Dropdown(
                    choices=get_checkpoints(), label="Checkpoint",
                    value=get_checkpoints()[0] if get_checkpoints() else None
                )
                gen_refresh = gr.Button("🔄 Refresh Checkpoints", size="sm")
                gen_prompt  = gr.Textbox(lines=4, label="Prompt", placeholder="Type a seed text…")
                with gr.Row():
                    gen_max_tok = gr.Slider(10, 2000, value=200, step=10, label="Max New Tokens")
                    gen_temp    = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature")
                gen_cache   = gr.Checkbox(value=True, label="Use KV Cache (faster)")
                gen_btn     = gr.Button("Generate ▶", variant="primary")
            with gr.Column(scale=2):
                gen_out = gr.Textbox(lines=18, label="Generated Text", interactive=False)

        def refresh_ckpts():
            ckpts = get_checkpoints()
            return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None)

        gen_refresh.click(refresh_ckpts, outputs=gen_ckpt)
        gen_btn.click(
            generate_text,
            inputs=[gen_ckpt, gen_prompt, gen_max_tok, gen_temp, gen_cache],
            outputs=gen_out,
        )

    # ── Tab 2: Training Monitor ──────────────────────────────────────────────
    with gr.Tab("📈 Training Monitor"):
        gr.Markdown("Launch a new training run directly from the browser and watch it train in real time.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                mon_preset    = gr.Dropdown(choices=list_presets(), value="small", label="Preset")
                mon_corpus    = gr.Textbox(value="data/input.txt", label="Corpus Path")
                mon_tok       = gr.Radio(["char", "bpe"], value="char", label="Tokenizer")
                mon_iters     = gr.Number(value=0, label="Max Iters (0 = preset default)")
                mon_batch     = gr.Number(value=0, label="Batch Size (0 = preset default)")
                with gr.Row():
                    mon_amp   = gr.Checkbox(value=True, label="AMP (Mixed Precision)")
                    mon_accum = gr.Number(value=1, minimum=1, label="Grad Accum Steps")
                with gr.Row():
                    mon_start = gr.Button("▶ Start Training", variant="primary")
                    mon_stop  = gr.Button("⏹ Stop", variant="stop")
                mon_ctrl_status = gr.Textbox(label="Control Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Live Training Status")
                mon_status    = gr.Markdown("*Not started.*")
                mon_loss_plot = gr.Plot(label="Loss Curves")

        # 2-second polling timer
        mon_timer = gr.Timer(value=2)

        mon_start.click(
            start_training,
            inputs=[mon_preset, mon_corpus, mon_tok, mon_iters, mon_batch, mon_amp, mon_accum],
            outputs=[mon_ctrl_status, mon_loss_plot],
        )
        mon_stop.click(stop_training, outputs=mon_ctrl_status)
        mon_timer.tick(poll_training, outputs=[mon_status, mon_loss_plot])

    # ── Tab 3: Attention Visualizer ──────────────────────────────────────────
    with gr.Tab("🔍 Attention Visualizer"):
        gr.Markdown("Inspect which tokens attend to which inside the model.")
        with gr.Row():
            with gr.Column(scale=1):
                viz_ckpt   = gr.Dropdown(choices=get_checkpoints(), label="Checkpoint")
                viz_text   = gr.Textbox(
                    lines=3, label="Input Text",
                    value="O Romeo, Romeo! wherefore art thou Romeo?"
                )
                with gr.Row():
                    viz_layer = gr.Number(value=0, label="Layer Index")
                    viz_head  = gr.Number(value=0, label="Head Index")
                viz_btn    = gr.Button("Visualize ▶", variant="primary")
                viz_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                viz_plot = gr.Plot(label="Attention Heatmap")

        viz_btn.click(
            viz_attention,
            inputs=[viz_ckpt, viz_text, viz_layer, viz_head],
            outputs=[viz_plot, viz_status],
        )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
