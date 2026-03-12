# src/training/pretrain.py
# ==========================
# Phase 1: Pretraining the GPT model from scratch.
#
# Objective: Next-token prediction (causal language modelling).
# Given a sequence of tokens [t₁, t₂, …, tₙ], predict each tᵢ₊₁
# given all previous tokens t₁…tᵢ.
# Loss = cross-entropy averaged over all positions.

import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.transformer import GPTModel, generate
from src.tokenizer.bpe_tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Sliding-window dataset for causal language modelling.

    Tokenises the entire text upfront and then yields overlapping
    (input, target) windows. Target = input shifted one position right.
    """

    def __init__(
        self,
        text:       str,
        tokenizer:  BPETokenizer,
        max_length: int,
        stride:     int,
    ):
        ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.inputs:  list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

        for start in range(0, len(ids) - max_length, stride):
            self.inputs.append( torch.tensor(ids[start          : start + max_length]))
            self.targets.append(torch.tensor(ids[start + 1      : start + max_length + 1]))

    def __len__(self)         -> int: return len(self.inputs)
    def __getitem__(self, i)       : return self.inputs[i], self.targets[i]


def make_loader(
    text:       str,
    tokenizer:  BPETokenizer,
    batch_size: int,
    max_length: int,
    stride:     int,
    shuffle:    bool = True,
) -> DataLoader:
    ds = PretrainDataset(text, tokenizer, max_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_ratio: float = 0.1) -> float:
    """
    Linearly increase LR during warmup, then decay with cosine annealing.
    Prevents large, destabilising updates at the start of training.
    """
    min_lr = max_lr * min_ratio
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_loss(x, y, model, device):
    logits = model(x.to(device))
    return nn.functional.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())


def loader_loss(loader, model, device, n_batches=None):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if n_batches and i >= n_batches:
                break
            total += batch_loss(x, y, model, device).item()
            count += 1
    model.train()
    return total / count if count else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Sample generation
# ─────────────────────────────────────────────────────────────────────────────

def sample(model, tokenizer, device, prompt, max_tokens=60):
    model.eval()
    ctx  = model.pos_emb.weight.shape[0]
    ids  = tokenizer.encode_to_tensor(prompt).to(device)
    with torch.no_grad():
        out = generate(model, ids, max_tokens, ctx, temperature=0.8, top_k=40)
    model.train()
    return tokenizer.decode_from_tensor(out).replace("\n", " ")


# ─────────────────────────────────────────────────────────────────────────────
# Loss plot
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_plot(train_losses, val_losses, tokens_seen, path="results/plots/pretrain_loss.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(x, train_losses, label="Train",      color="#2563eb", linewidth=2)
    ax1.plot(x, val_losses,   label="Validation", color="#dc2626", linewidth=2, linestyle="--")
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Seen")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ↳ Loss plot saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    model:         GPTModel,
    tokenizer:     BPETokenizer,
    train_text:    str,
    val_text:      str,
    device:        torch.device,
    cfg:           dict,
    checkpoint_dir: str = "checkpoints/pretrained",
) -> tuple[list, list, list]:
    """
    Train the GPT model on raw text using next-token prediction.

    Returns
        (train_losses, val_losses, tokens_seen) for plotting
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data loaders
    train_loader = make_loader(
        train_text, tokenizer, cfg["batch_size"], cfg["max_length"], cfg["stride"]
    )
    val_loader = make_loader(
        val_text, tokenizer, cfg["batch_size"], cfg["max_length"], cfg["stride"],
        shuffle=False
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    train_losses, val_losses, tokens_seen_log = [], [], []
    tokens_seen, step = 0, 0
    total_steps = cfg["num_epochs"] * len(train_loader)

    print(f"\n{'='*58}")
    print(f"  Pretraining — {cfg['num_epochs']} epochs | {total_steps:,} steps")
    print(f"  Model: {model.num_parameters():,} params")
    print(f"{'='*58}")

    t0 = time.time()
    for epoch in range(cfg["num_epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            # Update LR
            lr = cosine_lr(step, cfg["warmup_steps"], total_steps, cfg["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss = batch_loss(x_batch, y_batch, model, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens_seen += x_batch.numel()
            step        += 1

            if step % cfg["eval_freq"] == 0:
                tl = loader_loss(train_loader, model, device, cfg["eval_iter"])
                vl = loader_loss(val_loader,   model, device, cfg["eval_iter"])
                train_losses.append(tl)
                val_losses.append(vl)
                tokens_seen_log.append(tokens_seen)
                elapsed = time.time() - t0
                print(
                    f"  Ep {epoch+1:02d} | step {step:05d} | "
                    f"lr {lr:.2e} | train {tl:.4f} | val {vl:.4f} | "
                    f"ppl {math.exp(vl):.1f} | {elapsed:.0f}s"
                )

        # End-of-epoch sample
        print(f"\n  Sample (epoch {epoch+1}):")
        print(" ", sample(model, tokenizer, device, "The lighthouse keeper"))
        print()

        # Save checkpoint
        ckpt = {"epoch": epoch+1, "model": model.state_dict(), "config": model.cfg}
        torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pt"))
        torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))

    save_loss_plot(train_losses, val_losses, tokens_seen_log)
    final_ppl = math.exp(val_losses[-1]) if val_losses else float("nan")
    print(f"\nPretraining complete. Final val perplexity: {final_ppl:.2f}")
    return train_losses, val_losses, tokens_seen_log


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.model.transformer import GPTModel
    from src.tokenizer.bpe_tokenizer import BPETokenizer

    CONFIGS = {
        "small": {
            "vocab_size": 50257, "context_length": 256,
            "emb_dim": 256, "n_heads": 8, "n_layers": 6,
            "drop_rate": 0.1, "qkv_bias": False,
        },
        "medium": {
            "vocab_size": 50257, "context_length": 512,
            "emb_dim": 512, "n_heads": 8, "n_layers": 8,
            "drop_rate": 0.1, "qkv_bias": False,
        },
    }
    TRAIN_CFG = {
        "batch_size": 16, "max_length": 256, "stride": 128,
        "num_epochs": 10, "lr": 3e-4, "weight_decay": 0.1,
        "warmup_steps": 100, "eval_freq": 100, "eval_iter": 10,
        "train_split": 0.9,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/pretrain/data.txt")
    parser.add_argument("--model",  default="small", choices=["small","medium"])
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.epochs:
        TRAIN_CFG["num_epochs"] = args.epochs

    device    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = BPETokenizer()

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    split = int(len(text) * TRAIN_CFG["train_split"])
    train_text, val_text = text[:split], text[split:]

    model = GPTModel(CONFIGS[args.model]).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from: {args.resume}")

    pretrain(model, tokenizer, train_text, val_text, device, TRAIN_CFG)
