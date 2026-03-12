# src/training/finetune.py
# ==========================
# Phase 2: Supervised Fine-Tuning (SFT)
#
# Prompt format (Alpaca-style):
# ─────────────────────────────
# Below is an instruction that describes a task.
# Write a response that appropriately completes the request.
#
# ### Instruction:
# {instruction}
#
# ### Input:          ← optional
# {input}
#
# ### Response:
# {output}

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.transformer import GPTModel, generate
from src.tokenizer.bpe_tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_section}"
    "### Response:\n{response}"
)


def format_entry(entry: dict, include_response: bool = True) -> str:
    """Convert an instruction dict to a formatted string."""
    inp_section = f"### Input:\n{entry['input']}\n\n" if entry.get("input") else ""
    response    = entry.get("output", "") if include_response else ""
    return TEMPLATE.format(
        instruction=entry["instruction"],
        input_section=inp_section,
        response=response,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (supports story_completion + poetry)
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_or_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def load_finetune_data(data_dir: str) -> list[dict]:
    """
    Load all JSON finetune data from subdirectories.
    Looks for data.json in: story_completion/, poetry/
    """
    entries = []
    for subdir in ["story_completion", "poetry"]:
        path = os.path.join(data_dir, subdir, "data.json")
        if os.path.exists(path):
            items = load_jsonl_or_json(path)
            entries.extend(items)
            print(f"  Loaded {len(items):3d} entries from {path}")
    return entries


class SFTDataset(Dataset):
    def __init__(self, entries: list[dict], tokenizer: BPETokenizer, max_length: int):
        self.max_length = max_length
        self.data: list[torch.Tensor] = []

        for entry in entries:
            text = format_entry(entry, include_response=True)
            ids  = tokenizer.encode(text, allowed_special={"<|endoftext|>"})[:max_length]
            padded = [0] * max_length
            padded[:len(ids)] = ids
            self.data.append(torch.tensor(padded, dtype=torch.long))

    def __len__(self)        -> int:          return len(self.data)
    def __getitem__(self, i) -> tuple:
        seq = self.data[i]
        return seq[:-1], seq[1:]    # (input, target) shifted by 1


def make_sft_loader(
    entries:    list[dict],
    tokenizer:  BPETokenizer,
    batch_size: int,
    max_length: int,
    shuffle:    bool,
) -> DataLoader:
    ds = SFTDataset(entries, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def sft_loss(x, y, model, device):
    """Cross-entropy loss ignoring padding (token_id == 0)."""
    x, y  = x.to(device), y.to(device)
    logits = model(x)
    B, T, V = logits.shape
    return nn.functional.cross_entropy(
        logits.view(B * T, V), y.view(B * T), ignore_index=0
    )


def eval_sft_loader(loader, model, device, n_batches=None):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if n_batches and i >= n_batches:
                break
            total += sft_loss(x, y, model, device).item()
            count += 1
    model.train()
    return total / count if count else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def respond(instruction, inp, model, tokenizer, device,
            max_tokens=200, temperature=0.7, top_k=40):
    """Generate a model response for an instruction."""
    prompt = format_entry({"instruction": instruction, "input": inp}, include_response=False)
    ids    = tokenizer.encode_to_tensor(prompt).to(device)
    ctx    = model.pos_emb.weight.shape[0]
    model.eval()
    with torch.no_grad():
        out = generate(model, ids, max_tokens, ctx, temperature=temperature, top_k=top_k)
    full = tokenizer.decode_from_tensor(out)
    model.train()
    if "### Response:\n" in full:
        return full.split("### Response:\n")[-1].strip()
    return full[len(prompt):].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Loss plot
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_plot(train_losses, val_losses, path="results/plots/sft_loss.png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, train_losses, label="Train", color="#2563eb", linewidth=2)
    ax.plot(x, val_losses,   label="Val",   color="#dc2626", linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ↳ SFT loss plot saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tuning function
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    model:          GPTModel,
    tokenizer:      BPETokenizer,
    train_entries:  list[dict],
    val_entries:    list[dict],
    device:         torch.device,
    cfg:            dict,
    checkpoint_dir: str = "checkpoints/finetuned",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = make_sft_loader(train_entries, tokenizer, cfg["batch_size"], cfg["max_length"], True)
    val_loader   = make_sft_loader(val_entries,   tokenizer, cfg["batch_size"], cfg["max_length"], False)
    print(f"SFT train: {len(train_entries)} | val: {len(val_entries)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    train_losses, val_losses = [], []
    step = 0

    print(f"\n{'='*50}\n  Fine-tuning — {cfg['num_epochs']} epochs\n{'='*50}")
    demo = "Write a short story about a lighthouse keeper."

    for epoch in range(cfg["num_epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = sft_loss(x_batch, y_batch, model, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % cfg["eval_freq"] == 0:
                tl = eval_sft_loader(train_loader, model, device, cfg["eval_iter"])
                vl = eval_sft_loader(val_loader,   model, device, cfg["eval_iter"])
                train_losses.append(tl)
                val_losses.append(vl)
                print(f"  Ep {epoch+1} | step {step:04d} | train {tl:.4f} | val {vl:.4f} | ppl {math.exp(vl):.1f}")

        # Demo
        print(f"\n  Demo (epoch {epoch+1}):")
        print(" ", respond(demo, "", model, tokenizer, device))
        print()

        ckpt = {"epoch": epoch+1, "model": model.state_dict(), "config": model.cfg}
        torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pt"))
        torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
        print(f"  Saved checkpoint: epoch_{epoch+1:02d}.pt")

    save_loss_plot(train_losses, val_losses)
    print("\nFine-tuning complete.")
    return train_losses, val_losses


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from src.model.transformer import GPTModel
    from src.tokenizer.bpe_tokenizer import BPETokenizer

    SMALL_CFG = {
        "vocab_size": 50257, "context_length": 256,
        "emb_dim": 256, "n_heads": 8, "n_layers": 6,
        "drop_rate": 0.1, "qkv_bias": False,
    }
    SFT_CFG = {
        "batch_size": 8, "max_length": 256,
        "num_epochs": 3, "lr": 5e-5, "weight_decay": 0.01,
        "eval_freq": 50, "eval_iter": 5, "train_split": 0.85,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/pretrained/latest.pt")
    parser.add_argument("--data_dir",    default="data/finetune")
    parser.add_argument("--epochs",      default=None, type=int)
    parser.add_argument("--from_scratch",action="store_true")
    args = parser.parse_args()

    if args.epochs:
        SFT_CFG["num_epochs"] = args.epochs

    device    = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = BPETokenizer()

    if args.from_scratch:
        model = GPTModel(SMALL_CFG).to(device)
    else:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model = GPTModel(ckpt.get("config", SMALL_CFG)).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {args.checkpoint}")

    all_entries = load_finetune_data(args.data_dir)
    split       = int(len(all_entries) * SFT_CFG["train_split"])
    finetune(model, tokenizer, all_entries[:split], all_entries[split:], device, SFT_CFG)
