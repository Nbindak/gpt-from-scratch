# src/demo/interface.py
# =======================
# Core inference interface used by the Gradio app.
# Separated from app.py to keep UI code and generation logic independent.

import os
import time

import torch

from src.model.transformer import GPTModel, generate
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.finetune import format_entry


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SMALL_CFG = {
    "vocab_size": 50257, "context_length": 256,
    "emb_dim": 256, "n_heads": 8, "n_layers": 6,
    "drop_rate": 0.1, "qkv_bias": False,
}


def find_best_checkpoint() -> str | None:
    candidates = [
        "checkpoints/finetuned/latest.pt",
        "checkpoints/pretrained/latest.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_model(path: str, device: torch.device) -> tuple[GPTModel, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt.get("config", _DEFAULT_SMALL_CFG)
    m    = GPTModel(cfg).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generation interface
# ─────────────────────────────────────────────────────────────────────────────

class GenerationInterface:
    """
    Wraps a GPTModel + BPETokenizer with clean generation methods.
    Used by both the Gradio UI and the CLI generate script.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.device = torch.device(
            "cuda"  if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = BPETokenizer()
        path = checkpoint_path or find_best_checkpoint()

        if path and os.path.exists(path):
            self.model, self.cfg = load_model(path, self.device)
            print(f"Loaded: {path}  ({self.model.num_parameters():,} params, device={self.device})")
        else:
            print("No checkpoint found — using untrained model (for demo only)")
            self.cfg   = _DEFAULT_SMALL_CFG
            self.model = GPTModel(self.cfg).to(self.device)
            self.model.eval()

    @property
    def context_size(self) -> int:
        return self.model.pos_emb.weight.shape[0]

    # ── Free-form completion ─────────────────────────────────────────────
    def complete(
        self,
        prompt:      str,
        max_tokens:  int   = 200,
        temperature: float = 0.8,
        top_k:       int   = 40,
    ) -> tuple[str, dict]:
        """
        Generate a continuation for the given prompt.

        Returns
            (full_text, stats) where stats contains timing and token info.
        """
        t0  = time.time()
        ids = self.tokenizer.encode_to_tensor(prompt).to(self.device)
        n_prompt = ids.shape[-1]

        with torch.no_grad():
            out = generate(
                self.model, ids,
                max_new_tokens=max_tokens,
                context_size=self.context_size,
                temperature=temperature,
                top_k=top_k,
            )

        full      = self.tokenizer.decode_from_tensor(out)
        elapsed   = time.time() - t0
        n_new     = out.shape[-1] - n_prompt
        tok_per_s = n_new / elapsed if elapsed > 0 else 0

        stats = {
            "new_tokens":  n_new,
            "elapsed_s":   round(elapsed, 2),
            "tok_per_sec": round(tok_per_s, 1),
        }
        return full, stats

    # ── Instruction following ────────────────────────────────────────────
    def instruct(
        self,
        instruction: str,
        context:     str   = "",
        max_tokens:  int   = 250,
        temperature: float = 0.7,
        top_k:       int   = 40,
    ) -> tuple[str, dict]:
        """
        Generate a response to an instruction using the Alpaca prompt format.

        Returns
            (response_only, stats)  — the response portion only, not the full prompt.
        """
        prompt = format_entry(
            {"instruction": instruction, "input": context},
            include_response=False,
        )
        full, stats = self.complete(prompt, max_tokens, temperature, top_k)

        if "### Response:\n" in full:
            response = full.split("### Response:\n")[-1].strip()
        else:
            response = full[len(prompt):].strip()

        return response, stats

    # ── Model info ────────────────────────────────────────────────────────
    def info(self) -> str:
        n = self.model.num_parameters()
        return (
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Parameters | {n:,} ({n/1e6:.2f}M) |\n"
            f"| Vocab size | {self.cfg['vocab_size']:,} |\n"
            f"| Context length | {self.cfg['context_length']} tokens |\n"
            f"| Embedding dim | {self.cfg['emb_dim']} |\n"
            f"| Attention heads | {self.cfg['n_heads']} |\n"
            f"| Transformer layers | {self.cfg['n_layers']} |\n"
            f"| Dropout rate | {self.cfg['drop_rate']} |\n"
            f"| Device | {self.device} |\n"
        )
