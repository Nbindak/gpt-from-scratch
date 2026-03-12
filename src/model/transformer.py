# src/model/transformer.py
# ==========================
# Full GPT-style decoder-only Transformer built from scratch.
#
# Architecture
# ────────────
#   TokenEmbedding(vocab_size → emb_dim)
#   PositionEmbedding(context_length → emb_dim)   ← learned, not sinusoidal
#   Dropout
#   [TransformerBlock × N]
#     ├─ LayerNorm  (pre-norm)
#     ├─ MultiHeadCausalAttention
#     ├─ Residual (+x)
#     ├─ LayerNorm  (pre-norm)
#     ├─ FeedForward  (Linear → GELU → Linear)
#     └─ Residual (+x)
#   FinalLayerNorm
#   Linear head  (emb_dim → vocab_size, weight-tied with token embedding)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadCausalAttention


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """
    Layer normalisation with learnable scale (γ) and shift (β).

    Normalises across the last dimension (feature dimension) per example,
    making it batch-size independent — crucial for variable-length sequences.
    """
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean  = x.mean(dim=-1, keepdim=True)
        var   = x.var( dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + self.eps).sqrt() + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GPT-2 tanh approximation).

    Smoother than ReLU: allows small negative activations,
    improving gradient flow through deep networks.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with GELU between them.
    Hidden dim = 4 × emb_dim (standard GPT design).
    Each position processed independently — no cross-position mixing here
    (that's the attention's job).
    """
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["emb_dim"]
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block:  attention → FFN, both with pre-LayerNorm
    and residual (skip) connections.

    Pre-norm (normalise before the sub-layer, not after) is more stable
    for deep models and is the design used by GPT-2 and most modern LLMs.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.attn  = MultiHeadCausalAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff    = FeedForward(cfg)
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sub-layer
        x = x + self.drop(self.attn(self.norm1(x)))
        # Feed-forward sub-layer
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full GPT Model
# ─────────────────────────────────────────────────────────────────────────────

class GPTModel(nn.Module):
    """
    GPT-style decoder-only language model.

    cfg dict keys
    ─────────────
    vocab_size      : size of BPE vocabulary
    context_length  : maximum input sequence length
    emb_dim         : embedding / hidden dimension
    n_heads         : number of attention heads
    n_layers        : number of transformer blocks
    drop_rate       : dropout probability
    qkv_bias        : include bias in Q/K/V projections
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ── Embedding layers ─────────────────────────────────────────
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # ── Transformer blocks ───────────────────────────────────────
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # ── Output head ──────────────────────────────────────────────
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: the output projection shares weights with the
        # token embedding matrix. This reduces parameters and has been
        # shown empirically to improve language modelling performance.
        self.out_head.weight = self.tok_emb.weight

        # Initialise all weights
        self.apply(self._init_weights)

    # ── Weight initialisation ──────────────────────────────────────────
    def _init_weights(self, module: nn.Module):
        """Standard GPT-2 weight initialisation."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args
            idx : (batch, seq_len) integer token indices

        Returns
            logits : (batch, seq_len, vocab_size)
                     raw (un-softmaxed) scores over the vocabulary
                     for every position in the sequence
        """
        B, T = idx.shape

        # Token embedding + positional embedding
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop_emb(tok_emb + pos_emb)  # (B, T, emb_dim)

        # Pass through all transformer blocks
        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.out_head(x)   # (B, T, vocab_size)

    # ── Utilities ──────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        params = (
            self.parameters() if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"GPTModel(\n"
            f"  emb_dim={self.cfg['emb_dim']}, "
            f"n_layers={self.cfg['n_layers']}, "
            f"n_heads={self.cfg['n_heads']},\n"
            f"  context={self.cfg['context_length']}, "
            f"vocab={self.cfg['vocab_size']:,},\n"
            f"  params={n:,} ({n/1e6:.2f}M)\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:        GPTModel,
    idx:          torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature:  float = 1.0,
    top_k:        int | None = None,
    eos_id:       int | None = None,
) -> torch.Tensor:
    """
    Autoregressive text generation.

    temperature = 0    → greedy decoding (always pick most likely token)
    temperature > 0    → sample from softmax distribution
    top_k              → restrict sampling to the k most probable tokens
    eos_id             → stop early when end-of-sequence token is generated

    Args
        model         : trained GPTModel
        idx           : (1, T) starting token indices
        max_new_tokens: how many tokens to generate
        context_size  : model's maximum context window

    Returns
        (1, T + max_new_tokens) token indices
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to model's context window
        idx_cond = idx[:, -context_size:]
        logits   = model(idx_cond)          # (B, T, vocab)
        logits   = logits[:, -1, :]         # last position: (B, vocab)

        # ── Top-k filtering ──────────────────────────────────────────
        if top_k is not None and top_k > 0:
            top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold   = top_vals[:, -1].unsqueeze(-1)
            logits      = logits.masked_fill(logits < threshold, float("-inf"))

        # ── Sampling ─────────────────────────────────────────────────
        if temperature > 0.0:
            logits  = logits / temperature
            logits -= logits.max(dim=-1, keepdim=True).values  # stability
            probs   = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # ── EOS ───────────────────────────────────────────────────────
        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat([idx, idx_next], dim=1)

    return idx
