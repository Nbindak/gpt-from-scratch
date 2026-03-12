# src/model/attention.py
# ========================
# Multi-Head Causal Self-Attention — the core of the GPT architecture.
#
# Theory
# ──────
# Self-attention lets every token directly "look at" every other token in
# the sequence. For each position we compute three vectors:
#   Query (Q) — what this position is looking for
#   Key   (K) — what this position advertises / contains
#   Value (V) — the information this position will contribute
#
# Attention score between positions i and j:
#   score(i,j) = dot(Q_i, K_j) / sqrt(head_dim)
#
# The causal mask sets score(i,j) = -inf for all j > i so that the model
# cannot "see into the future" — essential for autoregressive generation.
#
# Multi-head: we run h independent attention functions in parallel, each
# with a smaller dimension (head_dim = d_out / h), then concatenate and
# project back to d_out.  Different heads learn to attend to different
# relationship types.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    """
    Multi-head self-attention with upper-triangular causal masking.

    Parameters
    ──────────
    d_in           : input embedding dimension
    d_out          : output dimension (must be divisible by num_heads)
    context_length : maximum sequence length (determines mask size)
    num_heads      : number of parallel attention heads
    dropout        : attention weight dropout probability
    qkv_bias       : whether to add a bias term to Q/K/V projections
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, (
            f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
        )

        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads   # dimension per head

        # Separate linear projections for Q, K, V
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection: combines all heads back to d_out
        self.out_proj = nn.Linear(d_out, d_out)

        self.attn_drop = nn.Dropout(dropout)

        # ── Causal mask ────────────────────────────────────────────────
        # Upper-triangular matrix (diagonal=1):
        #   mask[i][j] = 1 means position i CANNOT attend to position j
        # i.e. the current token cannot see future tokens.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (batch, seq_len, d_in) input embeddings

        Returns
            (batch, seq_len, d_out) context-aware representations
        """
        B, T, _ = x.shape

        # ── 1. Compute Q, K, V ──────────────────────────────────────
        Q = self.W_q(x)   # (B, T, d_out)
        K = self.W_k(x)
        V = self.W_v(x)

        # ── 2. Split into heads ─────────────────────────────────────
        # (B, T, d_out) → (B, T, H, head_dim) → (B, H, T, head_dim)
        def split(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split(Q), split(K), split(V)

        # ── 3. Scaled dot-product attention scores ──────────────────
        # scores[b, h, i, j] = how much token i attends to token j in head h
        scale  = self.head_dim ** -0.5
        scores = (Q @ K.transpose(-2, -1)) * scale   # (B, H, T, T)

        # ── 4. Apply causal mask ────────────────────────────────────
        causal = self.mask.bool()[:T, :T]             # (T, T)
        scores = scores.masked_fill(causal, float("-inf"))

        # ── 5. Attention weights → weighted values ──────────────────
        weights = F.softmax(scores, dim=-1)           # (B, H, T, T)
        weights = self.attn_drop(weights)

        ctx = weights @ V                             # (B, H, T, head_dim)

        # ── 6. Merge heads and project ──────────────────────────────
        ctx = ctx.transpose(1, 2).contiguous()        # (B, T, H, head_dim)
        ctx = ctx.view(B, T, self.d_out)              # (B, T, d_out)
        return self.out_proj(ctx)

    # ── Utilities ──────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"d_out={self.d_out}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )
