# src/evaluation/metrics.py
# ===========================
# Quantitative evaluation metrics for the GPT model.
#
# Metrics implemented
# ───────────────────
# 1. Perplexity      — how well the model predicts held-out text
# 2. Repetition rate — fraction of duplicate bigrams in generated text
# 3. Distinct-N      — vocabulary diversity in generated text
# 4. Average length  — mean word count of generated sequences
# 5. Ending quality  — fraction ending with proper punctuation

import math
import re

import torch
import torch.nn as nn

from src.model.transformer import GPTModel, generate
from src.tokenizer.bpe_tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perplexity
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    text:        str,
    model:       GPTModel,
    tokenizer:   BPETokenizer,
    device:      torch.device,
    stride:      int = 128,
    max_length:  int = 256,
) -> float:
    """
    Compute per-token perplexity on an arbitrary text.

    PPL = exp( −1/N · Σ log p(tᵢ | t_{<i}) )

    Lower perplexity = better predictions.
    A random model over 50k vocab → PPL ≈ 50,000.
    Well-trained small GPT on in-domain text: PPL 20–80.

    Uses a sliding window to handle texts longer than context_length.
    """
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    total_nll, total_n = 0.0, 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids) - 1, stride):
            chunk = ids[start : start + max_length + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]],  dtype=torch.long, device=device)

            logits    = model(x)                                         # (1, T, V)
            log_probs = nn.functional.log_softmax(logits, dim=-1)        # (1, T, V)
            token_lp  = log_probs[0].gather(1, y[0].unsqueeze(-1)).squeeze(-1)  # (T,)

            total_nll -= token_lp.sum().item()
            total_n   += y.numel()

    return math.exp(total_nll / total_n) if total_n > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repetition rate
# ─────────────────────────────────────────────────────────────────────────────

def repetition_rate(text: str) -> float:
    """
    Fraction of bigrams that appear more than once.

    0.0 = no repetition  (ideal)
    1.0 = entire text is repeated bigrams (degenerate)
    """
    words  = text.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return 0.0
    return 1.0 - len(set(bigrams)) / len(bigrams)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Distinct-N (vocabulary diversity)
# ─────────────────────────────────────────────────────────────────────────────

def distinct_n(texts: list[str], n: int = 2) -> float:
    """
    Distinct-N: ratio of unique n-grams to total n-grams across all texts.

    Higher = more diverse vocabulary usage.
    Reference: Li et al. (2016), "A Diversity-Promoting Objective Function
    for Neural Conversation Models"
    """
    all_ngrams, unique_ngrams = 0, set()
    for text in texts:
        tokens = text.lower().split()
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams    += len(ngrams)
        unique_ngrams |= set(ngrams)
    return len(unique_ngrams) / all_ngrams if all_ngrams > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generation batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_generations(
    model:       GPTModel,
    tokenizer:   BPETokenizer,
    device:      torch.device,
    prompts:     list[str],
    max_tokens:  int   = 150,
    temperature: float = 0.8,
    top_k:       int   = 40,
) -> dict:
    """
    Generate text for each prompt and compute quality metrics.

    Returns a dict with:
        samples         : list of generated texts
        repetition      : average repetition rate
        distinct_1      : distinct unigrams
        distinct_2      : distinct bigrams
        avg_words       : average word count
        pct_proper_end  : % of generations ending with . ! or ?
    """
    ctx     = model.pos_emb.weight.shape[0]
    samples = []

    model.eval()
    for prompt in prompts:
        ids = tokenizer.encode_to_tensor(prompt).to(device)
        with torch.no_grad():
            out = generate(model, ids, max_tokens, ctx,
                           temperature=temperature, top_k=top_k)
        full = tokenizer.decode_from_tensor(out)
        # Return only the newly generated portion
        samples.append(full[len(prompt):].strip())

    reps    = [repetition_rate(s) for s in samples]
    lengths = [len(s.split()) for s in samples]
    endings = [bool(re.search(r"[.!?]$", s.rstrip())) for s in samples]

    return {
        "samples":        samples,
        "repetition":     round(sum(reps) / len(reps), 4),
        "distinct_1":     round(distinct_n(samples, 1), 4),
        "distinct_2":     round(distinct_n(samples, 2), 4),
        "avg_words":      round(sum(lengths) / len(lengths), 1),
        "pct_proper_end": round(100 * sum(endings) / len(endings), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(metrics: dict):
    print("\n" + "─" * 42)
    print(f"  {'Metric':<25} {'Value':>12}")
    print("─" * 42)
    for k, v in metrics.items():
        if k == "samples":
            continue
        if isinstance(v, float):
            print(f"  {k:<25} {v:>12.4f}")
        else:
            print(f"  {k:<25} {str(v):>12}")
    print("─" * 42)
