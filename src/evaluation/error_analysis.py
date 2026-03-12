# src/evaluation/error_analysis.py
# ==================================
# Systematic identification of model failure modes.
#
# Failure modes tracked
# ─────────────────────
# 1. Repetition       — repeated phrases / bigrams
# 2. Context loss     — topic drift mid-generation
# 3. Truncation       — text cut off mid-sentence
# 4. Low diversity    — model always generates similar text
# 5. Prompt leakage   — model copies the prompt verbatim
# 6. Grammar issues   — simple grammatical errors (heuristic)

import re
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Individual failure-mode detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_repetition(text: str, threshold: float = 0.35) -> dict:
    """Flag if the bigram repetition rate exceeds threshold."""
    words  = text.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if not bigrams:
        return {"detected": False, "rate": 0.0, "detail": ""}
    rate = 1.0 - len(set(bigrams)) / len(bigrams)
    if rate > threshold:
        # Find most repeated bigrams
        counts = Counter(bigrams)
        top    = counts.most_common(3)
        top_str = ", ".join(f'"{b}" (×{c})' for b, c in top if c > 1)
        return {"detected": True, "rate": round(rate, 3), "detail": top_str}
    return {"detected": False, "rate": round(rate, 3), "detail": ""}


def detect_truncation(text: str) -> dict:
    """Flag if text appears to end mid-sentence."""
    stripped = text.rstrip()
    ends_ok  = bool(re.search(r"[.!?\"']$", stripped))
    # Also flag if the last word looks like a function word or article
    last_word = stripped.split()[-1].lower() if stripped.split() else ""
    suspicious_endings = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "is", "was"}
    ends_abruptly = (not ends_ok) or (last_word in suspicious_endings)
    return {
        "detected": ends_abruptly,
        "last_chars": stripped[-20:] if len(stripped) > 20 else stripped,
    }


def detect_prompt_leakage(prompt: str, generated: str, threshold: float = 0.5) -> dict:
    """Flag if the generated text is mostly copied from the prompt."""
    prompt_words = set(prompt.lower().split())
    gen_words    = generated.lower().split()
    if not gen_words:
        return {"detected": False, "overlap": 0.0}
    overlap = sum(1 for w in gen_words if w in prompt_words) / len(gen_words)
    return {"detected": overlap > threshold, "overlap": round(overlap, 3)}


def detect_low_diversity(texts: list[str], threshold: float = 0.3) -> dict:
    """
    Flag if generated texts across prompts are too similar to each other.
    Measures average pairwise bigram overlap.
    """
    if len(texts) < 2:
        return {"detected": False, "avg_overlap": 0.0}

    def bigrams(t):
        ws = t.lower().split()
        return set(f"{ws[i]} {ws[i+1]}" for i in range(len(ws)-1))

    overlaps = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            a, b = bigrams(texts[i]), bigrams(texts[j])
            if a | b:
                overlaps.append(len(a & b) / len(a | b))

    avg = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {"detected": avg > threshold, "avg_overlap": round(avg, 3)}


def detect_topic_drift(text: str) -> dict:
    """
    Heuristic: check if the semantic topic of the first and second halves
    diverge significantly, using keyword overlap as a proxy.
    """
    words = text.lower().split()
    if len(words) < 20:
        return {"detected": False, "detail": "Text too short to assess"}

    mid   = len(words) // 2
    first = set(w for w in words[:mid]  if len(w) > 4)
    second= set(w for w in words[mid:]  if len(w) > 4)

    if not first or not second:
        return {"detected": False, "detail": ""}

    overlap = len(first & second) / len(first | second)
    drifted = overlap < 0.15

    return {
        "detected": drifted,
        "overlap":  round(overlap, 3),
        "detail":   f"First-half keywords: {list(first)[:5]}, Second-half: {list(second)[:5]}"
        if drifted else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_sample(
    prompt:    str,
    generated: str,
    verbose:   bool = False,
) -> dict:
    """Run all failure mode detectors on a single (prompt, generated) pair."""
    results = {
        "repetition":    detect_repetition(generated),
        "truncation":    detect_truncation(generated),
        "prompt_leak":   detect_prompt_leakage(prompt, generated),
        "topic_drift":   detect_topic_drift(generated),
    }
    if verbose:
        print(f"\n  Prompt   : {prompt[:60]}…")
        print(f"  Generated: {generated[:80]}…")
        for name, r in results.items():
            flag = "⚠ " if r["detected"] else "✓ "
            print(f"  {flag}{name}: {r}")
    return results


def analyse_batch(
    prompts:    list[str],
    generated:  list[str],
) -> dict:
    """
    Run failure mode analysis on a batch of generations.
    Returns aggregated failure mode statistics and individual results.
    """
    assert len(prompts) == len(generated)

    individual = [
        analyse_sample(p, g) for p, g in zip(prompts, generated)
    ]

    # Count how many samples triggered each failure mode
    mode_counts = {
        "repetition": sum(1 for r in individual if r["repetition"]["detected"]),
        "truncation": sum(1 for r in individual if r["truncation"]["detected"]),
        "prompt_leak": sum(1 for r in individual if r["prompt_leak"]["detected"]),
        "topic_drift": sum(1 for r in individual if r["topic_drift"]["detected"]),
    }

    diversity = detect_low_diversity(generated)
    n = len(generated)

    return {
        "n_samples":        n,
        "failure_counts":   mode_counts,
        "low_diversity":    diversity,
        "individual":       individual,
    }


def print_error_report(analysis: dict):
    n = analysis["n_samples"]
    print("\n" + "─"*50)
    print("  Failure Mode Analysis")
    print("─"*50)
    for mode, count in analysis["failure_counts"].items():
        pct  = 100 * count / n if n else 0
        flag = "⚠" if count > 0 else "✓"
        print(f"  {flag} {mode:<15} : {count:2d}/{n} ({pct:.0f}%)")

    div = analysis["low_diversity"]
    flag = "⚠" if div["detected"] else "✓"
    print(f"  {flag} low_diversity  : avg_overlap={div['avg_overlap']:.3f}")
    print("─"*50)

    # Recommendations
    fc = analysis["failure_counts"]
    recommendations = []
    if fc.get("repetition", 0) / max(n, 1) > 0.3:
        recommendations.append("↓ Lower temperature or use repetition penalty to reduce loops.")
    if fc.get("truncation", 0) / max(n, 1) > 0.4:
        recommendations.append("↑ Increase max_new_tokens; check EOS token prevalence in training data.")
    if fc.get("topic_drift", 0) / max(n, 1) > 0.3:
        recommendations.append("↑ Increase context_length or train with longer examples.")
    if div["detected"]:
        recommendations.append("↑ Increase temperature slightly for more diverse outputs.")

    if recommendations:
        print("\n  Recommendations:")
        for r in recommendations:
            print(f"    {r}")
