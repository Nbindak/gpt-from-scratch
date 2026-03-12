# src/evaluation/human_eval.py
# ==============================
# LLM-as-judge evaluation (uses Ollama for local inference).
#
# The approach: feed each (instruction, expected_output, model_response)
# triple to a stronger LLM judge (e.g. llama3) and ask it to rate the
# model response 0–100. This is a scalable proxy for human evaluation.
#
# Requires: ollama running locally with `ollama pull llama3`
# Install : https://ollama.ai

import json
import re
import sys

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "llama3"


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Check whether an Ollama process is currently running."""
    if not _PSUTIL_OK:
        # Fall back to HTTP health check
        try:
            r = requests.get("http://localhost:11434/", timeout=2)
            return r.status_code == 200
        except Exception:
            return False
    for proc in psutil.process_iter(["name"]):
        if "ollama" in (proc.info.get("name") or "").lower():
            return True
    return False


def query_judge(prompt: str, model: str = JUDGE_MODEL) -> str:
    """Send a prompt to the Ollama judge and return its text response."""
    if not _REQUESTS_OK:
        raise ImportError("pip install requests")
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 42, "temperature": 0, "num_ctx": 2048},
    }
    with requests.post(OLLAMA_URL, json=data, stream=True, timeout=60) as r:
        r.raise_for_status()
        response = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            obj = json.loads(line)
            if "message" in obj:
                response += obj["message"]["content"]
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = (
    "You are evaluating the quality of a language model's response.\n\n"
    "Instruction: {instruction}\n"
    "Expected output: {expected}\n"
    "Model response: {response}\n\n"
    "Score the model response on a scale from 0 to 100 where:\n"
    "  100 = perfect: accurate, fluent, coherent, follows the instruction\n"
    "   75 = good: mostly correct with minor issues\n"
    "   50 = mediocre: partially correct but with notable problems\n"
    "   25 = poor: minimal relevance to the instruction\n"
    "    0 = completely wrong or empty\n\n"
    "Consider: accuracy, fluency, coherence, instruction-following.\n"
    "Respond with ONLY the integer score (0–100), nothing else."
)


def score_response(entry: dict, judge_model: str = JUDGE_MODEL) -> int:
    """Score a single (instruction, expected, model_response) triple."""
    if not entry.get("model_response"):
        return 0
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=entry["instruction"],
        expected=entry.get("output", ""),
        response=entry["model_response"],
    )
    try:
        raw = query_judge(prompt, judge_model)
        m   = re.search(r"\b(\d{1,3})\b", raw)
        s   = int(m.group(1)) if m else 50
        return min(100, max(0, s))
    except Exception as e:
        print(f"  Judge error: {e}", file=sys.stderr)
        return -1


def score_all(
    entries:     list[dict],
    judge_model: str = JUDGE_MODEL,
    verbose:     bool = True,
) -> tuple[list[int], float]:
    """
    Score all entries. Returns (scores list, average score).
    Entries must have 'instruction', 'output', 'model_response' keys.
    """
    if not is_ollama_running():
        raise RuntimeError(
            "Ollama is not running.\n"
            "Start it with: ollama serve\n"
            "Pull the judge model: ollama pull llama3"
        )

    scores = []
    for i, entry in enumerate(entries):
        s = score_response(entry, judge_model)
        scores.append(s)
        if verbose:
            indicator = "✓" if s >= 70 else "~" if s >= 40 else "✗"
            print(f"  [{i+1:3d}/{len(entries)}] {indicator} score={s:3d}  {entry['instruction'][:60]}")

    valid  = [s for s in scores if s >= 0]
    avg    = sum(valid) / len(valid) if valid else 0.0
    return scores, avg


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_judge_report(scores: list[int], avg: float):
    valid   = [s for s in scores if s >= 0]
    n_great = sum(1 for s in valid if s >= 75)
    n_ok    = sum(1 for s in valid if 50 <= s < 75)
    n_poor  = sum(1 for s in valid if s  < 50)

    print("\n" + "═"*44)
    print("  LLM-as-Judge Evaluation Results")
    print("═"*44)
    print(f"  Samples evaluated : {len(valid)}")
    print(f"  Average score     : {avg:.1f} / 100")
    print(f"  ≥75 (Good)        : {n_great} ({100*n_great/len(valid):.0f}%)")
    print(f"  50–74 (Mediocre)  : {n_ok}    ({100*n_ok/len(valid):.0f}%)")
    print(f"  <50  (Poor)       : {n_poor}   ({100*n_poor/len(valid):.0f}%)")
    print("═"*44)
