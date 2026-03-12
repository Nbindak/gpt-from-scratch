# 🧠 GPT From Scratch

A GPT-style decoder-only Transformer built **entirely from scratch** using PyTorch — no Hugging Face, no pre-built transformer libraries.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Quickstart](#quickstart)
6. [Phase 1 — Pretraining](#phase-1--pretraining)
7. [Phase 2 — Fine-Tuning](#phase-2--fine-tuning)
8. [Generation](#generation)
9. [Evaluation](#evaluation)
10. [Demo (Gradio)](#demo-gradio)
11. [Notebooks](#notebooks)
12. [Hyperparameter Guide](#hyperparameter-guide)
13. [Technical Limitations](#technical-limitations)

---

## Project Overview

| Phase | Goal | Data | Method |
|-------|------|------|--------|
| **Pretraining** | Learn language structure | Raw English stories (`data/pretrain/data.txt`) | Next-token prediction |
| **Fine-tuning** | Write stories & poems on instruction | 200+ pairs in `data/finetune/` | Supervised instruction tuning |

---

## Architecture

```
Input tokens  (batch, seq_len)
      │
[TokenEmbedding]  +  [PositionEmbedding]  →  Dropout
      │
┌─────▼─────────────────────────────────────┐
│  TransformerBlock  ×N                     │
│  ┌─ LayerNorm (pre-norm)                  │
│  ├─ MultiHeadCausalAttention              │
│  │   ├─ Q, K, V linear projections        │
│  │   ├─ Split into H heads                │
│  │   ├─ Scaled dot-product scores         │
│  │   ├─ Causal mask (upper triangular)    │
│  │   ├─ Softmax → Dropout                 │
│  │   └─ Output projection                 │
│  ├─ Residual (+x)                         │
│  ├─ LayerNorm (pre-norm)                  │
│  ├─ FeedForward: Linear→GELU→Linear       │
│  └─ Residual (+x)                         │
└───────────────────────────────────────────┘
      │
FinalLayerNorm  →  Linear head (vocab_size)
                   (weights tied with tok_emb)
```

All components are hand-coded in `src/model/`:
- `LayerNorm` with learnable γ/β parameters
- `GELU` activation (GPT-2 tanh approximation)
- `MultiHeadCausalAttention` with upper-triangular mask
- `FeedForward` (4× hidden expansion)
- `TransformerBlock` (pre-norm + residual connections)
- `GPTModel` with weight-tied embeddings

---

## Project Structure

```
gpt-from-scratch/
├── data/
│   ├── pretrain/
│   │   └── data.txt                  ← English stories corpus
│   └── finetune/
│       ├── story_completion/
│       │   └── data.json             ← 20 story instruction pairs
│       └── poetry/
│           └── data.json             ← 20 poetry instruction pairs
│
├── src/
│   ├── model/
│   │   ├── attention.py              ← MultiHeadCausalAttention
│   │   ├── transformer.py            ← GPTModel, TransformerBlock, generate()
│   │   └── __init__.py
│   ├── tokenizer/
│   │   ├── bpe_tokenizer.py          ← BPETokenizer wrapper (tiktoken)
│   │   ├── vocab.py                  ← Vocabulary config & stats
│   │   └── __init__.py
│   ├── training/
│   │   ├── pretrain.py               ← Phase 1 training loop
│   │   ├── finetune.py               ← Phase 2 SFT training loop
│   │   └── __init__.py
│   ├── evaluation/
│   │   ├── metrics.py                ← Perplexity, repetition, distinct-N
│   │   ├── human_eval.py             ← LLM-as-judge (Ollama)
│   │   ├── error_analysis.py         ← Failure mode detection
│   │   └── __init__.py
│   └── demo/
│       ├── app.py                    ← Gradio web UI
│       ├── interface.py              ← GenerationInterface class
│       ├── static/style.css
│       └── __init__.py
│
├── checkpoints/
│   ├── pretrained/                   ← epoch_01.pt … latest.pt
│   └── finetuned/                    ← epoch_01.pt … latest.pt
│
├── results/
│   ├── sample_generations/           ← JSON outputs from evaluation
│   └── plots/                        ← Loss curves, analysis charts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation.ipynb
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/gpt-from-scratch.git
cd gpt-from-scratch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch, tiktoken; print('OK', torch.__version__)"
```

---

## Quickstart

```bash
# Phase 1: Pretrain
python src/training/pretrain.py --data data/pretrain/data.txt

# Phase 2: Fine-tune
python src/training/finetune.py --checkpoint checkpoints/pretrained/latest.pt

# Generate a story
python -c "
from src.demo.interface import GenerationInterface
g = GenerationInterface()
r, _ = g.instruct('Write a short story about a lighthouse keeper.')
print(r)
"

# Launch web demo
python src/demo/app.py
```

---

## Phase 1 — Pretraining

**Goal**: Teach the model the structure of English via next-token prediction.

```bash
# Default (small model, 10 epochs)
python src/training/pretrain.py

# Custom options
python src/training/pretrain.py \
    --data   data/pretrain/data.txt \
    --model  medium \
    --epochs 20 \
    --resume checkpoints/pretrained/latest.pt
```

**What happens during pretraining:**
1. Text is tokenised with GPT-2 BPE (50,257-token vocabulary)
2. A sliding window creates overlapping (input, target) pairs
3. For each input window, the model predicts the next token at every position
4. Loss = cross-entropy over all positions
5. Optimiser: AdamW (β₁=0.9, β₂=0.95) + gradient clipping
6. LR schedule: linear warmup → cosine decay
7. Checkpoints and loss plots saved after every epoch → `results/plots/pretrain_loss.png`

**Interpreting the loss curve:**

| Behaviour | Meaning |
|-----------|---------|
| Train loss >> Val loss | Underfitting — train more |
| Val loss rises while train falls | Overfitting — add dropout, more data |
| Both plateau | Learning rate too low, or data exhausted |

---

## Phase 2 — Fine-Tuning

**Goal**: Teach the model to follow instructions and write stories.

```bash
# Fine-tune from pretrained checkpoint
python src/training/finetune.py \
    --checkpoint checkpoints/pretrained/latest.pt \
    --data_dir   data/finetune

# Start from scratch (no pretraining)
python src/training/finetune.py --from_scratch
```

**Prompt format (Alpaca-style):**
```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Write a short story about a lighthouse keeper.

### Response:
Old Thomas had kept the lighthouse on Gull Point...
```

**Fine-tuning dataset:**

| Category | File | Pairs |
|----------|------|-------|
| Story completion | `data/finetune/story_completion/data.json` | 20 |
| Poetry | `data/finetune/poetry/data.json` | 20 |
| **Total** | | **40** |

---

## Generation

```python
from src.demo.interface import GenerationInterface

g = GenerationInterface()  # loads best checkpoint automatically

# Story generation (instruction mode)
response, stats = g.instruct(
    instruction="Write a story about two friends reuniting.",
    context="Set in a harbour town.",
    max_tokens=300,
    temperature=0.75,
    top_k=40,
)
print(response)

# Text completion (freeform mode)
full_text, stats = g.complete(
    prompt="The lighthouse stood at the edge of the world,",
    max_tokens=150,
    temperature=0.8,
)
print(full_text)
```

**Generation parameters:**

| Parameter | Effect |
|-----------|--------|
| `temperature=0.0` | Greedy — always picks most likely token |
| `temperature=0.7` | Balanced creativity |
| `temperature=1.0` | Very creative, more errors |
| `top_k=40` | Sample from top 40 tokens only |
| `max_tokens` | Maximum new tokens to generate |

---

## Evaluation

```python
from src.evaluation.metrics import compute_perplexity, evaluate_generations
from src.evaluation.error_analysis import analyse_batch, print_error_report
from src.demo.interface import GenerationInterface
from src.tokenizer.bpe_tokenizer import BPETokenizer

g         = GenerationInterface()
tokenizer = BPETokenizer()

# Perplexity
with open('data/pretrain/data.txt') as f:
    text = f.read()
ppl = compute_perplexity(text[-5000:], g.model, tokenizer, g.device)
print(f"Perplexity: {ppl:.2f}")   # lower = better

# Generation quality
prompts = ["The lighthouse keeper", "She found the letter"]
metrics = evaluate_generations(g.model, tokenizer, g.device, prompts)
print(metrics)  # repetition, distinct-2, avg_words, pct_proper_end

# Failure mode analysis
analysis = analyse_batch(prompts, metrics['samples'])
print_error_report(analysis)
```

**Metrics explained:**

| Metric | Formula | Good range |
|--------|---------|------------|
| Perplexity | exp(avg NLL per token) | 20–80 (small model) |
| Repetition rate | Duplicate bigrams / total | < 0.2 |
| Distinct-2 | Unique bigrams / total | > 0.5 |
| % proper endings | Ends with `.!?` | > 70% |

**LLM-as-judge** (optional, requires [Ollama](https://ollama.ai)):
```bash
ollama serve
ollama pull llama3
# Then run notebook 02_evaluation.ipynb, cell 7
```

---

## Demo (Gradio)

```bash
pip install gradio

# Launch locally
python src/demo/app.py

# Create public share link
python src/demo/app.py --share

# Custom port
python src/demo/app.py --port 7861
```

Open **http://localhost:7860**

**Tabs:**
- **📖 Story Generator** — instruction → story/poem
- **✏️ Text Completion** — prompt → continued text
- **ℹ️ Model Info** — architecture statistics

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `01_data_exploration.ipynb` | Token counts, BPE visualisation, length distributions, SFT dataset analysis |
| `02_evaluation.ipynb` | Perplexity, generation quality, failure modes, temperature analysis, LLM judge |

```bash
cd notebooks
jupyter notebook
```

---

## Hyperparameter Guide

### Model sizes

| Config | `emb_dim` | `n_layers` | `n_heads` | Context | Params | VRAM |
|--------|-----------|-----------|-----------|---------|--------|------|
| Small  | 256 | 6 | 8 | 256 | ~10M | ~1 GB |
| Medium | 512 | 8 | 8 | 512 | ~50M | ~3 GB |

### Pretraining

| Param | Value | Why |
|-------|-------|-----|
| `lr` | `3e-4` | Standard AdamW for transformers |
| `weight_decay` | `0.1` | Prevents weight blow-up |
| `warmup_steps` | `100` | Avoids large initial updates |
| `batch_size` | `16` | Fits in ~2GB VRAM |
| `stride` | `128` | 50% window overlap = 2× data |

### Fine-tuning

| Param | Value | Why |
|-------|-------|-----|
| `lr` | `5e-5` | 6× lower to preserve pretrained features |
| `num_epochs` | `3` | SFT overfits quickly on small datasets |
| `batch_size` | `8` | SFT examples are longer |

---

## Technical Limitations

| Limitation | Root cause | Fix |
|------------|-----------|-----|
| **Repetition** | Small dataset + autoregressive generation | Lower temperature; use repetition penalty |
| **Context loss** | 256-token window | Upgrade to medium model (512 context) |
| **Short generations** | EOS token or lack of long training examples | More data; increase `max_new_tokens` |
| **Grammar errors** | Limited corpus | Larger, more diverse corpus |
| **Story incoherence** | Small model capacity | Medium model + more data |

### Common failure modes

**Repetition loop:**
```
"The keeper walked to the cliff. The keeper walked to the cliff. The keeper…"
```
*Fix:* `temperature=0.65`, `top_k=50`

**Topic drift (context loss):**
*Story starts about a lighthouse, ends discussing cooking.*  
*Fix:* Increase `context_length` to 512

**Truncated output:**
*"She opened the letter and"* ← stops mid-sentence  
*Fix:* Increase `max_new_tokens`

---

## Acknowledgements

Architecture based on *"Build a Large Language Model From Scratch"*  
by Sebastian Raschka (Manning, 2024).  
Code: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

All model code is a clean-room reimplementation for pedagogical purposes.
