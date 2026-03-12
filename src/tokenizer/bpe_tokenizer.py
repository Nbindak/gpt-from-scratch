# src/tokenizer/bpe_tokenizer.py
# ================================
# BPE Tokenizer wrapper and utilities.
#
# What is Byte-Pair Encoding (BPE)?
# ──────────────────────────────────
# BPE is a subword tokenisation algorithm:
#   1. Start with a vocabulary of individual characters (bytes).
#   2. Repeatedly find the most frequent adjacent pair of symbols.
#   3. Merge that pair into a new symbol and add it to the vocabulary.
#   4. Repeat until the desired vocabulary size is reached.
#
# This means common words become single tokens ("the" → [the])
# while rare words are split into known subword pieces ("lighthouse" → ["light","house"])
# and truly unknown strings fall back to character/byte level.
#
# We use OpenAI's tiktoken library which provides the exact GPT-2 BPE
# tokeniser with a 50,257-token vocabulary — fast, well-tested, production-grade.

import re
from typing import Iterator

import torch

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class BPETokenizer:
    """
    Thin wrapper around tiktoken's GPT-2 encoding that adds:
      - encode / decode helpers returning tensors
      - batch encoding
      - sliding-window chunking (for pretraining data)
      - vocabulary statistics
    """

    def __init__(self, encoding_name: str = "gpt2"):
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required: pip install tiktoken\n"
                "  tiktoken uses GPT-2's BPE vocabulary (50,257 tokens)."
            )
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size: int  = self._enc.n_vocab
        self.eot_token:  int  = self._enc.eot_token   # <|endoftext|>

    # ── Core encode / decode ───────────────────────────────────────────
    def encode(self, text: str, allowed_special: set | None = None) -> list[int]:
        """Convert a text string to a list of integer token IDs."""
        special = allowed_special or {"<|endoftext|>"}
        return self._enc.encode(text, allowed_special=special)

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Convert a list of token IDs back to a string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._enc.decode(token_ids)

    # ── Tensor helpers ─────────────────────────────────────────────────
    def encode_to_tensor(self, text: str) -> torch.Tensor:
        """Encode text and return a (1, T) long tensor."""
        ids = self.encode(text)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def decode_from_tensor(self, tensor: torch.Tensor) -> str:
        """Decode a (1, T) or (T,) tensor to text."""
        return self.decode(tensor.squeeze(0).tolist())

    # ── Batch encoding ─────────────────────────────────────────────────
    def encode_batch(
        self, texts: list[str], max_length: int, pad_id: int = 0
    ) -> torch.Tensor:
        """
        Encode a list of strings to a (B, max_length) padded tensor.
        Sequences longer than max_length are truncated.
        """
        batch = []
        for text in texts:
            ids = self.encode(text)[:max_length]
            # Pad with pad_id if shorter than max_length
            ids += [pad_id] * (max_length - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long)

    # ── Sliding-window chunker ─────────────────────────────────────────
    def sliding_window_chunks(
        self,
        text: str,
        max_length: int,
        stride: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenise text and yield (input, target) tensor pairs using a
        sliding window. Target is input shifted by one (next-token prediction).

        Args
            max_length : tokens per window
            stride     : step between consecutive windows
                         (stride < max_length → overlapping windows)

        Yields
            (input_ids, target_ids) each shape (max_length,)
        """
        ids = self.encode(text, allowed_special={"<|endoftext|>"})
        for start in range(0, len(ids) - max_length, stride):
            x = torch.tensor(ids[start          : start + max_length], dtype=torch.long)
            y = torch.tensor(ids[start + 1      : start + max_length + 1], dtype=torch.long)
            yield x, y

    # ── Vocabulary statistics ──────────────────────────────────────────
    def token_count(self, text: str) -> int:
        """Return the number of tokens in a text string."""
        return len(self.encode(text))

    def token_frequency(self, text: str) -> dict[int, int]:
        """Return a dict mapping token_id → frequency in text."""
        freq: dict[int, int] = {}
        for tok in self.encode(text):
            freq[tok] = freq.get(tok, 0) + 1
        return freq

    def vocabulary_coverage(self, text: str) -> float:
        """
        Fraction of vocabulary tokens that appear in the given text.
        A value of 1.0 means every token in the vocabulary was used.
        """
        unique = len(set(self.encode(text)))
        return unique / self.vocab_size

    def average_token_length(self, text: str) -> float:
        """Average number of characters per token (higher = denser compression)."""
        tokens = self.encode(text)
        if not tokens:
            return 0.0
        return len(text) / len(tokens)

    # ── Display helpers ────────────────────────────────────────────────
    def show_tokenisation(self, text: str, max_tokens: int = 30) -> str:
        """
        Return a readable representation of how a text is tokenised.
        Useful for debugging and understanding the tokeniser.
        """
        ids   = self.encode(text)[:max_tokens]
        parts = [f"[{self._enc.decode([tok])!r}]" for tok in ids]
        suffix = "…" if len(self.encode(text)) > max_tokens else ""
        return " ".join(parts) + suffix

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size:,}, eot={self.eot_token})"
