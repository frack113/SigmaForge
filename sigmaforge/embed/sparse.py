from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

STOP_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "are",
        "the",
        "in",
        "for",
        "is",
        "of",
        "and",
        "to",
        "how",
        "does",
        "a",
        "an",
        "at",
        "on",
        "with",
        "as",
        "not",
        "be",
        "or",
        "from",
        "by",
        "it",
        "its",
        "that",
        "this",
        "which",
        "can",
        "when",
        "if",
        "where",
        "will",
        "use",
        "used",
        "vs",
        "between",
        "than",
        "but",
        "must",
        "should",
        "would",
        "shall",
        "may",
        "might",
        "need",
        "do",
        "did",
        "has",
        "have",
        "had",
        "being",
        "been",
        "were",
        "was",
        "into",
        "about",
        "up",
        "out",
        "all",
        "any",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "just",
        "also",
        "then",
        "now",
        "no",
        "yes",
        "q",
    }
)

_TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b")
_MAX_HASH_ID = 2**24


def _token_id(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % _MAX_HASH_ID


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]


class Bm25SparseEncoder:
    def __call__(self, texts: Sequence[str]) -> list[SparseVector]:
        return self.encode(texts)

    def encode(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self.encode_text(text) for text in texts]

    def encode_text(self, text: str) -> SparseVector:
        tokens = [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS]
        if not tokens:
            return SparseVector((), ())
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        indices: list[int] = []
        values: list[float] = []
        for token, frequency in frequencies.items():
            values.append(1.0 + math.log(frequency))
            indices.append(_token_id(token))
        return SparseVector(indices=tuple(indices), values=tuple(values))


def create_sparse_encoder() -> Bm25SparseEncoder:
    return Bm25SparseEncoder()
