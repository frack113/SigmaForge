from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sigmaforge.search import SearchResult


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    query: str
    answer: str
    sources: tuple[SearchResult, ...]
    model: str | None = None
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "model": self.model,
            "fallback": self.fallback,
        }
