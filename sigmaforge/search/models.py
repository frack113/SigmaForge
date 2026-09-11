from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    id: str
    collection: str
    text: str
    score: float
    payload: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return {key: value for key, value in self.payload.items() if key != "text"}

    @property
    def fusion_key(self) -> str:
        return f"{self.collection}:{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }
