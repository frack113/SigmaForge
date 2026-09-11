from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class IngestRequest:
    directory: str | None = None
    recursive: bool = True
    mode: str = "flat"
    selected_dirs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestResult:
    file: str
    success: bool
    rule_id: str | None = None
    error: str | None = None
    chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "success": self.success,
            "rule_id": self.rule_id,
            "error": self.error,
            "chunks": self.chunks,
        }


@dataclass(slots=True)
class IngestSummary:
    collection: str
    total: int = 0
    indexed: int = 0
    failed: int = 0
    chunks: int = 0
    results: list[IngestResult] = field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        collection: str,
        results: list[IngestResult],
        chunks: int = 0,
    ) -> IngestSummary:
        success = sum(1 for result in results if result.success)
        return cls(
            collection=collection,
            total=len(results),
            indexed=success,
            failed=len(results) - success,
            chunks=chunks,
            results=list(results),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "total": self.total,
            "indexed": self.indexed,
            "failed": self.failed,
            "chunks": self.chunks,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    source_file: str
    chunk_type: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["source_file"] = self.source_file
        payload["chunk_type"] = self.chunk_type
        payload["chunk_index"] = self.chunk_index
        return payload

    def point_id(self, collection: str) -> str:
        identifier = f"{collection}:{self.source_file}:{self.chunk_type}:{self.chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identifier))
