from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sigmaforge.search.models import SearchResult


def format_result(result: SearchResult) -> dict[str, Any]:
    metadata = result.metadata
    base: dict[str, Any] = {
        "id": result.id,
        "text": result.text,
        "score": result.score,
        "collection": result.collection,
        "source_file": metadata.get("source_file", ""),
        "file_path": metadata.get("file_path", metadata.get("source_file", "")),
        "line_start": metadata.get("line_start", metadata.get("line_number", "")),
        "metadata": metadata,
    }

    if result.collection == "sigma_rules":
        base.update(
            {
                "rule_id": metadata.get("rule_id", ""),
                "title": metadata.get("title", ""),
                "level": metadata.get("level", ""),
                "status": metadata.get("status", ""),
                "chunk_type": metadata.get("chunk_type", ""),
                "product": metadata.get("product", ""),
                "category": metadata.get("category", ""),
            }
        )
    elif result.collection == "sigma_docs":
        base.update(
            {
                "doc_type": metadata.get("doc_type", ""),
                "heading_text": metadata.get("heading_text", ""),
                "heading_level": metadata.get("heading_level", 0),
                "original_url": metadata.get("original_url", ""),
                "source_rule_id": metadata.get("rule_id", ""),
            }
        )

    return base


def get_citation(result: SearchResult) -> str:
    metadata = result.metadata
    source = metadata.get("file_path") or metadata.get("source_file") or ""
    line = metadata.get("line_start") or metadata.get("line_number") or ""
    if source and line:
        return f"{source}:{line}"
    return str(source)


def format_context(
    results: Sequence[SearchResult],
    *,
    max_results: int = 5,
    max_chars: int = 1200,
) -> str:
    blocks: list[str] = []
    budget = max_chars
    for index, result in enumerate(results[: max(0, max_results)], start=1):
        metadata = result.metadata
        source = metadata.get("source_file") or metadata.get("file_path") or result.collection
        title = metadata.get("title")
        header = f"[{index}] {result.collection} / {source}"
        if title:
            header += f" / {title}"
        available = budget - len(header) - 1
        if available <= 0:
            break
        text = result.text
        if len(text) > available:
            if available <= 3:
                break
            text = text[: available - 3].rstrip() + "..."
        block = f"{header}\n{text}"
        blocks.append(block)
        budget -= len(block)
    return "\n\n".join(blocks)
