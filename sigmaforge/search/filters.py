from __future__ import annotations

import re
from collections.abc import Mapping

from qdrant_client import models

FILTER_KEYS: frozenset[str] = frozenset(
    {
        "rule_id",
        "title",
        "author",
        "level",
        "status",
        "product",
        "category",
        "service",
        "date",
        "modified",
        "chunk_type",
        "collection",
        "tags",
        "references",
    }
)

LIST_FILTER_KEYS: frozenset[str] = frozenset({"tags", "references"})

_FILTER_PATTERN = re.compile(r"(\w+):\s*(\S+)")
_VALUE_CLEAN_RE = re.compile(r"[\s,;:]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_value(raw: str) -> str:
    value = raw.strip().strip("'\"")
    return _VALUE_CLEAN_RE.sub(" ", value).strip()


def _remove_known_filter(match: re.Match[str]) -> str:
    if match.group(1).lower() in FILTER_KEYS:
        return ""
    return match.group(0)


def parse_query_filters(query: str) -> tuple[dict[str, str], str]:
    filters: dict[str, str] = {}
    for match in _FILTER_PATTERN.finditer(query):
        key = match.group(1).lower()
        if key not in FILTER_KEYS:
            continue
        value = _clean_value(match.group(2))
        if value:
            filters[key] = value
    cleaned = _FILTER_PATTERN.sub(_remove_known_filter, query)
    return filters, _WHITESPACE_RE.sub(" ", cleaned).strip()


def _split_values(value: str) -> list[str]:
    return [part for part in re.split(r"[,\s]+", value) if part]


def build_qdrant_filter(
    filters: Mapping[str, str] | None = None,
    extra_filter: models.Filter | None = None,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    for key, value in (filters or {}).items():
        if not value:
            continue
        if key in LIST_FILTER_KEYS:
            values = _split_values(value)
            if values:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=values))
                )
        else:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    if extra_filter is None:
        return models.Filter(must=conditions) if conditions else None
    if not conditions:
        return extra_filter
    return models.Filter(must=[*conditions, extra_filter])
