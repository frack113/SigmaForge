from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from sigmaforge.errors import SearchError
from sigmaforge.search.models import SearchResult

RRF_K_DEFAULT = 60


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[SearchResult]],
    *,
    k: int = RRF_K_DEFAULT,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[SearchResult]:
    if k <= 0:
        raise SearchError("rrf k must be positive")

    resolved_weights = dict(weights or {})
    scores: dict[str, float] = {}
    results: dict[str, SearchResult] = {}
    for name, ranked_results in rankings.items():
        weight = float(resolved_weights.get(name, 1.0))
        if weight <= 0.0:
            continue
        for rank, result in enumerate(ranked_results, start=1):
            key = result.fusion_key
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            results.setdefault(key, result)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))
    if limit is not None:
        ranked_keys = ranked_keys[: max(0, limit)]
    return [replace(results[key], score=scores[key]) for key in ranked_keys]
