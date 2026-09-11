from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from qdrant_client import models

from sigmaforge.embed.sparse import SparseVector
from sigmaforge.errors import SearchError
from sigmaforge.qdrant import QdrantStore, SearchHit
from sigmaforge.search.context import format_context, format_result, get_citation
from sigmaforge.search.filters import build_qdrant_filter, parse_query_filters
from sigmaforge.search.fusion import RRF_K_DEFAULT, reciprocal_rank_fusion
from sigmaforge.search.models import SearchResult

DEFAULT_TOP_K = 15
SIMILARITY_THRESHOLD = 0.0
DEFAULT_ALPHA = 0.5

ALPHA_BY_COLLECTION: dict[str, float] = {
    "sigma_rules": 0.5,
    "sigma_docs": 0.7,
    "sigma_spec": 0.3,
}


class DenseEncoderLike(Protocol):
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        ...


class SparseEncoderLike(Protocol):
    def encode_text(self, text: str) -> SparseVector:
        ...


class SearchEngine:
    def __init__(
        self,
        store: QdrantStore,
        dense_encoder: DenseEncoderLike,
        sparse_encoder: SparseEncoderLike | None = None,
        *,
        collections: Sequence[str] | None = None,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        alpha: float = DEFAULT_ALPHA,
        alpha_by_collection: Mapping[str, float] | None = None,
        rrf_k: int = RRF_K_DEFAULT,
        rrf_weights: Mapping[str, float] | None = None,
    ) -> None:
        self._store = store
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._collections = tuple(collections or store.manager.collections)
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold
        self._alpha = alpha
        self._alpha_by_collection = dict(alpha_by_collection or {})
        self._rrf_k = rrf_k
        self._rrf_weights = dict(rrf_weights or {})

    @property
    def collections(self) -> tuple[str, ...]:
        return self._collections

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collections: Sequence[str] | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: models.Filter | None = None,
    ) -> list[SearchResult]:
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        limit = top_k if top_k is not None else self._top_k
        if limit <= 0:
            return []

        resolved_filters = dict(filters or {})
        inline_filters, clean_query = parse_query_filters(normalized)
        resolved_filters.update(inline_filters)
        embed_query = clean_query if clean_query else normalized

        selected = list(collections or self._collections)
        if (
            "references" in resolved_filters
            and "sigma_docs" in self._collections
            and "sigma_docs" not in selected
        ):
            selected.append("sigma_docs")
        if not selected:
            return []

        qdrant_filter = build_qdrant_filter(resolved_filters, extra_filter)
        dense_vector = self._dense_vector(embed_query)
        sparse_vector = self._sparse_vector(embed_query)

        per_collection_limit = max(limit * 2, 10)
        rankings: dict[str, list[SearchResult]] = {}
        for collection in selected:
            results = self._search_collection(
                collection,
                embed_query,
                per_collection_limit,
                qdrant_filter,
                dense_vector,
                sparse_vector,
            )
            if results:
                rankings[collection] = results
        if not rankings:
            return []
        return reciprocal_rank_fusion(
            rankings,
            k=self._rrf_k,
            weights=self._rrf_weights,
            limit=limit,
        )

    def search_collection(
        self,
        collection: str,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: models.Filter | None = None,
    ) -> list[SearchResult]:
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        limit = top_k if top_k is not None else self._top_k
        if limit <= 0:
            return []

        resolved_filters = dict(filters or {})
        inline_filters, clean_query = parse_query_filters(normalized)
        resolved_filters.update(inline_filters)
        embed_query = clean_query if clean_query else normalized

        qdrant_filter = build_qdrant_filter(resolved_filters, extra_filter)
        return self._search_collection(
            collection,
            embed_query,
            limit,
            qdrant_filter,
            self._dense_vector(embed_query),
            self._sparse_vector(embed_query),
        )

    def format_result(self, result: SearchResult) -> dict[str, Any]:
        return format_result(result)

    def get_citation(self, result: SearchResult) -> str:
        return get_citation(result)

    def format_context(
        self,
        results: Sequence[SearchResult],
        *,
        max_results: int = 5,
        max_chars: int = 1200,
    ) -> str:
        return format_context(results, max_results=max_results, max_chars=max_chars)

    def _search_collection(
        self,
        collection: str,
        query: str,
        limit: int,
        qdrant_filter: models.Filter | None,
        dense_vector: list[float] | None,
        sparse_vector: SparseVector | None,
    ) -> list[SearchResult]:
        alpha = float(self._alpha_by_collection.get(collection, self._alpha))
        dense_weight = min(max(alpha, 0.0), 1.0)
        sparse_weight = max(0.0, 1.0 - dense_weight)
        if self._sparse_encoder is None:
            dense_weight = 1.0
            sparse_weight = 0.0

        dense_hits: list[SearchHit] = []
        if dense_weight > 0.0 and dense_vector:
            dense_hits = self._store.query_dense(
                collection,
                dense_vector,
                limit=limit,
                query_filter=qdrant_filter,
            )

        sparse_hits: list[SearchHit] = []
        if sparse_weight > 0.0 and sparse_vector is not None and sparse_vector.indices:
            sparse_hits = self._store.query_sparse(
                collection,
                list(sparse_vector.indices),
                list(sparse_vector.values),
                limit=limit,
                query_filter=qdrant_filter,
            )

        dense_hits = [hit for hit in dense_hits if hit.score >= self._similarity_threshold]
        sparse_hits = [hit for hit in sparse_hits if hit.score >= self._similarity_threshold]

        rankings: dict[str, list[SearchResult]] = {}
        if dense_hits:
            rankings["dense"] = [self._to_result(hit, collection) for hit in dense_hits]
        if sparse_hits:
            rankings["sparse"] = [self._to_result(hit, collection) for hit in sparse_hits]
        if not rankings:
            return []

        return reciprocal_rank_fusion(
            rankings,
            k=self._rrf_k,
            weights={"dense": dense_weight, "sparse": sparse_weight},
            limit=limit,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return query.replace("`", "").rstrip("?").strip()

    def _dense_vector(self, query: str) -> list[float] | None:
        if not query.strip():
            return None
        try:
            vectors = self._dense_encoder.encode([query], is_query=True)
        except Exception as exc:
            raise SearchError(f"dense encoding failed: {exc}") from exc
        if not vectors or not vectors[0]:
            return None
        return list(vectors[0])

    def _sparse_vector(self, query: str) -> SparseVector | None:
        if self._sparse_encoder is None or not query.strip():
            return None
        try:
            vector = self._sparse_encoder.encode_text(query)
        except Exception as exc:
            raise SearchError(f"sparse encoding failed: {exc}") from exc
        if not vector.indices:
            return None
        return vector

    @staticmethod
    def _to_result(hit: SearchHit, collection: str) -> SearchResult:
        payload = dict(hit.payload or {})
        resolved_collection = str(payload.get("collection") or collection)
        return SearchResult(
            id=str(hit.id),
            collection=resolved_collection,
            text=str(payload.get("text") or ""),
            score=float(hit.score or 0.0),
            payload=payload,
        )
