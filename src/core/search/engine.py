"""Search functionality for RAG pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

from src.core.pipeline.ingestion import DEFAULT_MODEL, build_embed_model
from src.core.search.router import route_query
from src.core.search.retrievers import get_collection_retriever
from src.infrastructure.database import DatabaseService
from src.infrastructure.llm.llamacpp.client import LlamaClient
from src.infrastructure.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 15
SIMILARITY_THRESHOLD = 0.0

FILTER_KEYS = frozenset(
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

_FILTER_PATTERN = re.compile(r"(\w+):(\S+)")
_VALUE_CLEAN_RE = re.compile(r"[\s,;:]+")


def parse_query_filters(query: str) -> tuple[dict[str, str], str]:
    """Extract key:value filters from query, return (filters, cleaned_query)."""
    filters: dict[str, str] = {}
    for match in _FILTER_PATTERN.finditer(query):
        key, raw_value = match.group(1).lower(), match.group(2)
        if key in FILTER_KEYS:
            filters[key] = _VALUE_CLEAN_RE.sub(" ", raw_value.strip()).strip()
    cleaned = _FILTER_PATTERN.sub("", query).strip()
    return filters, cleaned


def build_qdrant_filter(filters: dict[str, str]) -> Filter | None:
    """Build a Qdrant Filter from parsed filter dict."""
    if not filters:
        return None
    conditions: list[Any] = []
    for key, value in filters.items():
        if key == "tags":
            conditions.append(FieldCondition(key=key, match=MatchText(text=value)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=list(conditions))


_async_embed_model: Any | None = None
_search_embed_model_lock = threading.Lock()


def reset_search_embed_model() -> None:
    """Reset the cached search embedding model singleton and retriever cache."""
    global _async_embed_model
    with _search_embed_model_lock:
        _async_embed_model = None
    from src.core.search.retrievers import reset_retriever_cache

    reset_retriever_cache()


def _get_search_embed_model() -> Any:
    global _async_embed_model
    if _async_embed_model is not None:
        return _async_embed_model
    with _search_embed_model_lock:
        if _async_embed_model is not None:
            return _async_embed_model
        model_name = (
            DatabaseService.get_instance().get_active_embedding_model_name() or DEFAULT_MODEL
        )
        _async_embed_model = build_embed_model(model_name)
    return _async_embed_model


DEFAULT_COLLECTIONS = ["sigma_rules", "sigma_docs", "sigma_spec"]

# Per-collection hybrid alpha values (1.0 = pure dense, 0.0 = pure sparse)
# Tuned for each collection's usage pattern — see Q2.1
ALPHA_BY_COLLECTION: dict[str, float] = {
    "sigma_rules": 0.5,  # balanced — rules mix technical keywords & semantics
    "sigma_docs": 0.7,  # keyword-leaning — reference docs use precise terms
    "sigma_spec": 0.3,  # semantic-leaning — specs describe abstract concepts
}

RRF_K_DEFAULT = 60


def _node_to_result(node: Any) -> dict[str, Any]:
    """Convert a LlamaIndex NodeWithScore to the legacy dict format.

    Handles both flat payload and _node_content formats used by Qdrant.
    """
    text = node.text if hasattr(node, "text") else ""
    metadata = node.metadata if hasattr(node, "metadata") else {}
    score = float(getattr(node, "score", 0.0) or 0.0)

    return {
        "text": text,
        "score": score,
        "metadata": metadata,
    }


async def search(
    query: str,
    collection_name: str = "sigma_docs",
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    qdrant_filter: Filter | None = None,
) -> list[dict[str, Any]]:
    """Search a single collection using direct Qdrant client.

    This function is kept for backward compatibility and for use cases
    where searching a single collection is preferred over multi-collection fusion.

    Args:
        query: Natural language query (supports key:value filters).
        collection_name: Qdrant collection name.
        top_k: Number of results to return.
        similarity_threshold: Minimum similarity score.
        qdrant_filter: Optional pre-built Qdrant Filter to apply.

    Returns:
        List of search results with metadata.
    """
    if not query:
        logger.warning("Empty query provided")
        return []

    _parsed_qdrant_filter: Filter | None = None
    if qdrant_filter is None:
        filters, clean_query = parse_query_filters(query)
        _parsed_qdrant_filter = build_qdrant_filter(filters)
        embedding_query = clean_query if clean_query else query
    else:
        embedding_query = query

    try:
        client = get_qdrant_client()
        embed_model = _get_search_embed_model()
        query_embedding = await embed_model.aget_query_embedding(embedding_query)

        points = client.query_points(
            collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=_parsed_qdrant_filter or qdrant_filter,
        )
        scored_points = points.points

        if not scored_points:
            return []

        results = []
        for point in scored_points:
            score_val = point.score if point.score else 0.0
            if score_val >= similarity_threshold:
                payload = point.payload or {}

                text = payload.get("text", "")
                metadata: dict[str, Any] = {}

                if text:
                    metadata = {
                        k: v for k, v in payload.items() if k not in ("text", "_node_content")
                    }
                else:
                    node_content = payload.get("_node_content", "{}")
                    if isinstance(node_content, str):
                        import json

                        try:
                            node_data = json.loads(node_content)
                            text = node_data.get("text", "")
                            metadata = node_data.get("metadata", {})
                        except (json.JSONDecodeError, KeyError):
                            text = ""
                            metadata = payload
                    else:
                        text = node_content.get("text", "") if node_content else ""
                        metadata = node_content.get("metadata", {}) if node_content else {}

                results.append(
                    {
                        "text": text,
                        "score": score_val,
                        "metadata": metadata,
                    }
                )

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def format_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Format a search result for display.

    Args:
        result: Raw search result from Qdrant.

    Returns:
        Formatted result with title, description, metadata.
    """
    return {
        "text": result.get("text", ""),
        "score": result.get("score", 0.0),
        "metadata": result.get("metadata", {}),
        "file_path": result.get("metadata", {}).get("file_path", ""),
        "line_number": result.get("metadata", {}).get("line_start", ""),
    }


def format_result_by_collection(result: dict[str, Any]) -> dict[str, Any]:
    """Format a search result adapted to its source collection.

    Reads ``collection`` from metadata (injected by transforms via TransformConfig)
    and returns a display-oriented dict with collection-specific fields.

    Args:
        result: Raw search result from Qdrant.

    Returns:
        Formatted result with collection-appropriate fields.
    """
    meta = result.get("metadata", {})
    collection = meta.get("collection", "")

    base = {
        "text": result.get("text", ""),
        "score": result.get("score", 0.0),
        "collection": collection,
        "source_file": meta.get("source_file", ""),
        "file_path": meta.get("file_path", ""),
        "line_start": meta.get("line_start", ""),
        "metadata": meta,
    }

    if collection == "sigma_rules":
        return {
            **base,
            "rule_id": meta.get("rule_id", ""),
            "title": meta.get("title", ""),
            "level": meta.get("level", ""),
            "status": meta.get("status", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "product": meta.get("product", ""),
            "category": meta.get("category", ""),
        }

    if collection == "sigma_docs":
        return {
            **base,
            "doc_type": meta.get("doc_type", ""),
            "heading_text": meta.get("heading_text", ""),
            "heading_level": meta.get("heading_level", 0),
            "original_url": meta.get("original_url", ""),
            "source_rule_id": meta.get("rule_id", ""),
        }

    # sigma_spec or unknown
    return base


def get_citation(result: dict[str, Any]) -> str:
    """Get citation string for a result.

    Args:
        result: Search result.

    Returns:
        Citation in format "path/to/file.yaml:line".
    """
    file_path = result.get("metadata", {}).get("file_path", "")
    line_start = result.get("metadata", {}).get("line_start", "")

    if file_path and line_start:
        return f"{file_path}:{line_start}"

    return ""


class SearchEngine:
    """Search engine for multi-collection Sigma RAG."""

    def __init__(
        self,
        collection_names: list[str] | None = None,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        use_router: bool = False,
        llm_client: LlamaClient | None = None,
        alpha: float | None = None,
        alpha_by_collection: dict[str, float] | None = None,
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize search engine.

        Args:
            collection_names: List of Qdrant collections to search.
                Defaults to [sigma_rules, sigma_docs, sigma_spec].
            top_k: Number of results per collection.
            similarity_threshold: Minimum similarity score.
            use_router: Enable LLM-based query routing to search only
                relevant collections instead of all three.
            llm_client: Optional LlamaClient for the router.
                When not provided, creates a new LlamaClient.
            alpha: Fallback hybrid weight when a collection has no per-collection
                override.  Defaults to 0.3.
            alpha_by_collection: Per-collection hybrid weights.  Takes precedence
                over the global ``alpha``.  See ``ALPHA_BY_COLLECTION`` for defaults.
            rrf_k: RRF constant (default 60).  Lower values give top ranks more
                weight; higher values flatten the score distribution.
            rrf_weights: Per-collection RRF boost factors (default 1.0 for all).
                Weighted RRF formula: ``score = weight / (rrf_k + rank)``.
        """
        self.collection_names = collection_names or list(DEFAULT_COLLECTIONS)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_router = use_router
        self._llm_client = llm_client
        self._fallback_alpha = alpha if alpha is not None else 0.3
        merged = dict(ALPHA_BY_COLLECTION)
        if alpha_by_collection:
            merged.update(alpha_by_collection)
        self._alpha_by_collection = merged
        self._rrf_k = rrf_k
        self._rrf_weights = rrf_weights or {}

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search across configured collections and fuse results with QueryFusionRetriever.

        Uses LlamaIndex's built-in RRF fusion instead of manual implementation.
        When ``use_router`` is enabled, classifies the query first and searches only
        the relevant collections.  Falls back to all collections if routing fails.

        When ``metadata_filter`` is provided, applies a Qdrant metadata filter to
        each collection search before RRF fusion.
        """
        query = query.replace("`", "").rstrip("?").strip()
        limit = top_k if top_k is not None else self.top_k
        per_collection_k = max(limit * 2, 10)

        # Parse inline key:value filters (e.g. references:url) from query
        inline_filters, clean_query = parse_query_filters(query)
        if inline_filters:
            metadata_filter = {**(metadata_filter or {}), **inline_filters}
            query = clean_query if clean_query else query

        # When filtering by references, always include sigma_docs
        has_ref_filter = metadata_filter and "references" in metadata_filter

        # Determine which collections to search
        cols_to_search: list[str] = []
        if self.use_router:
            try:
                routed = await route_query(query, llm_client=self._llm_client)
                cols_to_search = [c for c in routed if c in self.collection_names]
                if not cols_to_search:
                    cols_to_search = self.collection_names
            except Exception as e:
                logger.warning("Router failed, falling back to all collections: %s", e)
                cols_to_search = self.collection_names
        else:
            cols_to_search = self.collection_names

        # When filtering by references, always include sigma_docs even if
        # the router chose otherwise — reference docs live there.
        if has_ref_filter and "sigma_docs" not in cols_to_search:
            cols_to_search.append("sigma_docs")

        # Build Qdrant filter for metadata filtering
        qdrant_filter = build_qdrant_filter(metadata_filter) if metadata_filter else None

        # Create retrievers for each collection
        retrievers = []
        for col in cols_to_search:
            try:
                retriever = get_collection_retriever(
                    collection_name=col,
                    top_k=per_collection_k,
                    metadata_filter=qdrant_filter,
                    alpha=self._alpha_by_collection.get(col, self._fallback_alpha),
                )
                retrievers.append(retriever)
            except Exception as e:
                logger.warning("Failed to create retriever for '%s', skipping: %s", col, e)

        if not retrievers:
            logger.error("No retrievers available for collections: %s", cols_to_search)
            return []

        # Air-gap: avoid QueryFusionRetriever which tries to load an LLM (OpenAI).
        # Always use manual RRF fusion directly with hybrid retrievers.
        return await self._search_manual_rrf(
            query, cols_to_search, limit, qdrant_filter, retrievers
        )

    async def _search_manual_rrf(
        self,
        query: str,
        collections: list[str],
        limit: int,
        qdrant_filter: Filter | None,
        retrievers: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Manual RRF fusion using hybrid-aware retrievers.

        Uses VectorIndexRetriever with HYBRID mode when retrievers are provided;
        falls back to dense-only ``search()`` otherwise.
        """
        per_collection_k = max(limit * 2, 10)

        if retrievers:
            # Use hybrid retrievers (sparse + dense) for each collection
            async def _retrieve_from(retriever: Any) -> list[dict[str, Any]]:
                try:
                    nodes = await retriever.aretrieve(query)
                    return [
                        {
                            "text": n.node.text or "",
                            "score": float(n.score) if n.score else 0.0,
                            "metadata": dict(n.node.metadata) if n.node.metadata else {},
                        }
                        for n in nodes
                    ]
                except Exception as e:
                    logger.warning("Hybrid retrieval failed, skipping: %s", e)
                    return []

            tasks = [_retrieve_from(r) for r in retrievers]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Fallback: dense-only search per collection
            tasks = [
                search(
                    query=query,
                    collection_name=col,
                    top_k=per_collection_k,
                    similarity_threshold=self.similarity_threshold,
                    qdrant_filter=qdrant_filter,
                )
                for col in collections
            ]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        rrf_k = self._rrf_k
        rrf_weights = self._rrf_weights

        rrf_scores: dict[tuple[str, str], dict[str, Any]] = {}
        for col_idx, col_results in enumerate(all_results):
            if isinstance(col_results, Exception):
                logger.warning(
                    "Collection search failed during RRF fusion, skipping: %s", col_results
                )
                continue
            results_list: list[dict[str, Any]] = col_results  # type: ignore[assignment]
            col_name = collections[col_idx] if col_idx < len(collections) else ""
            weight = rrf_weights.get(col_name, 1.0)
            for rank, result in enumerate(results_list, start=1):
                rrf_score = weight / (rrf_k + rank)
                text = result.get("text", "")
                meta = result.get("metadata", {})
                file_path = meta.get("file_path", "") if isinstance(meta, dict) else ""
                dedup_key = (file_path, text[:64])
                if dedup_key not in rrf_scores:
                    result["rrf_score"] = rrf_score
                    rrf_scores[dedup_key] = result
                else:
                    rrf_scores[dedup_key]["rrf_score"] += rrf_score

        fused = sorted(rrf_scores.values(), key=lambda r: r.get("rrf_score", 0), reverse=True)
        return fused[:limit]

    async def search_collection(
        self, query: str, collection_name: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Search a single specific collection using direct Qdrant client."""
        limit = top_k if top_k is not None else self.top_k
        return await search(
            query=query,
            collection_name=collection_name,
            top_k=limit,
            similarity_threshold=self.similarity_threshold,
        )

    def format_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Format search result."""
        return format_search_result(result)

    def get_citation(self, result: dict[str, Any]) -> str:
        """Get citation for result."""
        return get_citation(result)
