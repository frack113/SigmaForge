"""LlamaIndex retrievers for each Qdrant collection."""

from __future__ import annotations

import logging
import threading
from typing import Any

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from qdrant_client import AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config.settings import get_config
from src.infrastructure.vectorstore.client import get_qdrant_client

logger = logging.getLogger(__name__)

_async_client: AsyncQdrantClient | None = None
_async_client_lock = threading.Lock()

_index_cache: dict[tuple[str, float], VectorStoreIndex] = {}
_index_cache_lock = threading.Lock()


def _get_async_client() -> AsyncQdrantClient:
    global _async_client
    if _async_client is not None:
        return _async_client
    with _async_client_lock:
        if _async_client is not None:
            return _async_client
        cfg = get_config()
        _async_client = AsyncQdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    return _async_client


def reset_retriever_cache() -> None:
    """Clear cached indexes and the shared async client (call on model reset)."""
    global _async_client
    with _index_cache_lock:
        _index_cache.clear()
    with _async_client_lock:
        _async_client = None


def _build_llama_filters(
    qdrant_filter: Any | None,
) -> MetadataFilters | None:
    """Convert a Qdrant Filter to LlamaIndex MetadataFilters.

    Only handles simple Must conditions with MatchValue (exact match).
    Tags (MatchText) are skipped since LlamaIndex metadata filters use exact matching.
    """
    if qdrant_filter is None:
        return None

    conditions = getattr(qdrant_filter, "must", []) or []
    if not conditions:
        return None

    filters_list: list[MetadataFilter] = []
    for cond in conditions:
        key = getattr(cond, "key", "")
        match_val = getattr(cond, "match", None)
        value = getattr(match_val, "value", None) if match_val is not None else None
        if key and value is not None:
            filters_list.append(MetadataFilter(key=key, value=str(value)))

    return MetadataFilters(filters=filters_list) if filters_list else None  # type: ignore[arg-type]


def _build_index(collection_name: str, alpha: float) -> VectorStoreIndex:
    """Build (or retrieve from cache) a VectorStoreIndex for a collection."""
    cache_key = (collection_name, alpha)
    with _index_cache_lock:
        cached = _index_cache.get(cache_key)
        if cached is not None:
            return cached

    from src.core.search.engine import _get_search_embed_model
    from src.core.search.sparse_encoder import create_sparse_encoder

    client = get_qdrant_client()
    aclient = _get_async_client()
    sparse_encoder = create_sparse_encoder()
    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name=collection_name,
        enable_hybrid=True,
        sparse_doc_fn=sparse_encoder,
        sparse_query_fn=sparse_encoder,
        sparse_vector_name="text-sparse",
    )

    embed_model = _get_search_embed_model()
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    with _index_cache_lock:
        _index_cache[cache_key] = index
    return index


def get_collection_retriever(
    collection_name: str,
    top_k: int = 30,
    metadata_filter: Any | None = None,
    alpha: float = 0.3,
) -> VectorIndexRetriever:
    """Get a LlamaIndex retriever for a specific Qdrant collection.

    The underlying VectorStoreIndex is cached per (collection, alpha) pair.
    A lightweight VectorIndexRetriever is created per call with the
    per-query top_k and metadata_filter.

    Args:
        collection_name: Qdrant collection name.
        top_k: Number of results per collection (before fusion).
        metadata_filter: Optional Qdrant Filter to apply as metadata filter.
        alpha: Hybrid search weight (1.0=pure dense, 0.0=pure sparse, 0.3=keyword-leaning).

    Returns:
        Configured VectorIndexRetriever.
    """
    try:
        index = _build_index(collection_name, alpha)
        llama_filters = _build_llama_filters(metadata_filter)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
            filters=llama_filters,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            alpha=alpha,
        )
        return retriever
    except Exception as e:
        logger.warning("Failed to create retriever for '%s': %s", collection_name, e)
        raise
