"""Collection management for Qdrant."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import qdrant_client
from qdrant_client.http.exceptions import UnexpectedResponse

from .client import get_qdrant_client

logger = logging.getLogger(__name__)


def collection_hnsw_config(
    collection_name: str,
) -> qdrant_client.models.HnswConfigDiff | None:
    """Return per-collection HNSW config.

    ``sigma_rules`` is kept in-RAM with higher ``ef_construct`` for
    better recall.  ``sigma_docs`` and ``sigma_spec`` (cold collections)
    use on-disk storage to save RAM.
    """
    if collection_name == "sigma_rules":
        return qdrant_client.models.HnswConfigDiff(
            m=16,
            ef_construct=200,
            full_scan_threshold=10000,
            on_disk=False,
        )
    if collection_name in ("sigma_docs", "sigma_spec"):
        return qdrant_client.models.HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10000,
            on_disk=True,
        )
    return None


def _get_collections_sync(client) -> Any:
    """Synchronous wrapper for client.get_collections()."""
    return client.get_collections()


def _get_collection_sync(client, collection_name: str):
    """Synchronous wrapper for client.get_collection()."""
    return client.get_collection(collection_name=collection_name)


def _count_sync(client, collection_name: str) -> int:
    """Synchronous wrapper for client.count()."""
    result = client.count(collection_name=collection_name)
    return result.count if result else 0


def _create_collection_sync(
    client,
    collection_name: str,
    vectors_config,
    sparse_vectors_config=None,
    quantization_config=None,
    hnsw_config=None,
):
    """Synchronous wrapper for client.create_collection()."""
    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "vectors_config": vectors_config,
    }
    if sparse_vectors_config is not None:
        kwargs["sparse_vectors_config"] = sparse_vectors_config
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if hnsw_config is not None:
        kwargs["hnsw_config"] = hnsw_config
    client.create_collection(**kwargs)


PAYLOAD_INDEXES: list[tuple[str, str]] = [
    ("rule_id", "keyword"),
    ("title", "text"),
    ("author", "keyword"),
    ("level", "keyword"),
    ("status", "keyword"),
    ("product", "keyword"),
    ("category", "keyword"),
    ("service", "keyword"),
    ("date", "keyword"),
    ("modified", "keyword"),
    ("chunk_type", "keyword"),
    ("collection", "keyword"),
    ("tags", "keyword"),
    ("references", "keyword"),
]


def _create_payload_indexes_sync(client, collection_name: str) -> None:
    """Create payload indexes for filterable fields."""
    for field_name, field_type in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
                wait=False,
            )
        except Exception:
            logger.debug(
                "Payload index '%s' already exists or failed for '%s'", field_name, collection_name
            )


def _delete_collection_sync(client, collection_name: str):
    """Synchronous wrapper for client.delete_collection()."""
    client.delete_collection(collection_name=collection_name)


async def list_collections(host: str, port: int) -> list[dict[str, Any]]:
    """List all collections with detailed info."""
    client = get_qdrant_client(host=host, port=port)
    try:
        collections_response = await asyncio.to_thread(_get_collections_sync, client)

        detailed_collections = []
        for info in collections_response.collections:
            name = getattr(info, "name", None)
            if not name:
                continue
            details = await get_collection(host, port, name)
            details["name"] = name
            detailed_collections.append(details)
        return detailed_collections
    except Exception as e:
        logger.error("Failed to list collections: %s: %s", type(e).__name__, e, exc_info=True)
        raise
    finally:
        client.close()


async def create_collection(
    host: str,
    port: int,
    collection_name: str,
    vector_size: int = 384,
    enable_hybrid: bool = True,
    enable_quantization: bool = True,
) -> bool:
    """Create a new collection (with optional sparse vector support for hybrid search).

    When *enable_quantization* is True, ScalarQuantization INT8 is applied
    to reduce vector memory footprint by ~4x with minimal recall loss when
    combined with rescore.

    HNSW config is selected per-collection via :func:`collection_hnsw_config`:
    ``sigma_rules`` uses in-RAM HNSW with higher ``ef_construct`` for
    better recall, while ``sigma_docs`` and ``sigma_spec`` use on-disk
    storage to save RAM.
    """
    client = get_qdrant_client(host=host, port=port)
    try:
        vectors_config = qdrant_client.models.VectorParams(
            size=vector_size,
            distance=qdrant_client.models.Distance.COSINE,
        )
        sparse_vectors_config: dict[str, Any] | None = None
        if enable_hybrid:
            sparse_vectors_config = {
                "text-sparse": qdrant_client.models.SparseVectorParams(
                    index=qdrant_client.models.SparseIndexParams(),
                    modifier=qdrant_client.models.Modifier.IDF,
                )
            }
        quantization_config = None
        if enable_quantization:
            quantization_config = qdrant_client.models.ScalarQuantization(
                scalar=qdrant_client.models.ScalarQuantizationConfig(
                    type=qdrant_client.models.ScalarType.INT8,
                    always_ram=True,
                    quantile=0.5,
                )
            )
        hnsw_config = collection_hnsw_config(collection_name)
        await asyncio.to_thread(
            _create_collection_sync,
            client,
            collection_name,
            vectors_config,
            sparse_vectors_config,
            quantization_config,
            hnsw_config,
        )
        await asyncio.to_thread(_create_payload_indexes_sync, client, collection_name)
        logger.info(
            "Collection '%s' created successfully (hybrid=%s).", collection_name, enable_hybrid
        )
        return True
    except UnexpectedResponse as e:
        # Collection already exists (409 Conflict) - that is acceptable
        if "already exists" in str(e):
            logger.info("Collection '%s' already exists.", collection_name)
            return True
        logger.error(
            "Qdrant error for collection '%s': %s",
            collection_name,
            e,
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(
            "Failed to create collection '%s': %s: %s",
            collection_name,
            type(e).__name__,
            e,
            exc_info=True,
        )
        raise
    finally:
        client.close()


async def delete_collection(host: str, port: int, collection_name: str) -> bool:
    """Delete an existing collection."""
    client = get_qdrant_client(host=host, port=port)
    try:
        await asyncio.to_thread(_delete_collection_sync, client, collection_name)
        logger.info("Collection '%s' deleted successfully.", collection_name)
        return True
    except Exception as e:
        logger.error(
            "Failed to delete collection '%s': %s: %s",
            collection_name,
            type(e).__name__,
            e,
            exc_info=True,
        )
        raise
    finally:
        client.close()


async def get_collection(host: str, port: int, collection_name: str) -> dict[str, Any]:
    """Get information about a collection with defensive extraction."""
    client = get_qdrant_client(host=host, port=port)
    try:
        info = await asyncio.to_thread(_get_collection_sync, client, collection_name)

        points_count = 0
        try:
            points_count = await asyncio.to_thread(_count_sync, client, collection_name)
        except Exception:
            pass

        info_str = str(info)

        shards = 1
        shard_match = re.search(r"shard_number=(\d+)", info_str)
        if shard_match:
            shards = int(shard_match.group(1))
        elif "shard_number" in info_str:
            match = re.search(r"(\d+)", info_str.split("shard_number")[-1].split(",")[0])
            if match:
                shards = int(match.group(1))

        vector_size = 384
        size_match = re.search(r"size=(\d+)", info_str)
        if size_match:
            vector_size = int(size_match.group(1))
        else:
            size_fallback = re.search(r"(\d+)", info_str.split("vectors_config")[-1].split(",")[0])
            if size_fallback:
                vector_size = int(size_fallback.group(1))

        return {
            "points": points_count,
            "shards": shards,
            "vector_size": vector_size,
            "status": str(info.status),
        }
    except Exception as e:
        logger.error(
            "Failed to get collection '%s': %s: %s",
            collection_name,
            type(e).__name__,
            e,
            exc_info=True,
        )
        raise
    finally:
        client.close()
