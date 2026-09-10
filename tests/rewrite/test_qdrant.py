from __future__ import annotations

from typing import Iterator

import pytest

from sigmaforge.qdrant import QdrantCollectionManager, QdrantConnection, QdrantStore


@pytest.fixture
def connection() -> Iterator[QdrantConnection]:
    conn = QdrantConnection(location=":memory:")
    yield conn
    conn.close()


@pytest.fixture
def store(connection: QdrantConnection) -> QdrantStore:
    manager = QdrantCollectionManager(
        connection,
        vector_size=4,
        collections=("sigma_docs",),
        enable_hybrid=True,
    )
    manager.ensure()
    return QdrantStore(manager)


def test_manager_creates_collection(store: QdrantStore) -> None:
    assert store.manager.exists("sigma_docs")
    assert store.manager.count("sigma_docs") == 0
    info = store.manager.collection_info("sigma_docs")
    assert info["name"] == "sigma_docs"


def test_store_upsert_and_query(store: QdrantStore) -> None:
    texts = ["windows powershell command execution"]
    dense_vectors = [[0.1, 0.2, 0.3, 0.4]]
    payloads = [{"source_file": "win.yaml", "chunk_type": "rule", "source": "local"}]
    sparse_vectors = [((0, 1), (0.1, 0.2))]

    store.upsert_texts(
        collection="sigma_docs",
        texts=texts,
        dense_vectors=dense_vectors,
        payloads=payloads,
        sparse_vectors=sparse_vectors,
    )

    assert store.manager.count("sigma_docs") == 1
    dense_hits = store.query_dense("sigma_docs", [0.1, 0.2, 0.3, 0.4], limit=1)
    assert len(dense_hits) == 1
    assert dense_hits[0].payload["text"] == texts[0]
    assert dense_hits[0].payload["source_file"] == "win.yaml"

    sparse_hits = store.query_sparse("sigma_docs", [0, 1], [0.1, 0.2], limit=1)
    assert len(sparse_hits) == 1
    assert sparse_hits[0].id == dense_hits[0].id


def test_store_delete_by_source(store: QdrantStore) -> None:
    store.upsert_texts(
        collection="sigma_docs",
        texts=["one", "two"],
        dense_vectors=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        payloads=[
            {"source_file": "win.yaml", "chunk_type": "rule"},
            {"source_file": "win.yaml", "chunk_type": "spec"},
        ],
    )

    store.delete_by_source("sigma_docs", "win.yaml")

    assert store.manager.count("sigma_docs") == 0


def test_point_id_is_deterministic() -> None:
    payload = {"source_file": "win.yaml", "chunk_type": "rule"}
    first = QdrantStore.make_point_id("sigma_docs", payload)
    second = QdrantStore.make_point_id("sigma_docs", payload)
    other = QdrantStore.make_point_id(
        "sigma_docs", {"source_file": "win.yaml", "chunk_type": "spec"}
    )

    assert first == second
    assert first != other
