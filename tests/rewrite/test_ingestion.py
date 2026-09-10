from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sigmaforge.embed import Bm25SparseEncoder, DenseEncoder
from sigmaforge.ingestion import (
    Chunk,
    IngestRequest,
    IngestionPipeline,
    SigmaRule,
    chunk_rule,
    flat_rule_text,
    parse_sigma_rule,
    rich_rule_chunks,
    rule_metadata,
    split_text,
)
from sigmaforge.qdrant import QdrantCollectionManager, QdrantConnection, QdrantStore


class FakeModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(
        self,
        texts,
        batch_size: int | None = None,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ):
        return [[0.1] * self.dim for _ in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    conn = QdrantConnection(location=":memory:")
    manager = QdrantCollectionManager(
        conn,
        vector_size=4,
        collections=("sigma_rules", "sigma_docs"),
        enable_hybrid=True,
    )
    manager.ensure()
    yield QdrantStore(manager)
    conn.close()


@pytest.fixture
def pipeline(store: QdrantStore) -> IngestionPipeline:
    return IngestionPipeline(
        store,
        DenseEncoder("fake-model", model=FakeModel(dim=4)),
        Bm25SparseEncoder(),
    )


def _rule() -> SigmaRule:
    return SigmaRule(
        id="suspicious",
        title="Suspicious Rule",
        condition="selection of process_name",
        logsource={"product": "windows", "category": "process_creation"},
        detection={"process_name": {"contains": "powershell"}},
    )


def _write_rule(path: Path, rule_id: str = "test_rule") -> Path:
    path.write_text(
        f"id: {rule_id}\n"
        "title: Test Rule\n"
        "description: Detects suspicious activity.\n"
        "condition: selection of process_name\n"
        "logsource:\n"
        "  product: windows\n"
        "  category: process_creation\n"
        "detection:\n"
        "  process_name:\n"
        "    contains: powershell\n",
        encoding="utf-8",
    )
    return path


def test_parse_sigma_rule(tmp_path: Path) -> None:
    path = _write_rule(tmp_path / "rule.yaml")

    rule = parse_sigma_rule(path)

    assert rule is not None
    assert rule.id == "test_rule"
    assert rule.title == "Test Rule"
    assert rule.file_path == str(path)
    assert rule.logsource["product"] == "windows"


def test_parse_sigma_rule_rejects_non_rule(tmp_path: Path) -> None:
    path = tmp_path / "not-rule.yaml"
    path.write_text("title: not a rule\n", encoding="utf-8")

    assert parse_sigma_rule(path) is None


def test_parse_sigma_rule_rejects_non_yaml_extension(tmp_path: Path) -> None:
    path = tmp_path / "rule.txt"
    path.write_text("detection:\n  process_name: 1\n", encoding="utf-8")

    assert parse_sigma_rule(path) is None


def test_flat_rule_text_contains_core_fields() -> None:
    text = flat_rule_text(_rule())

    assert "Title: Suspicious Rule" in text
    assert "Rule ID: suspicious" in text
    assert "Condition: selection of process_name" in text
    assert "product=windows" in text
    assert "process_name: contains powershell" in text


def test_rule_metadata_exposes_search_fields() -> None:
    metadata = rule_metadata(_rule())

    assert metadata["rule_id"] == "suspicious"
    assert metadata["title"] == "Suspicious Rule"
    assert metadata["product"] == "windows"
    assert metadata["category"] == "process_creation"


def test_rich_rule_chunks_creates_unique_chunks() -> None:
    chunks = rich_rule_chunks(_rule())
    chunk_types = [chunk.chunk_type for chunk in chunks]

    assert len(chunks) > 1
    assert "executive_summary" in chunk_types
    assert any(chunk_type.startswith("detection_block:") for chunk_type in chunk_types)
    assert len({chunk.point_id("sigma_rules") for chunk in chunks}) == len(chunks)


def test_chunk_rule_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        chunk_rule(_rule(), "bogus")


def test_split_text_handles_empty() -> None:
    assert split_text("   ") == []


def test_split_text_short_text_single_chunk() -> None:
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_split_text_long_text_respects_chunk_size() -> None:
    text = " ".join(f"token{i}" for i in range(1000))

    chunks = split_text(text, chunk_size=80, chunk_overlap=10)

    assert chunks
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert chunks[0].startswith("token0")
    assert "token999" in chunks[-1]


def test_split_text_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("text", chunk_size=10, chunk_overlap=10)


def test_index_chunks_uses_stable_ids_for_same_chunk_type(store: QdrantStore) -> None:
    pipeline = IngestionPipeline(
        store,
        DenseEncoder("fake-model", model=FakeModel(dim=4)),
        Bm25SparseEncoder(),
    )
    chunks = [
        Chunk(text="one", source_file="source.txt", chunk_type="part", chunk_index=0),
        Chunk(text="two", source_file="source.txt", chunk_type="part", chunk_index=1),
    ]

    indexed = pipeline.index_chunks("sigma_docs", chunks)

    assert indexed == 2
    assert store.manager.count("sigma_docs") == 2


def test_ingest_sigma_directory_indexes_flat_rule(
    tmp_path: Path,
    store: QdrantStore,
    pipeline: IngestionPipeline,
) -> None:
    _write_rule(tmp_path / "rule.yaml")

    summary = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path)),
        collection="sigma_rules",
    )

    assert summary.total == 1
    assert summary.indexed == 1
    assert summary.failed == 0
    assert summary.chunks == 1
    assert store.manager.count("sigma_rules") == 1

    query_vector = pipeline.dense_encoder.encode(["powershell"])[0]
    hits = store.query_dense("sigma_rules", query_vector, limit=1)
    assert hits[0].payload["rule_id"] == "test_rule"
    assert hits[0].payload["chunk_type"] == "rule"
    assert hits[0].payload["source_file"].endswith("rule.yaml")


def test_ingest_sigma_directory_is_idempotent(
    tmp_path: Path,
    store: QdrantStore,
    pipeline: IngestionPipeline,
) -> None:
    _write_rule(tmp_path / "rule.yaml")

    first = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path)),
        collection="sigma_rules",
    )
    second = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path)),
        collection="sigma_rules",
    )

    assert first.chunks == second.chunks == 1
    assert store.manager.count("sigma_rules") == 1


def test_ingest_sigma_directory_rich_mode(
    tmp_path: Path,
    store: QdrantStore,
    pipeline: IngestionPipeline,
) -> None:
    _write_rule(tmp_path / "rule.yaml")

    summary = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path), mode="rich"),
        collection="sigma_rules",
    )

    assert summary.chunks > 1
    assert store.manager.count("sigma_rules") == summary.chunks

    query_vector = pipeline.dense_encoder.encode(["powershell"])[0]
    hits = store.query_dense("sigma_rules", query_vector, limit=summary.chunks)
    assert len(hits) == summary.chunks
    chunk_types = {hit.payload["chunk_type"] for hit in hits}
    assert "executive_summary" in chunk_types
    assert any(chunk_type.startswith("detection_block:") for chunk_type in chunk_types)


def test_ingest_sigma_directory_reports_invalid_rule(
    tmp_path: Path,
    store: QdrantStore,
    pipeline: IngestionPipeline,
) -> None:
    (tmp_path / "invalid.yaml").write_text("title: not a rule\n", encoding="utf-8")

    summary = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path)),
        collection="sigma_rules",
    )

    assert summary.total == 1
    assert summary.failed == 1
    assert summary.indexed == 0
    assert store.manager.count("sigma_rules") == 0
    assert summary.results[0].error is not None


def test_ingest_sigma_directory_respects_selected_dirs(
    tmp_path: Path,
    store: QdrantStore,
    pipeline: IngestionPipeline,
) -> None:
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    selected.mkdir()
    other.mkdir()
    _write_rule(selected / "a.yaml", rule_id="a")
    _write_rule(other / "b.yaml", rule_id="b")

    summary = pipeline.ingest_sigma_directory(
        IngestRequest(directory=str(tmp_path), selected_dirs=["selected"]),
        collection="sigma_rules",
    )

    assert summary.total == 1
    assert summary.indexed == 1
    query_vector = pipeline.dense_encoder.encode(["powershell"])[0]
    hits = store.query_dense("sigma_rules", query_vector, limit=1)
    assert hits[0].payload["rule_id"] == "a"


def test_ingest_text_directory_chunks_and_indexes(
    tmp_path: Path,
    store: QdrantStore,
) -> None:
    pipeline = IngestionPipeline(
        store,
        DenseEncoder("fake-model", model=FakeModel(dim=4)),
        Bm25SparseEncoder(),
        chunk_size=60,
        chunk_overlap=10,
    )
    (tmp_path / "doc.md").write_text("alpha beta gamma delta " * 20, encoding="utf-8")

    summary = pipeline.ingest_text_directory(
        IngestRequest(directory=str(tmp_path)),
        collection="sigma_docs",
    )

    assert summary.total == 1
    assert summary.indexed == 1
    assert summary.chunks > 1
    assert store.manager.count("sigma_docs") == summary.chunks

    query_vector = pipeline.dense_encoder.encode(["alpha"])[0]
    hits = store.query_dense("sigma_docs", query_vector, limit=summary.chunks)
    assert hits[0].payload["chunk_type"] == "doc"
    assert hits[0].payload["source_file"].endswith("doc.md")
