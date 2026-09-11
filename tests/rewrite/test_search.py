from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from qdrant_client import models

from sigmaforge.embed import Bm25SparseEncoder
from sigmaforge.errors import SearchError
from sigmaforge.qdrant import QdrantCollectionManager, QdrantConnection, QdrantStore
from sigmaforge.search import (
    SearchEngine,
    SearchResult,
    build_qdrant_filter,
    format_context,
    format_result,
    get_citation,
    parse_query_filters,
    reciprocal_rank_fusion,
)


class FixedDenseEncoder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        self.calls.extend(texts)
        return [list(self.vector) for _ in texts]


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
        collections=("sigma_rules", "sigma_docs"),
        enable_hybrid=True,
    )
    manager.ensure()
    return QdrantStore(manager)


def _result(
    result_id: str,
    collection: str = "sigma_rules",
    text: str = "text",
    score: float = 1.0,
    payload: dict[str, Any] | None = None,
) -> SearchResult:
    return SearchResult(
        id=result_id,
        collection=collection,
        text=text,
        score=score,
        payload=payload if payload is not None else {"source_file": f"{result_id}.yaml"},
    )


def _upsert_point(
    store: QdrantStore,
    collection: str,
    text: str,
    payload: dict[str, Any],
    dense: list[float] | None = None,
) -> None:
    sparse_encoder = Bm25SparseEncoder()
    vector = sparse_encoder.encode_text(text)
    store.upsert_texts(
        collection=collection,
        texts=[text],
        dense_vectors=[dense or [1.0, 0.0, 0.0, 0.0]],
        payloads=[payload],
        sparse_vectors=[(vector.indices, vector.values)],
    )


def test_parse_query_filters_extracts_known_filters() -> None:
    filters, cleaned = parse_query_filters("powershell rule_id:alpha level:high")

    assert filters == {"rule_id": "alpha", "level": "high"}
    assert cleaned == "powershell"


def test_parse_query_filters_keeps_unknown_colons() -> None:
    filters, cleaned = parse_query_filters("process_name:powershell rule_id:alpha")

    assert filters == {"rule_id": "alpha"}
    assert cleaned == "process_name:powershell"


def test_parse_query_filters_returns_empty_cleaned_for_only_filters() -> None:
    filters, cleaned = parse_query_filters("rule_id:alpha")

    assert filters == {"rule_id": "alpha"}
    assert cleaned == ""


def test_build_qdrant_filter_returns_none_without_filters() -> None:
    assert build_qdrant_filter() is None
    assert build_qdrant_filter({}) is None


def test_build_qdrant_filter_uses_scalar_match_value() -> None:
    qfilter = build_qdrant_filter({"rule_id": "alpha"})

    assert qfilter is not None
    data = qfilter.model_dump()
    condition = data["must"][0]
    assert condition["key"] == "rule_id"
    assert condition["match"]["value"] == "alpha"


def test_build_qdrant_filter_uses_match_any_for_list_filters() -> None:
    qfilter = build_qdrant_filter({"tags": "alpha beta"})

    assert qfilter is not None
    data = qfilter.model_dump()
    condition = data["must"][0]
    assert condition["key"] == "tags"
    assert condition["match"]["any"] == ["alpha", "beta"]


def test_build_qdrant_filter_combines_extra_filter() -> None:
    extra = models.Filter(
        must=[models.FieldCondition(key="status", match=models.MatchValue(value="active"))]
    )

    qfilter = build_qdrant_filter({"rule_id": "alpha"}, extra)

    assert qfilter is not None
    data = qfilter.model_dump()
    assert isinstance(data["must"], list)
    assert len(data["must"]) == 2


def test_reciprocal_rank_fusion_merges_rankings() -> None:
    first = _result("1")
    second = _result("2")
    duplicate = _result("1", score=0.5)

    fused = reciprocal_rank_fusion({"dense": [first, second], "sparse": [duplicate]})

    assert [result.id for result in fused] == ["1", "2"]
    assert fused[0].score == pytest.approx(2 / 61)
    assert first.score == 1.0


def test_reciprocal_rank_fusion_ignores_non_positive_weights() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [_result("1")], "sparse": [_result("2")]},
        weights={"dense": 1.0, "sparse": 0.0},
    )

    assert [result.id for result in fused] == ["1"]


def test_reciprocal_rank_fusion_limits_results() -> None:
    fused = reciprocal_rank_fusion({"dense": [_result(str(i)) for i in range(5)]}, limit=2)

    assert len(fused) == 2


def test_reciprocal_rank_fusion_rejects_non_positive_k() -> None:
    with pytest.raises(SearchError):
        reciprocal_rank_fusion({"dense": []}, k=0)


def test_format_context_orders_and_labels_results() -> None:
    results = [
        _result("1", text="alpha", payload={"source_file": "a.yaml", "title": "Alpha"}),
        _result("2", collection="sigma_docs", text="beta", payload={"source_file": "b.md"}),
    ]

    context = format_context(results, max_results=1)

    assert context.startswith("[1] sigma_rules / a.yaml / Alpha")
    assert "alpha" in context
    assert "[2]" not in context


def test_format_context_respects_budget() -> None:
    result = _result("1", text="a" * 100)

    context = format_context([result], max_chars=50)

    assert len(context) <= 50
    assert context.endswith("...")


def test_get_citation_uses_source_and_line() -> None:
    result = _result("1", payload={"source_file": "a.yaml", "line_start": 10})

    assert get_citation(result) == "a.yaml:10"

    result_without_line = _result("1", payload={"source_file": "a.yaml"})
    assert get_citation(result_without_line) == "a.yaml"


def test_format_result_exposes_collection_specific_fields() -> None:
    rule = _result(
        "1",
        text="rule text",
        payload={"source_file": "a.yaml", "rule_id": "r", "title": "T", "level": "high"},
    )
    formatted = format_result(rule)

    assert formatted["rule_id"] == "r"
    assert formatted["title"] == "T"

    doc = _result(
        "2",
        collection="sigma_docs",
        text="doc text",
        payload={"source_file": "b.md", "doc_type": "doc", "original_url": "https://example.com"},
    )
    formatted_doc = format_result(doc)

    assert formatted_doc["doc_type"] == "doc"
    assert formatted_doc["original_url"] == "https://example.com"


def test_search_empty_query_returns_empty_and_does_not_encode(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    engine = SearchEngine(store, encoder)

    assert engine.search("") == []
    assert engine.search("???") == []
    assert encoder.calls == []


def test_search_normalizes_query(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "ps.yaml", "chunk_type": "rule", "rule_id": "ps"},
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("`powershell`?")

    assert len(results) == 1
    assert encoder.calls == ["powershell"]


def test_search_falls_back_to_normalized_query_when_only_filters(
    store: QdrantStore,
) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "ps.yaml", "chunk_type": "rule", "rule_id": "alpha"},
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("rule_id:alpha")

    assert len(results) == 1
    assert results[0].metadata["rule_id"] == "alpha"
    assert encoder.calls == ["rule_id:alpha"]


def test_search_returns_filtered_results(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {
            "source_file": "alpha.yaml",
            "chunk_type": "rule",
            "rule_id": "alpha",
            "status": "active",
        },
    )
    _upsert_point(
        store,
        "sigma_rules",
        "wmi process creation",
        {
            "source_file": "beta.yaml",
            "chunk_type": "rule",
            "rule_id": "beta",
            "status": "disabled",
        },
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("powershell rule_id:alpha", top_k=5)

    assert len(results) == 1
    assert results[0].metadata["rule_id"] == "alpha"
    assert encoder.calls == ["powershell"]


def test_search_extra_filter_applies_to_results(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {
            "source_file": "alpha.yaml",
            "chunk_type": "rule",
            "rule_id": "alpha",
            "status": "active",
        },
    )
    _upsert_point(
        store,
        "sigma_rules",
        "wmi process creation",
        {
            "source_file": "beta.yaml",
            "chunk_type": "rule",
            "rule_id": "beta",
            "status": "disabled",
        },
    )
    engine = SearchEngine(store, encoder, alpha=1.0)
    extra_filter = models.Filter(
        must=[models.FieldCondition(key="status", match=models.MatchValue(value="active"))]
    )

    results = engine.search("powershell", extra_filter=extra_filter)

    assert len(results) == 1
    assert results[0].metadata["rule_id"] == "alpha"


def test_search_list_filter_matches_any_tag(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {
            "source_file": "alpha.yaml",
            "chunk_type": "rule",
            "rule_id": "alpha",
            "tags": ["powershell", "windows"],
        },
    )
    _upsert_point(
        store,
        "sigma_rules",
        "wmi process creation",
        {
            "source_file": "beta.yaml",
            "chunk_type": "rule",
            "rule_id": "beta",
            "tags": ["wmi"],
        },
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("powershell tags:powershell")

    assert len(results) == 1
    assert results[0].metadata["rule_id"] == "alpha"


def test_search_references_filter_includes_sigma_docs(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "alpha.yaml", "chunk_type": "rule", "rule_id": "alpha"},
    )
    _upsert_point(
        store,
        "sigma_docs",
        "powershell documentation",
        {
            "source_file": "doc.md",
            "chunk_type": "doc",
            "references": "docref",
        },
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("powershell references:docref", collections=("sigma_rules",))

    assert len(results) == 1
    assert results[0].collection == "sigma_docs"


def test_search_collection_queries_single_collection(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "alpha.yaml", "chunk_type": "rule", "rule_id": "alpha"},
    )
    _upsert_point(
        store,
        "sigma_docs",
        "powershell documentation",
        {"source_file": "doc.md", "chunk_type": "doc", "references": "docref"},
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search_collection("sigma_rules", "powershell")

    assert len(results) == 1
    assert results[0].collection == "sigma_rules"


def test_search_sparse_only_returns_matching_document(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {
            "source_file": "alpha.yaml",
            "chunk_type": "rule",
            "rule_id": "powershell",
        },
        dense=[0.0, 1.0, 0.0, 0.0],
    )
    _upsert_point(
        store,
        "sigma_rules",
        "wmi event",
        {"source_file": "beta.yaml", "chunk_type": "rule", "rule_id": "wmi"},
        dense=[0.0, 1.0, 0.0, 0.0],
    )
    engine = SearchEngine(store, encoder, Bm25SparseEncoder(), alpha=0.0)

    results = engine.search("powershell", collections=("sigma_rules",))

    assert len(results) == 1
    assert results[0].metadata["rule_id"] == "powershell"


def test_search_sparse_only_skips_stopword_only_query(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "alpha.yaml", "chunk_type": "rule", "rule_id": "alpha"},
    )
    engine = SearchEngine(store, encoder, Bm25SparseEncoder(), alpha=0.0)

    results = engine.search("the", collections=("sigma_rules",))

    assert results == []


def test_search_similarity_threshold_filters_low_scores(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([0.0, 1.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "alpha.yaml", "chunk_type": "rule", "rule_id": "alpha"},
        dense=[1.0, 0.0, 0.0, 0.0],
    )
    engine = SearchEngine(store, encoder, alpha=1.0, similarity_threshold=0.5)

    results = engine.search("powershell", collections=("sigma_rules",))

    assert results == []


def test_search_top_k_limits_results(store: QdrantStore) -> None:
    encoder = FixedDenseEncoder([1.0, 0.0, 0.0, 0.0])
    _upsert_point(
        store,
        "sigma_rules",
        "powershell process creation",
        {"source_file": "alpha.yaml", "chunk_type": "rule", "rule_id": "alpha"},
    )
    _upsert_point(
        store,
        "sigma_rules",
        "wmi process creation",
        {"source_file": "beta.yaml", "chunk_type": "rule", "rule_id": "beta"},
    )
    engine = SearchEngine(store, encoder, alpha=1.0)

    results = engine.search("powershell", top_k=1)

    assert len(results) == 1
