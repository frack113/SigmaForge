from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from qdrant_client import models

from sigmaforge.errors import QdrantError
from sigmaforge.qdrant.collections import QdrantCollectionManager


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(
        self,
        manager: QdrantCollectionManager,
        sparse_name: str | None = None,
    ) -> None:
        self.manager = manager
        self.sparse_name = sparse_name or manager.sparse_name

    @staticmethod
    def make_point_id(collection: str, payload: Mapping[str, Any]) -> str:
        source_file = payload.get("source_file") or payload.get("source")
        chunk_type = payload.get("chunk_type")
        if source_file and chunk_type:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{source_file}:{chunk_type}"))
        return str(uuid.uuid4())

    def upsert_texts(
        self,
        collection: str,
        texts: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        payloads: Sequence[Mapping[str, Any]],
        sparse_vectors: Sequence[tuple[Sequence[int], Sequence[float]]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        if len(texts) != len(dense_vectors) or len(texts) != len(payloads):
            raise ValueError("texts, dense_vectors, and payloads must have the same length")
        points: list[models.PointStruct] = []
        for index, text in enumerate(texts):
            payload = dict(payloads[index])
            payload["text"] = text
            point_id = str(ids[index]) if ids is not None else self.make_point_id(collection, payload)
            vector: dict[str, Any] = {"dense": list(dense_vectors[index])}
            if self.manager.enable_hybrid:
                indices: Sequence[int] = []
                values: Sequence[float] = []
                if sparse_vectors is not None:
                    indices, values = sparse_vectors[index]
                vector[self.sparse_name] = models.SparseVector(
                    indices=[int(item) for item in indices],
                    values=[float(item) for item in values],
                )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
        try:
            self.manager.client().upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise QdrantError(str(exc)) from exc

    def replace_source_points(
        self,
        collection: str,
        source_file: str,
        texts: Sequence[str],
        dense_vectors: Sequence[Sequence[float]],
        payloads: Sequence[Mapping[str, Any]],
        sparse_vectors: Sequence[tuple[Sequence[int], Sequence[float]]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> None:
        self.delete_by_source(collection, source_file)
        self.upsert_texts(
            collection=collection,
            texts=texts,
            dense_vectors=dense_vectors,
            payloads=payloads,
            sparse_vectors=sparse_vectors,
            ids=ids,
        )

    def delete_by_source(self, collection: str, source_file: str) -> None:
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file),
                )
            ]
        )
        try:
            self.manager.client().delete(
                collection_name=collection,
                points_selector=filter,
                wait=True,
            )
        except Exception as exc:
            raise QdrantError(str(exc)) from exc

    def query_dense(
        self,
        collection: str,
        vector: Sequence[float],
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[SearchHit]:
        try:
            response = self.manager.client().query_points(
                collection_name=collection,
                query=list(vector),
                using="dense",
                limit=limit,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise QdrantError(str(exc)) from exc
        return [self._to_hit(point) for point in response.points]

    def query_sparse(
        self,
        collection: str,
        indices: Sequence[int],
        values: Sequence[float],
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[SearchHit]:
        if not self.manager.enable_hybrid or not indices:
            return []
        try:
            response = self.manager.client().query_points(
                collection_name=collection,
                query=models.SparseVector(
                    indices=[int(item) for item in indices],
                    values=[float(item) for item in values],
                ),
                using=self.sparse_name,
                limit=limit,
                query_filter=query_filter,
            )
        except Exception as exc:
            raise QdrantError(str(exc)) from exc
        return [self._to_hit(point) for point in response.points]

    def _to_hit(self, point: models.ScoredPoint) -> SearchHit:
        return SearchHit(
            id=str(point.id),
            score=float(point.score),
            payload=dict(point.payload or {}),
        )
