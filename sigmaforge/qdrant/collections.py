from __future__ import annotations

from qdrant_client import models

from sigmaforge.errors import QdrantError
from sigmaforge.qdrant.connection import QdrantConnection

DEFAULT_COLLECTIONS: tuple[str, ...] = ("sigma_rules", "sigma_docs", "sigma_spec")
DEFAULT_VECTOR_SIZE = 384
SPARSE_NAME = "text-sparse"

ON_DISK_BY_COLLECTION: dict[str, bool] = {
    "sigma_rules": False,
    "sigma_docs": True,
    "sigma_spec": True,
}

HNSW_BY_COLLECTION: dict[str, models.HnswConfigDiff] = {
    "sigma_rules": models.HnswConfigDiff(
        m=16,
        ef_construct=200,
        full_scan_threshold=10_000,
    ),
    "sigma_docs": models.HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10_000,
    ),
    "sigma_spec": models.HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10_000,
    ),
}

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "source": models.PayloadSchemaType.KEYWORD,
    "source_file": models.PayloadSchemaType.KEYWORD,
    "collection": models.PayloadSchemaType.KEYWORD,
    "chunk_type": models.PayloadSchemaType.KEYWORD,
    "rule_id": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.KEYWORD,
    "author": models.PayloadSchemaType.KEYWORD,
    "level": models.PayloadSchemaType.KEYWORD,
    "status": models.PayloadSchemaType.KEYWORD,
    "product": models.PayloadSchemaType.KEYWORD,
    "category": models.PayloadSchemaType.KEYWORD,
    "service": models.PayloadSchemaType.KEYWORD,
    "modified": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "references": models.PayloadSchemaType.KEYWORD,
}


class QdrantCollectionManager:
    def __init__(
        self,
        connection: QdrantConnection,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        enable_hybrid: bool = True,
        collections: tuple[str, ...] = DEFAULT_COLLECTIONS,
        sparse_name: str = SPARSE_NAME,
    ) -> None:
        self.connection = connection
        self.vector_size = vector_size
        self.enable_hybrid = enable_hybrid
        self.collections = collections
        self.sparse_name = sparse_name

    def client(self):
        return self.connection.client()

    def exists(self, name: str) -> bool:
        try:
            collections = self.client().get_collections().collections
        except Exception as exc:
            raise QdrantError(str(exc)) from exc
        return any(collection.name == name for collection in collections)

    def ensure(self, names: tuple[str, ...] | None = None) -> None:
        for name in names or self.collections:
            self.ensure_collection(name)

    def ensure_collection(self, name: str) -> None:
        if self.exists(name):
            return
        vectors_config = {"dense": self._dense_params(name)}
        sparse_vectors_config = None
        if self.enable_hybrid:
            sparse_vectors_config = {
                self.sparse_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        try:
            self.client().create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        except Exception as exc:
            raise QdrantError(str(exc)) from exc
        for field_name, field_schema in PAYLOAD_INDEXES.items():
            try:
                self.client().create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as exc:
                raise QdrantError(str(exc)) from exc

    def delete_collection(self, name: str) -> None:
        if not self.exists(name):
            return
        try:
            self.client().delete_collection(collection_name=name)
        except Exception as exc:
            raise QdrantError(str(exc)) from exc

    def count(self, name: str) -> int:
        try:
            return int(self.client().count(collection_name=name, exact=True).count)
        except Exception as exc:
            raise QdrantError(str(exc)) from exc

    def collection_info(self, name: str) -> dict[str, object]:
        try:
            info = self.client().get_collection(collection_name=name)
        except Exception as exc:
            raise QdrantError(str(exc)) from exc
        return {
            "name": name,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    def _dense_params(self, name: str) -> models.VectorParams:
        return models.VectorParams(
            size=self.vector_size,
            distance=models.Distance.COSINE,
            hnsw_config=HNSW_BY_COLLECTION.get(name),
            on_disk=ON_DISK_BY_COLLECTION.get(name, False),
        )
