from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from sigmaforge.db.docs import DocRegistry
from sigmaforge.embed.dense import DenseEncoder
from sigmaforge.embed.sparse import Bm25SparseEncoder, SparseVector
from sigmaforge.ingestion.files import iter_files
from sigmaforge.ingestion.models import Chunk, IngestRequest, IngestResult, IngestSummary
from sigmaforge.ingestion.sigma import (
    SigmaRule,
    chunk_rule,
    iter_sigma_rule_files,
    parse_sigma_rule,
)
from sigmaforge.ingestion.text import split_text
from sigmaforge.qdrant.store import QdrantStore

logger = logging.getLogger(__name__)

DEFAULT_TEXT_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".rst", ".txt", ".adoc")


class IngestionPipeline:
    def __init__(
        self,
        store: QdrantStore,
        dense_encoder: DenseEncoder,
        sparse_encoder: Bm25SparseEncoder | None = None,
        registry: DocRegistry | None = None,
        *,
        default_directory: str | Path | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        enable_hybrid: bool | None = None,
    ) -> None:
        self.store = store
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.registry = registry
        self.default_directory = default_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_hybrid = (
            store.manager.enable_hybrid if enable_hybrid is None else bool(enable_hybrid)
        )

    def index_chunks(self, collection: str, chunks: Sequence[Chunk]) -> int:
        prepared = [chunk for chunk in chunks if chunk.text.strip()]
        if not prepared:
            return 0
        self.store.manager.ensure_collection(collection)
        grouped: dict[str, list[Chunk]] = {}
        for chunk in prepared:
            grouped.setdefault(chunk.source_file, []).append(chunk)
        total = 0
        for source_file, source_chunks in grouped.items():
            texts = [chunk.text for chunk in source_chunks]
            dense_vectors = self.dense_encoder.encode(texts)
            sparse_vectors = self._encode_sparse(texts)
            payloads = [chunk.payload() for chunk in source_chunks]
            ids = [chunk.point_id(collection) for chunk in source_chunks]
            self.store.replace_source_points(
                collection=collection,
                source_file=source_file,
                texts=texts,
                dense_vectors=dense_vectors,
                payloads=payloads,
                sparse_vectors=sparse_vectors,
                ids=ids,
            )
            total += len(texts)
        return total

    def ingest_sigma_directory(
        self,
        request: IngestRequest | None = None,
        collection: str = "sigma_rules",
    ) -> IngestSummary:
        request = request or IngestRequest()
        if request.mode not in {"flat", "rich"}:
            raise ValueError("mode must be 'flat' or 'rich'")
        directory = self._resolve_directory(request.directory)
        files = iter_sigma_rule_files(
            directory,
            recursive=request.recursive,
            selected_dirs=request.selected_dirs,
        )
        results: list[IngestResult] = []
        chunks: list[Chunk] = []
        for path in files:
            result = IngestResult(file=str(path), success=False)
            try:
                rule = parse_sigma_rule(path)
                if rule is None:
                    raise ValueError("not a valid sigma rule")
                created = chunk_rule(rule, request.mode)
                result.success = True
                result.rule_id = rule.id
                result.chunks = len(created)
                chunks.extend(created)
            except Exception as exc:
                result.error = str(exc)
                logger.warning("Failed to ingest sigma rule %s: %s", path, exc)
            results.append(result)
            self._mark(str(path), collection, result.success)
        indexed_chunks = self.index_chunks(collection, chunks)
        return IngestSummary.from_results(collection, results, chunks=indexed_chunks)

    def ingest_sigma_rules(
        self,
        rules: Sequence[SigmaRule],
        collection: str = "sigma_rules",
        mode: str = "flat",
    ) -> IngestSummary:
        if mode not in {"flat", "rich"}:
            raise ValueError("mode must be 'flat' or 'rich'")
        results: list[IngestResult] = []
        chunks: list[Chunk] = []
        for rule in rules:
            source = rule.file_path or rule.id
            result = IngestResult(file=source, success=False, rule_id=rule.id)
            try:
                created = chunk_rule(rule, mode)
                result.success = True
                result.chunks = len(created)
                chunks.extend(created)
            except Exception as exc:
                result.error = str(exc)
                logger.warning("Failed to ingest sigma rule %s: %s", source, exc)
            results.append(result)
            self._mark(source, collection, result.success)
        indexed_chunks = self.index_chunks(collection, chunks)
        return IngestSummary.from_results(collection, results, chunks=indexed_chunks)

    def ingest_text_directory(
        self,
        request: IngestRequest | None = None,
        collection: str = "sigma_docs",
        extensions: Sequence[str] = DEFAULT_TEXT_EXTENSIONS,
        source: str = "local",
    ) -> IngestSummary:
        request = request or IngestRequest()
        directory = self._resolve_directory(request.directory)
        files = iter_files(
            directory,
            extensions=extensions,
            recursive=request.recursive,
            selected_dirs=request.selected_dirs,
        )
        results: list[IngestResult] = []
        chunks: list[Chunk] = []
        for path in files:
            result = IngestResult(file=str(path), success=False)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                pieces = split_text(text, self.chunk_size, self.chunk_overlap)
                result.success = True
                result.chunks = len(pieces)
                for index, piece in enumerate(pieces):
                    chunks.append(
                        Chunk(
                            text=piece,
                            source_file=str(path),
                            chunk_type="doc",
                            chunk_index=index,
                            metadata={"source": source, "title": path.stem},
                        )
                    )
            except Exception as exc:
                result.error = str(exc)
                logger.warning("Failed to ingest text file %s: %s", path, exc)
            results.append(result)
            self._mark(str(path), source, result.success)
        indexed_chunks = self.index_chunks(collection, chunks)
        return IngestSummary.from_results(collection, results, chunks=indexed_chunks)

    def _encode_sparse(
        self, texts: Sequence[str]
    ) -> Sequence[tuple[Sequence[int], Sequence[float]]] | None:
        if self.sparse_encoder is None or not self.enable_hybrid:
            return None
        vectors = self.sparse_encoder(texts)
        return [self._normalize_sparse(vector) for vector in vectors]

    def _mark(self, path: str, source: str, success: bool) -> None:
        if self.registry is None:
            return
        self.registry.mark(source, path, "indexed" if success else "failed")

    def _resolve_directory(self, directory: str | None) -> Path:
        candidate = Path(directory or self.default_directory or "")
        if not candidate:
            raise FileNotFoundError("No ingestion directory provided")
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Ingestion directory not found: {candidate}")
        return candidate

    @staticmethod
    def _normalize_sparse(
        vector: SparseVector | tuple[Sequence[int], Sequence[float]],
    ) -> tuple[Sequence[int], Sequence[float]]:
        if isinstance(vector, tuple):
            return vector[0], vector[1]
        return vector.indices, vector.values
