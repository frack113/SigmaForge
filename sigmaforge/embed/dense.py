from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class DenseEncoder:
    def __init__(
        self,
        model_name: str,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        batch_size: int = 64,
        query_prefix: str = "",
        passage_prefix: str = "",
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model = model

    def _resolve_model_name(self) -> str:
        if self.model_dir is not None:
            candidate = self.model_dir / self.model_name
            if candidate.exists():
                return str(candidate)
        return self.model_name

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._resolve_model_name(),
                device=self.device,
            )
        return self._model

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed = [f"{prefix}{text}" for text in texts]
        vectors = model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    @property
    def dim(self) -> int:
        model = self._get_model()
        return int(model.get_sentence_embedding_dimension())
