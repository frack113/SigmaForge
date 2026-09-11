from __future__ import annotations

from sigmaforge.embed import Bm25SparseEncoder, DenseEncoder


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


def test_sparse_encoder_is_deterministic() -> None:
    encoder = Bm25SparseEncoder()
    first = encoder.encode_text("attack attack defense")
    second = encoder.encode_text("attack attack defense")

    assert first == second
    assert len(first.indices) == 2
    assert all(value > 0 for value in first.values)


def test_sparse_encoder_batches_texts() -> None:
    encoder = Bm25SparseEncoder()
    vectors = encoder(["attack", "defense", "the a"])

    assert len(vectors) == 3
    assert vectors[0].indices
    assert vectors[1].indices
    assert vectors[2].indices == ()


def test_dense_encoder_uses_injected_model() -> None:
    encoder = DenseEncoder("fake-model", model=FakeModel(dim=4))

    vectors = encoder.encode(["one", "two"])

    assert vectors == [[0.1] * 4, [0.1] * 4]
    assert encoder.dim == 4
