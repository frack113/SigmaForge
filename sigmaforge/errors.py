from __future__ import annotations


class SigmaForgeError(Exception):
    pass


class ConfigError(SigmaForgeError):
    pass


class DatabaseError(SigmaForgeError):
    pass


class QdrantError(SigmaForgeError):
    pass


class EmbeddingError(SigmaForgeError):
    pass


class LlamaError(SigmaForgeError):
    pass


class SearchError(SigmaForgeError):
    pass
