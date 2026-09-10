"""Application constants - single source of truth for cross-cutting values."""

SCHEMA_VERSION = 20260704

SIGMA_SPEC_REPO = "https://github.com/SigmaHQ/sigma-specification"
SIGMA_SPEC_REF = "main"

DEFAULT_LLAMA_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_QDRANT_BASE_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_COLLECTION = "sigma_docs"
DEFAULT_QDRANT_VECTOR_SIZE = 384
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ── Legacy TOML configuration reference (removed) ──
# Config is now managed via DuckDB and the Config page.
