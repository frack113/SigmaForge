from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from sigmaforge.db.config import ConfigStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_COLLECTIONS: tuple[str, ...] = ("sigma_rules", "sigma_docs", "sigma_spec")


def _default_data_dir() -> Path:
    return DATA_DIR


def _default_duckdb_path() -> Path:
    return DATA_DIR / "duckdb" / "sigmaforge.duckdb"


def _default_qdrant_storage_path() -> Path:
    return DATA_DIR / "qdrant"


def _default_logs_dir() -> Path:
    return DATA_DIR / "logs"


def _default_temp_dir() -> Path:
    return DATA_DIR / "temp"


def _default_local_documents_dir() -> Path:
    return DATA_DIR / "documents" / "local"


def _default_github_dir() -> Path:
    return DATA_DIR / "github"


def _default_spec_dir() -> Path:
    return DATA_DIR / "specification"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str) -> int:
    return int(value.strip())


def _parse_float(value: str) -> float:
    return float(value.strip())


def _parse_collection(value: str) -> tuple[str, ...]:
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part)


def _as_path(value: Any) -> Path:
    return Path(str(value))


def _as_str(value: Any) -> str:
    return str(value)


def _as_int(value: Any) -> int:
    return int(value)


def _as_float(value: Any) -> float:
    return float(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _parse_bool(str(value))


def _as_collections(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return _parse_collection(value)
    return tuple(str(item) for item in value)


def _set_data_dir(config: Config, data_dir: Path) -> None:
    config.data_dir = data_dir
    config.duckdb_path = data_dir / "duckdb" / "sigmaforge.duckdb"
    config.qdrant_storage_path = data_dir / "qdrant"
    config.logs_dir = data_dir / "logs"
    config.temp_dir = data_dir / "temp"
    config.local_documents_dir = data_dir / "documents" / "local"
    config.github_dir = data_dir / "github"
    config.spec_dir = data_dir / "specification"


@dataclass(slots=True)
class Config:
    data_dir: Path = field(default_factory=_default_data_dir)
    duckdb_path: Path = field(default_factory=_default_duckdb_path)
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_storage_path: Path = field(default_factory=_default_qdrant_storage_path)
    vector_size: int = 384
    collections: tuple[str, ...] = field(default_factory=lambda: DEFAULT_COLLECTIONS)
    llama_base_url: str = "http://127.0.0.1:8080"
    llama_model: str = "sigma"
    llama_api_key: str = "sigma-key"
    llama_timeout: float = 120.0
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 64
    logs_dir: Path = field(default_factory=_default_logs_dir)
    temp_dir: Path = field(default_factory=_default_temp_dir)
    local_documents_dir: Path = field(default_factory=_default_local_documents_dir)
    github_dir: Path = field(default_factory=_default_github_dir)
    spec_dir: Path = field(default_factory=_default_spec_dir)
    chunk_size: int = 1024
    chunk_overlap: int = 100
    chunk_size_rules: int = 512
    chunk_overlap_rules: int = 50
    log_level: str = "INFO"
    hf_offline: bool = True

    def apply_env(self, env: Mapping[str, str] | None = None) -> None:
        source = os.environ if env is None else env
        if value := source.get("SIGMAFORGE_DATA_DIR"):
            _set_data_dir(self, Path(value))
        if value := source.get("SIGMAFORGE_DUCKDB_PATH"):
            self.duckdb_path = Path(value)
        if value := source.get("SIGMAFORGE_QDRANT_HOST"):
            self.qdrant_host = value
        if value := source.get("SIGMAFORGE_QDRANT_PORT"):
            self.qdrant_port = _parse_int(value)
        if value := source.get("SIGMAFORGE_QDRANT_STORAGE_PATH"):
            self.qdrant_storage_path = Path(value)
        if value := source.get("SIGMAFORGE_VECTOR_SIZE"):
            self.vector_size = _parse_int(value)
        if value := source.get("SIGMAFORGE_COLLECTIONS"):
            self.collections = _parse_collection(value)
        if value := source.get("SIGMAFORGE_LLAMA_BASE_URL"):
            self.llama_base_url = value
        if value := source.get("SIGMAFORGE_LLAMA_MODEL"):
            self.llama_model = value
        if value := source.get("SIGMAFORGE_LLAMA_API_KEY"):
            self.llama_api_key = value
        if value := source.get("SIGMAFORGE_LLAMA_TIMEOUT"):
            self.llama_timeout = _parse_float(value)
        if value := source.get("SIGMAFORGE_EMBEDDING_MODEL"):
            self.embedding_model = value
        if value := source.get("SIGMAFORGE_EMBEDDING_DEVICE"):
            self.embedding_device = value
        if value := source.get("SIGMAFORGE_EMBEDDING_BATCH_SIZE"):
            self.embedding_batch_size = _parse_int(value)
        if value := source.get("SIGMAFORGE_LOG_LEVEL"):
            self.log_level = value
        if value := source.get("HF_HUB_OFFLINE"):
            self.hf_offline = _parse_bool(value)

    def apply_db(self, store: ConfigStore) -> None:
        values = store.get_all()
        if value := values.get("paths.data"):
            _set_data_dir(self, _as_path(value))
        if value := values.get("paths.duckdb"):
            self.duckdb_path = _as_path(value)
        if value := values.get("services.qdrant.host"):
            self.qdrant_host = _as_str(value)
        if value := values.get("services.qdrant.port"):
            self.qdrant_port = _as_int(value)
        if value := values.get("services.qdrant.storage_path"):
            self.qdrant_storage_path = _as_path(value)
        if value := values.get("services.llama.base_url"):
            self.llama_base_url = _as_str(value)
        if value := values.get("services.llama.model"):
            self.llama_model = _as_str(value)
        if value := values.get("services.llama.api_key"):
            self.llama_api_key = _as_str(value)
        if value := values.get("models.embedding.active"):
            self.embedding_model = _as_str(value)
        if value := values.get("models.embedding.device"):
            self.embedding_device = _as_str(value)
        if value := values.get("models.embedding.batch_size"):
            self.embedding_batch_size = _as_int(value)
        if value := values.get("search.vector_size"):
            self.vector_size = _as_int(value)
        if value := values.get("search.collections"):
            self.collections = _as_collections(value)
        if value := values.get("logging.level"):
            self.log_level = _as_str(value)
        if value := values.get("ingestion.chunk_size"):
            self.chunk_size = _as_int(value)
        if value := values.get("ingestion.chunk_overlap"):
            self.chunk_overlap = _as_int(value)
        if value := values.get("ingestion.chunk_size_rules"):
            self.chunk_size_rules = _as_int(value)
        if value := values.get("ingestion.chunk_overlap_rules"):
            self.chunk_overlap_rules = _as_int(value)
        if value := values.get("ingestion.local_documents_dir"):
            self.local_documents_dir = _as_path(value)
        if value := values.get("ingestion.github_dir"):
            self.github_dir = _as_path(value)
        if value := values.get("ingestion.spec_dir"):
            self.spec_dir = _as_path(value)
        if value := values.get("hf_offline"):
            self.hf_offline = _as_bool(value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(
    *,
    env: Mapping[str, str] | None = None,
    config_store: ConfigStore | None = None,
) -> Config:
    config = Config()
    config.apply_env(env)
    if config_store is not None:
        config.apply_db(config_store)
    return config
