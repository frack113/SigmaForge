from __future__ import annotations

from pathlib import Path

from sigmaforge.config import Config, load_config
from sigmaforge.db import ConfigStore, Database


def test_config_defaults() -> None:
    config = Config()
    assert config.qdrant_host == "127.0.0.1"
    assert config.qdrant_port == 6333
    assert config.vector_size == 384
    assert config.collections == ("sigma_rules", "sigma_docs", "sigma_spec")
    assert config.llama_base_url == "http://127.0.0.1:8080"
    assert config.embedding_batch_size == 64
    assert config.hf_offline is True


def test_config_apply_env(monkeypatch) -> None:
    monkeypatch.setenv("SIGMAFORGE_DATA_DIR", "/tmp/sigmaforge-data")
    monkeypatch.setenv("SIGMAFORGE_QDRANT_PORT", "6334")
    monkeypatch.setenv("SIGMAFORGE_VECTOR_SIZE", "512")
    monkeypatch.setenv("SIGMAFORGE_COLLECTIONS", "sigma_rules,sigma_docs")
    monkeypatch.setenv("SIGMAFORGE_EMBEDDING_BATCH_SIZE", "12")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    config = Config()
    config.apply_env()

    assert config.data_dir == Path("/tmp/sigmaforge-data")
    assert config.duckdb_path == Path("/tmp/sigmaforge-data/duckdb/sigmaforge.duckdb")
    assert config.qdrant_port == 6334
    assert config.vector_size == 512
    assert config.collections == ("sigma_rules", "sigma_docs")
    assert config.embedding_batch_size == 12
    assert config.hf_offline is False


def test_load_config_applies_db_overrides() -> None:
    db = Database(":memory:")
    db.init_schema()
    store = ConfigStore(db)
    store.set("paths.duckdb", "/tmp/override.duckdb")
    store.set("services.qdrant.port", 6335)
    store.set("services.llama.base_url", "http://127.0.0.1:9999")
    store.set("search.collections", ["sigma_rules", "sigma_spec"])
    store.set("models.embedding.active", "e5-large")
    store.set("ingestion.chunk_size", 777)

    config = load_config(env={}, config_store=store)

    assert config.duckdb_path == Path("/tmp/override.duckdb")
    assert config.qdrant_port == 6335
    assert config.llama_base_url == "http://127.0.0.1:9999"
    assert config.collections == ("sigma_rules", "sigma_spec")
    assert config.embedding_model == "e5-large"
    assert config.chunk_size == 777
    db.close()
