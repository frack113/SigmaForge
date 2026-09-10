from __future__ import annotations

from typing import Iterator

import pytest

from sigmaforge.db import ConfigStore, Database, DocRegistry, ModelStore, Task, TaskStore


@pytest.fixture
def db(tmp_path) -> Iterator[Database]:
    database = Database(tmp_path / "sigmaforge.duckdb")
    database.init_schema()
    yield database
    database.close()


def test_config_store_roundtrip(db: Database) -> None:
    store = ConfigStore(db)
    store.set("services.qdrant.port", 6333)
    store.set("logging.level", "DEBUG")

    assert store.get("services.qdrant.port") == 6333
    assert store.get("logging.level") == "DEBUG"
    assert store.get("missing") is None
    assert store.get("missing", "default") == "default"
    assert store.get_all() == {"services.qdrant.port": 6333, "logging.level": "DEBUG"}
    assert store.delete("services.qdrant.port") is True
    assert store.get("services.qdrant.port") is None


def test_model_store_active_selection(db: Database) -> None:
    store = ModelStore(db)
    store.upsert("embedding", "small", "/models/small", active=False)
    store.upsert("embedding", "large", "/models/large", active=True)

    assert store.get_active("embedding") == "large"
    store.set_active("embedding", "small")
    assert store.get_active("embedding") == "small"
    assert store.list("embedding") == [
        store.get("embedding", "large"),
        store.get("embedding", "small"),
    ]


def test_task_store_status_updates(db: Database) -> None:
    store = TaskStore(db)
    store.create(Task(id="t1", type="ingest", status="running", payload={"source": "local"}))

    task = store.get("t1")
    assert task is not None
    assert task.status == "running"
    assert task.payload == {"source": "local"}

    assert store.update_status("t1", "completed", progress=1.0, message="done") is True
    updated = store.get("t1")
    assert updated is not None
    assert updated.status == "completed"
    assert updated.progress == 1.0
    assert updated.message == "done"
    assert store.update_status("missing", "done") is False
    assert len(store.list()) == 1


def test_doc_registry_marks(db: Database) -> None:
    registry = DocRegistry(db)
    registry.mark("local", "rules/win.yaml", "indexed", indexed_at="2026-01-01T00:00:00")

    record = registry.get("local", "rules/win.yaml")
    assert record is not None
    assert record.status == "indexed"
    assert record.indexed_at == "2026-01-01T00:00:00"
    assert len(registry.list_records()) == 1
    assert len(registry.list_records(status="indexed")) == 1
    assert registry.delete("local", "rules/win.yaml") is True
    assert registry.get("local", "rules/win.yaml") is None
