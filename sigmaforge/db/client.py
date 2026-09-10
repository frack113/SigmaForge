from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Sequence

import duckdb

from sigmaforge.errors import DatabaseError

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        active BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT now(),
        PRIMARY KEY (kind, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        status TEXT NOT NULL,
        progress DOUBLE NOT NULL DEFAULT 0.0,
        message TEXT NOT NULL DEFAULT '',
        payload TEXT NOT NULL DEFAULT '{}',
        created_at TIMESTAMP DEFAULT now(),
        updated_at TIMESTAMP DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS docs_registry (
        source TEXT NOT NULL,
        path TEXT NOT NULL,
        status TEXT NOT NULL,
        indexed_at TIMESTAMP,
        PRIMARY KEY (source, path)
    )
    """,
)


class Database:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = Path(path) if path != ":memory:" else Path(":memory:")
        self._lock = threading.RLock()
        try:
            self._conn = duckdb.connect(str(path))
        except duckdb.Error as exc:
            raise DatabaseError(str(exc)) from exc

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            try:
                self._conn.execute(sql, list(params))
                self._conn.commit()
            except duckdb.Error as exc:
                raise DatabaseError(str(exc)) from exc

    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            try:
                cursor = self._conn.execute(sql, list(params))
                return cursor.fetchall()
            except duckdb.Error as exc:
                raise DatabaseError(str(exc)) from exc

    def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> tuple[Any, ...] | None:
        rows = self.fetch_all(sql, params)
        return rows[0] if rows else None

    def fetch_dict(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        row = self.fetch_one(sql, params)
        if row is None:
            return None
        cursor = self._conn.description
        if cursor is None:
            return dict(zip(("value",), row))
        return {column[0]: value for column, value in zip(cursor, row)}

    def init_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
