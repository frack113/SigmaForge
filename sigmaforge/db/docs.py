from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sigmaforge.db.client import Database


@dataclass(frozen=True, slots=True)
class DocRecord:
    source: str
    path: str
    status: str
    indexed_at: str | None = None


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class DocRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def mark(
        self,
        source: str,
        path: str,
        status: str,
        indexed_at: str | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO docs_registry (source, path, status, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (source, path) DO UPDATE SET
                status = excluded.status,
                indexed_at = excluded.indexed_at
            """,
            [source, path, status, indexed_at],
        )

    def get(self, source: str, path: str) -> DocRecord | None:
        row = self.db.fetch_one(
            "SELECT source, path, status, indexed_at FROM docs_registry WHERE source = ? AND path = ?",
            [source, path],
        )
        if row is None:
            return None
        return DocRecord(
            source=row[0],
            path=row[1],
            status=row[2],
            indexed_at=_format_timestamp(row[3]),
        )

    def list_records(self, status: str | None = None) -> list[DocRecord]:
        if status is None:
            rows = self.db.fetch_all(
                "SELECT source, path, status, indexed_at FROM docs_registry ORDER BY source, path"
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT source, path, status, indexed_at
                FROM docs_registry
                WHERE status = ?
                ORDER BY source, path
                """,
                [status],
            )
        return [
            DocRecord(
                source=row[0],
                path=row[1],
                status=row[2],
                indexed_at=_format_timestamp(row[3]),
            )
            for row in rows
        ]

    def delete(self, source: str, path: str) -> bool:
        if self.get(source, path) is None:
            return False
        self.db.execute(
            "DELETE FROM docs_registry WHERE source = ? AND path = ?",
            [source, path],
        )
        return True

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {"source": item.source, "path": item.path, "status": item.status, "indexed_at": item.indexed_at}
            for item in self.list_records()
        ]
