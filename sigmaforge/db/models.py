from __future__ import annotations

from dataclasses import dataclass

from sigmaforge.db.client import Database


@dataclass(frozen=True, slots=True)
class ModelRecord:
    kind: str
    name: str
    path: str
    active: bool


class ModelStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, kind: str, name: str, path: str, active: bool = False) -> None:
        self.db.execute(
            """
            INSERT INTO models (kind, name, path, active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (kind, name) DO UPDATE SET
                path = excluded.path,
                active = excluded.active
            """,
            [kind, name, path, active],
        )

    def get(self, kind: str, name: str) -> ModelRecord | None:
        row = self.db.fetch_one(
            "SELECT kind, name, path, active FROM models WHERE kind = ? AND name = ?",
            [kind, name],
        )
        if row is None:
            return None
        return ModelRecord(kind=row[0], name=row[1], path=row[2], active=bool(row[3]))

    def get_active(self, kind: str) -> str | None:
        row = self.db.fetch_one(
            "SELECT name FROM models WHERE kind = ? AND active = TRUE ORDER BY name LIMIT 1",
            [kind],
        )
        return row[0] if row else None

    def set_active(self, kind: str, name: str) -> bool:
        if self.get(kind, name) is None:
            return False
        self.db.execute(
            "UPDATE models SET active = FALSE WHERE kind = ? AND name != ?",
            [kind, name],
        )
        self.db.execute(
            "UPDATE models SET active = TRUE WHERE kind = ? AND name = ?",
            [kind, name],
        )
        return True

    def list(self, kind: str | None = None) -> list[ModelRecord]:
        if kind is None:
            rows = self.db.fetch_all(
                "SELECT kind, name, path, active FROM models ORDER BY kind, name"
            )
        else:
            rows = self.db.fetch_all(
                "SELECT kind, name, path, active FROM models WHERE kind = ? ORDER BY name",
                [kind],
            )
        return [ModelRecord(row[0], row[1], row[2], bool(row[3])) for row in rows]

    def delete(self, kind: str, name: str) -> bool:
        if self.get(kind, name) is None:
            return False
        self.db.execute("DELETE FROM models WHERE kind = ? AND name = ?", [kind, name])
        return True
