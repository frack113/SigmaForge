from __future__ import annotations

import json
from typing import Any

from sigmaforge.db.client import Database


def _encode(value: Any) -> str:
    return json.dumps(value)


def _decode(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class ConfigStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.fetch_one("SELECT value FROM config WHERE key = ?", [key])
        if row is None:
            return default
        return _decode(row[0])

    def set(self, key: str, value: Any) -> None:
        self.db.execute(
            """
            INSERT INTO config (key, value, updated_at)
            VALUES (?, ?, now())
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = now()
            """,
            [key, _encode(value)],
        )

    def delete(self, key: str) -> bool:
        row = self.db.fetch_one("SELECT 1 FROM config WHERE key = ?", [key])
        if row is None:
            return False
        self.db.execute("DELETE FROM config WHERE key = ?", [key])
        return True

    def get_all(self) -> dict[str, Any]:
        rows = self.db.fetch_all("SELECT key, value FROM config")
        return {key: _decode(value) for key, value in rows}
