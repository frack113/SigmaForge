from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sigmaforge.db.client import Database


def _timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None if value is None else str(value)


@dataclass(slots=True)
class Task:
    id: str
    type: str
    status: str
    progress: float = 0.0
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class TaskStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, task: Task) -> None:
        self.db.execute(
            """
            INSERT INTO tasks (id, type, status, progress, message, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                task.id,
                task.type,
                task.status,
                task.progress,
                task.message,
                json.dumps(task.payload),
            ],
        )

    def get(self, task_id: str) -> Task | None:
        row = self.db.fetch_one(
            """
            SELECT id, type, status, progress, message, payload, created_at, updated_at
            FROM tasks
            WHERE id = ?
            """,
            [task_id],
        )
        if row is None:
            return None
        return Task(
            id=row[0],
            type=row[1],
            status=row[2],
            progress=float(row[3]),
            message=row[4],
            payload=json.loads(row[5]),
            created_at=_timestamp(row[6]),
            updated_at=_timestamp(row[7]),
        )

    def update_status(
        self,
        task_id: str,
        status: str,
        progress: float | None = None,
        message: str | None = None,
    ) -> bool:
        current = self.get(task_id)
        if current is None:
            return False
        next_progress = current.progress if progress is None else progress
        next_message = current.message if message is None else message
        self.db.execute(
            """
            UPDATE tasks
            SET status = ?, progress = ?, message = ?, updated_at = now()
            WHERE id = ?
            """,
            [status, next_progress, next_message, task_id],
        )
        return True

    def update_payload(self, task_id: str, payload: dict[str, Any]) -> bool:
        if self.get(task_id) is None:
            return False
        self.db.execute(
            "UPDATE tasks SET payload = ?, updated_at = current_timestamp WHERE id = ?",
            [json.dumps(payload), task_id],
        )
        return True

    def list(self, limit: int = 100) -> list[Task]:
        rows = self.db.fetch_all(
            """
            SELECT id, type, status, progress, message, payload, created_at, updated_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        )
        return [
            Task(
                id=row[0],
                type=row[1],
                status=row[2],
                progress=float(row[3]),
                message=row[4],
                payload=json.loads(row[5]),
                created_at=_timestamp(row[6]),
                updated_at=_timestamp(row[7]),
            )
            for row in rows
        ]
