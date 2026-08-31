"""Durable, owner-initiated steering messages for a running Agent."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunIntervention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intervention_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request_id: str = Field(min_length=16, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=2_000)
    status: Literal["queued", "awaiting_confirmation", "applied", "cancelled"]
    created_at: datetime
    updated_at: datetime


class InterventionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with closing(self._connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS run_interventions (
                    intervention_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, message TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                )"""
            )

    def queue(self, *, request_id: str, conversation_id: str, message: str) -> RunIntervention:
        now = datetime.now(UTC)
        item = RunIntervention(
            intervention_id=uuid.uuid4().hex, request_id=request_id, conversation_id=conversation_id,
            message=message.strip(), status="queued", created_at=now, updated_at=now,
        )
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO run_interventions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (item.intervention_id, item.request_id, item.conversation_id, item.message,
                 item.status, item.created_at.isoformat(), item.updated_at.isoformat()),
            )
            connection.commit()
        return item

    def list(self, *, request_id: str, conversation_id: str) -> tuple[RunIntervention, ...]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM run_interventions WHERE request_id=? AND conversation_id=? ORDER BY created_at",
                (request_id, conversation_id),
            ).fetchall()
        return tuple(_item(row) for row in rows)

    def transition(self, intervention_id: str, status: Literal["awaiting_confirmation", "applied", "cancelled"]) -> RunIntervention:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM run_interventions WHERE intervention_id=?", (intervention_id,)).fetchone()
            if row is None:
                raise KeyError("运行插入指令不存在")
            now = datetime.now(UTC)
            connection.execute("UPDATE run_interventions SET status=?, updated_at=? WHERE intervention_id=?", (status, now.isoformat(), intervention_id))
            connection.commit()
            return _item(connection.execute("SELECT * FROM run_interventions WHERE intervention_id=?", (intervention_id,)).fetchone())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def _item(row: sqlite3.Row) -> RunIntervention:
    return RunIntervention(
        intervention_id=row["intervention_id"], request_id=row["request_id"], conversation_id=row["conversation_id"],
        message=row["message"], status=row["status"], created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
