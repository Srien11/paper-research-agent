"""Concurrency-bounded SQLite storage for short-lived session turns."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import (
    MemorySourceRef,
    ShortTermMemoryTurn,
    normalize_session_id,
)


class ShortTermMemoryStore(Protocol):
    def recent(
        self, session_id: str, *, now: datetime | None = None
    ) -> tuple[ShortTermMemoryTurn, ...]: ...

    def append(self, turn: ShortTermMemoryTurn, *, now: datetime | None = None) -> bool: ...


class SQLiteShortTermMemory:
    """Persist complete validated turns and physically purge expired plaintext."""

    def __init__(self, path: Path, *, config: ShortTermMemoryConfig):
        self.path = path
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def recent(
        self, session_id: str, *, now: datetime | None = None
    ) -> tuple[ShortTermMemoryTurn, ...]:
        normalized_session = normalize_session_id(session_id)
        current = _utc(now)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._purge_expired(connection, current)
            rows = connection.execute(
                """SELECT turn_id, session_id, created_at, expires_at, user_question,
                          standalone_question, status, assistant_claims_json, source_refs_json
                   FROM memory_turns
                   WHERE session_id = ?
                   ORDER BY created_at DESC, turn_id DESC
                   LIMIT ?""",
                (normalized_session, self.config.context_turn_limit),
            ).fetchall()
            connection.commit()
        return tuple(self._row_to_turn(row) for row in reversed(rows))

    def append(self, turn: ShortTermMemoryTurn, *, now: datetime | None = None) -> bool:
        current = _utc(now)
        if turn.expires_at.astimezone(UTC) <= current:
            return False
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._purge_expired(connection, current)
            connection.execute(
                """INSERT INTO memory_turns (
                       turn_id, session_id, created_at, expires_at, user_question,
                       standalone_question, status, assistant_claims_json, source_refs_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(turn_id) DO NOTHING""",
                (
                    turn.turn_id,
                    turn.session_id,
                    turn.created_at.astimezone(UTC).isoformat(),
                    turn.expires_at.astimezone(UTC).isoformat(),
                    turn.user_question,
                    turn.standalone_question,
                    turn.status,
                    json.dumps(turn.assistant_claims, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(
                        [source.model_dump(mode="json") for source in turn.source_refs],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            inserted = connection.execute("SELECT changes()").fetchone()[0] == 1
            connection.execute(
                """DELETE FROM memory_turns
                   WHERE turn_id IN (
                       SELECT turn_id FROM memory_turns
                       WHERE session_id = ?
                       ORDER BY created_at DESC, turn_id DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (turn.session_id, self.config.max_turns_per_session),
            )
            connection.commit()
        return bool(inserted)

    def clear(self, session_id: str) -> int:
        normalized_session = normalize_session_id(session_id)
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM memory_turns WHERE session_id = ?", (normalized_session,)
            )
        return int(cursor.rowcount)

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    standalone_question TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assistant_claims_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_session_created
                    ON memory_turns(session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS memory_expires_at
                    ON memory_turns(expires_at);
                PRAGMA user_version = 1;
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(memory_turns)")}
            if "standalone_question" not in columns:
                connection.execute("ALTER TABLE memory_turns ADD COLUMN standalone_question TEXT")
                connection.execute(
                    "UPDATE memory_turns SET standalone_question = user_question "
                    "WHERE standalone_question IS NULL"
                )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @staticmethod
    def _purge_expired(connection: sqlite3.Connection, now: datetime) -> None:
        connection.execute(
            "DELETE FROM memory_turns WHERE expires_at <= ?", (now.astimezone(UTC).isoformat(),)
        )

    @staticmethod
    def _row_to_turn(row: Sequence[object]) -> ShortTermMemoryTurn:
        source_payload = json.loads(str(row[8]))
        raw_status = str(row[6])
        if raw_status not in ("answered", "insufficient_evidence"):
            raise ValueError("stored memory turn has an invalid status")
        status = cast(Literal["answered", "insufficient_evidence"], raw_status)
        return ShortTermMemoryTurn(
            turn_id=str(row[0]),
            session_id=str(row[1]),
            created_at=datetime.fromisoformat(str(row[2])),
            expires_at=datetime.fromisoformat(str(row[3])),
            user_question=str(row[4]),
            standalone_question=str(row[5]),
            status=status,
            assistant_claims=tuple(json.loads(str(row[7]))),
            source_refs=tuple(MemorySourceRef.model_validate(item) for item in source_payload),
        )


def _utc(value: datetime | None) -> datetime:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None:
        raise ValueError("memory clock must be timezone-aware")
    return resolved.astimezone(UTC)
