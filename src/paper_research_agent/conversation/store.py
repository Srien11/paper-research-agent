"""SQLite-backed authoritative ledger shared by every Web route."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from paper_research_agent.agent.orchestrator.models import (
    AgentRunStart,
    ConversationWorkspace,
    MainAgentResult,
)
from paper_research_agent.conversation.models import (
    ConversationEpisode,
    ConversationResolution,
    ConversationStatus,
    ConversationTurn,
)


class ConversationStore(Protocol):
    def begin_turn(self, conversation_id: str, user_question: str) -> ConversationTurn: ...

    def complete_turn(
        self,
        turn_id: str,
        *,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None = None,
        source_ids: Sequence[str] = (),
    ) -> bool: ...

    def recent(self, conversation_id: str, *, limit: int = 8) -> tuple[ConversationTurn, ...]: ...

    def history(
        self, conversation_id: str, *, limit: int = 500
    ) -> tuple[ConversationTurn, ...]: ...

    def episodes(
        self, conversation_id: str, *, limit: int = 100
    ) -> tuple[ConversationEpisode, ...]: ...

    def begin_agent_run(
        self, *, request_id: str, conversation_id: str, user_question: str
    ) -> AgentRunStart: ...

    def load_workspace(self, conversation_id: str) -> ConversationWorkspace: ...

    def load_agent_run(self, request_id: str) -> MainAgentResult | None: ...

    def clear(self, conversation_id: str) -> int: ...


class SQLiteConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def begin_turn(self, conversation_id: str, user_question: str) -> ConversationTurn:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        created = datetime.now(UTC)
        turn_id = uuid.uuid4().hex
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_turns "
                    "WHERE conversation_id = ?",
                    (normalized_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO conversation_turns (
                       turn_id, conversation_id, sequence, user_question, status, created_at,
                       source_ids_json, selected_history_turn_ids_json,
                       selected_history_relevances_json
                   ) VALUES (?, ?, ?, ?, 'pending', ?, '[]', '[]', '[]')""",
                (turn_id, normalized_id, sequence, normalized_question, created.isoformat()),
            )
            connection.commit()
        return ConversationTurn(
            turn_id=turn_id,
            conversation_id=normalized_id,
            sequence=sequence,
            user_question=normalized_question,
            status="pending",
            created_at=created,
        )

    def begin_agent_run(
        self, *, request_id: str, conversation_id: str, user_question: str
    ) -> AgentRunStart:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        normalized_request = _request_id(request_id)
        created = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_id, turn_id, status, result_json FROM main_agent_runs "
                "WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()
            if existing is not None:
                run_id = str(existing[0])
                turn_id = str(existing[1])
                workspace = self._workspace_row(connection, normalized_id)
                if str(existing[2]) == "completed":
                    result = (
                        MainAgentResult.model_validate_json(str(existing[3]))
                        if existing[3] is not None
                        else None
                    )
                    connection.commit()
                    return AgentRunStart(
                        run_id=run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=turn_id,
                        workspace=workspace,
                        outcome="completed_cached",
                        result=result,
                    )
                connection.commit()
                return AgentRunStart(
                    run_id=run_id,
                    request_id=normalized_request,
                    conversation_id=normalized_id,
                    turn_id=turn_id,
                    workspace=workspace,
                    outcome="running_reused",
                )
            run_id = uuid.uuid4().hex
            turn_id = uuid.uuid4().hex
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM conversation_turns "
                    "WHERE conversation_id = ?",
                    (normalized_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """INSERT INTO conversation_turns (
                       turn_id, conversation_id, sequence, user_question, status, created_at,
                       source_ids_json, selected_history_turn_ids_json,
                       selected_history_relevances_json
                   ) VALUES (?, ?, ?, ?, 'pending', ?, '[]', '[]', '[]')""",
                (turn_id, normalized_id, sequence, normalized_question, created.isoformat()),
            )
            workspace = ConversationWorkspace(
                conversation_id=normalized_id,
                version=0,
                updated_at=created,
            )
            connection.execute(
                """INSERT INTO conversation_workspaces (
                       conversation_id, schema_version, version, state_json, updated_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    normalized_id,
                    workspace.schema_version,
                    workspace.version,
                    workspace.model_dump_json(),
                    created.isoformat(),
                ),
            )
            connection.execute(
                """INSERT INTO main_agent_runs (
                       run_id, request_id, conversation_id, turn_id, base_workspace_version,
                       status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)""",
                (
                    run_id,
                    normalized_request,
                    normalized_id,
                    turn_id,
                    workspace.version,
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
            connection.commit()
        return AgentRunStart(
            run_id=run_id,
            request_id=normalized_request,
            conversation_id=normalized_id,
            turn_id=turn_id,
            workspace=workspace,
            outcome="created",
        )

    def load_workspace(self, conversation_id: str) -> ConversationWorkspace:
        normalized = _conversation_id(conversation_id)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state_json FROM conversation_workspaces WHERE conversation_id = ?",
                (normalized,),
            ).fetchone()
        if row is None:
            raise ValueError("conversation workspace not found")
        return ConversationWorkspace.model_validate_json(str(row[0]))

    def load_agent_run(self, request_id: str) -> MainAgentResult | None:
        normalized = _request_id(request_id)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status, result_json FROM main_agent_runs WHERE request_id = ?",
                (normalized,),
            ).fetchone()
        if row is None or str(row[0]) != "completed" or row[1] is None:
            return None
        return MainAgentResult.model_validate_json(str(row[1]))

    def complete_turn(
        self,
        turn_id: str,
        *,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None = None,
        source_ids: Sequence[str] = (),
    ) -> bool:
        completed = datetime.now(UTC).isoformat()
        with self._lock, closing(self._connect()) as connection, connection:
            current = connection.execute(
                "SELECT conversation_id, sequence FROM conversation_turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            cursor = connection.execute(
                """UPDATE conversation_turns
                   SET standalone_question = ?, route = ?, status = ?, assistant_summary = ?,
                       source_ids_json = ?, episode_id = ?, selected_history_turn_ids_json = ?,
                       selected_history_relevances_json = ?, rewrite_confidence = ?, completed_at = ?
                   WHERE turn_id = ? AND status = 'pending'""",
                (
                    resolution.standalone_question,
                    route,
                    status,
                    _summary(assistant_summary),
                    json.dumps(tuple(dict.fromkeys(source_ids)), ensure_ascii=False),
                    resolution.episode_id,
                    json.dumps(resolution.selected_turn_ids, ensure_ascii=False),
                    json.dumps(
                        tuple(item.relevance for item in resolution.selected_candidates),
                        ensure_ascii=False,
                    ),
                    resolution.confidence,
                    completed,
                    turn_id,
                ),
            )
            if cursor.rowcount == 1 and resolution.episode_id is not None and current is not None:
                connection.execute(
                    """INSERT INTO conversation_episodes (
                           conversation_id, episode_id, summary, last_sequence, updated_at
                       ) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(conversation_id, episode_id) DO UPDATE SET
                           summary = excluded.summary,
                           last_sequence = excluded.last_sequence,
                           updated_at = excluded.updated_at""",
                    (
                        str(current[0]),
                        resolution.episode_id,
                        resolution.standalone_question[:2_000],
                        int(current[1]),
                        completed,
                    ),
                )
        return cursor.rowcount == 1

    def recent(self, conversation_id: str, *, limit: int = 8) -> tuple[ConversationTurn, ...]:
        if limit <= 0 or limit > 100:
            raise ValueError("recent conversation limit must be between 1 and 100")
        rows = self._rows(conversation_id, limit=limit)
        return tuple(reversed(tuple(self._row(row) for row in rows)))

    def history(
        self, conversation_id: str, *, limit: int = 500
    ) -> tuple[ConversationTurn, ...]:
        if limit <= 0 or limit > 2_000:
            raise ValueError("conversation history limit must be between 1 and 2000")
        rows = self._rows(conversation_id, limit=limit)
        return tuple(reversed(tuple(self._row(row) for row in rows)))

    def clear(self, conversation_id: str) -> int:
        normalized = _conversation_id(conversation_id)
        with self._lock, closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (normalized,)
            )
            connection.execute(
                "DELETE FROM conversation_episodes WHERE conversation_id = ?", (normalized,)
            )
            connection.execute(
                "DELETE FROM conversation_workspaces WHERE conversation_id = ?", (normalized,)
            )
            connection.execute(
                "DELETE FROM main_agent_runs WHERE conversation_id = ?", (normalized,)
            )
        return int(cursor.rowcount)

    def episodes(
        self, conversation_id: str, *, limit: int = 100
    ) -> tuple[ConversationEpisode, ...]:
        normalized = _conversation_id(conversation_id)
        if limit <= 0 or limit > 500:
            raise ValueError("episode limit must be between 1 and 500")
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT conversation_id, episode_id, summary, last_sequence, updated_at
                   FROM conversation_episodes WHERE conversation_id = ?
                   ORDER BY last_sequence DESC LIMIT ?""",
                (normalized, limit),
            ).fetchall()
        return tuple(
            ConversationEpisode(
                conversation_id=str(row[0]),
                episode_id=str(row[1]),
                summary=str(row[2]),
                last_sequence=int(row[3]),
                updated_at=datetime.fromisoformat(str(row[4])),
            )
            for row in reversed(rows)
        )

    def _rows(self, conversation_id: str, *, limit: int) -> list[tuple[object, ...]]:
        normalized = _conversation_id(conversation_id)
        with self._lock, closing(self._connect()) as connection:
            return connection.execute(
                """SELECT turn_id, conversation_id, sequence, user_question,
                          standalone_question, route, status, assistant_summary,
                          source_ids_json, episode_id, selected_history_turn_ids_json,
                          selected_history_relevances_json, rewrite_confidence,
                          created_at, completed_at
                   FROM conversation_turns
                   WHERE conversation_id = ? AND status != 'pending'
                   ORDER BY sequence DESC LIMIT ?""",
                (normalized, limit),
            ).fetchall()

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    user_question TEXT NOT NULL,
                    standalone_question TEXT,
                    route TEXT,
                    status TEXT NOT NULL,
                    assistant_summary TEXT,
                    source_ids_json TEXT NOT NULL,
                    episode_id TEXT,
                    selected_history_turn_ids_json TEXT NOT NULL,
                    selected_history_relevances_json TEXT NOT NULL DEFAULT '[]',
                    rewrite_confidence REAL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(conversation_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS conversation_sequence_idx
                    ON conversation_turns(conversation_id, sequence DESC);
                CREATE INDEX IF NOT EXISTS conversation_episode_idx
                    ON conversation_turns(conversation_id, episode_id);
                CREATE TABLE IF NOT EXISTS conversation_episodes (
                    conversation_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(conversation_id, episode_id)
                );
                CREATE TABLE IF NOT EXISTS conversation_workspaces (
                    conversation_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS main_agent_runs (
                    run_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    base_workspace_version INTEGER NOT NULL,
                    committed_workspace_version INTEGER,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(turn_id) REFERENCES conversation_turns(turn_id)
                );
                CREATE INDEX IF NOT EXISTS main_agent_runs_conversation_idx
                    ON main_agent_runs(conversation_id, created_at DESC);
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(conversation_turns)")
            }
            if "selected_history_relevances_json" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_turns ADD COLUMN "
                    "selected_history_relevances_json TEXT NOT NULL DEFAULT '[]'"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA secure_delete = ON")
        return connection

    @staticmethod
    def _workspace_row(
        connection: sqlite3.Connection, conversation_id: str
    ) -> ConversationWorkspace:
        row = connection.execute(
            "SELECT state_json FROM conversation_workspaces WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ValueError("conversation workspace not found")
        return ConversationWorkspace.model_validate_json(str(row[0]))

    @staticmethod
    def _row(row: tuple[object, ...]) -> ConversationTurn:
        return ConversationTurn(
            turn_id=str(row[0]),
            conversation_id=str(row[1]),
            sequence=int(str(row[2])),
            user_question=str(row[3]),
            standalone_question=str(row[4]) if row[4] is not None else None,
            route=str(row[5]) if row[5] is not None else None,
            status=cast(ConversationStatus, str(row[6])),
            assistant_summary=str(row[7]) if row[7] is not None else None,
            source_ids=tuple(json.loads(str(row[8]))),
            episode_id=str(row[9]) if row[9] is not None else None,
            selected_history_turn_ids=tuple(json.loads(str(row[10]))),
            selected_history_relevances=tuple(json.loads(str(row[11]))),
            rewrite_confidence=float(str(row[12])) if row[12] is not None else None,
            created_at=datetime.fromisoformat(str(row[13])),
            completed_at=datetime.fromisoformat(str(row[14])) if row[14] is not None else None,
        )


@dataclass
class _AgentRunRecord:
    run_id: str
    request_id: str
    conversation_id: str
    turn_id: str
    status: str
    base_workspace_version: int
    result: MainAgentResult | None = None


class InMemoryConversationStore:
    """Process-local implementation used by tests and embedded callers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, ConversationTurn] = {}
        self._workspaces: dict[str, ConversationWorkspace] = {}
        self._runs: dict[str, _AgentRunRecord] = {}

    def begin_turn(self, conversation_id: str, user_question: str) -> ConversationTurn:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        with self._lock:
            sequence = 1 + max(
                (
                    item.sequence
                    for item in self._turns.values()
                    if item.conversation_id == normalized_id
                ),
                default=0,
            )
            turn = ConversationTurn(
                turn_id=uuid.uuid4().hex,
                conversation_id=normalized_id,
                sequence=sequence,
                user_question=normalized_question,
                status="pending",
                created_at=datetime.now(UTC),
            )
            self._turns[turn.turn_id] = turn
        return turn

    def begin_agent_run(
        self, *, request_id: str, conversation_id: str, user_question: str
    ) -> AgentRunStart:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        normalized_request = _request_id(request_id)
        created = datetime.now(UTC)
        with self._lock:
            existing = self._runs.get(normalized_request)
            if existing is not None:
                workspace = self._workspaces.get(normalized_id)
                if workspace is None:
                    raise ValueError("conversation workspace not found")
                if existing.status == "completed":
                    return AgentRunStart(
                        run_id=existing.run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=existing.turn_id,
                        workspace=workspace,
                        outcome="completed_cached",
                        result=existing.result,
                    )
                return AgentRunStart(
                    run_id=existing.run_id,
                    request_id=normalized_request,
                    conversation_id=normalized_id,
                    turn_id=existing.turn_id,
                    workspace=workspace,
                    outcome="running_reused",
                )
            run_id = uuid.uuid4().hex
            turn_id = uuid.uuid4().hex
            sequence = 1 + max(
                (
                    item.sequence
                    for item in self._turns.values()
                    if item.conversation_id == normalized_id
                ),
                default=0,
            )
            turn = ConversationTurn(
                turn_id=turn_id,
                conversation_id=normalized_id,
                sequence=sequence,
                user_question=normalized_question,
                status="pending",
                created_at=created,
            )
            self._turns[turn_id] = turn
            workspace = ConversationWorkspace(
                conversation_id=normalized_id,
                version=0,
                updated_at=created,
            )
            self._workspaces[normalized_id] = workspace
            self._runs[normalized_request] = _AgentRunRecord(
                run_id=run_id,
                request_id=normalized_request,
                conversation_id=normalized_id,
                turn_id=turn_id,
                status="running",
                base_workspace_version=0,
            )
        return AgentRunStart(
            run_id=run_id,
            request_id=normalized_request,
            conversation_id=normalized_id,
            turn_id=turn_id,
            workspace=workspace,
            outcome="created",
        )

    def load_workspace(self, conversation_id: str) -> ConversationWorkspace:
        normalized = _conversation_id(conversation_id)
        with self._lock:
            workspace = self._workspaces.get(normalized)
        if workspace is None:
            raise ValueError("conversation workspace not found")
        return workspace

    def load_agent_run(self, request_id: str) -> MainAgentResult | None:
        normalized = _request_id(request_id)
        with self._lock:
            record = self._runs.get(normalized)
        if record is None or record.status != "completed":
            return None
        return record.result

    def complete_turn(
        self,
        turn_id: str,
        *,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None = None,
        source_ids: Sequence[str] = (),
    ) -> bool:
        with self._lock:
            current = self._turns.get(turn_id)
            if current is None or current.status != "pending":
                return False
            self._turns[turn_id] = current.model_copy(
                update={
                    "standalone_question": resolution.standalone_question,
                    "route": route,
                    "status": status,
                    "assistant_summary": _summary(assistant_summary),
                    "source_ids": tuple(dict.fromkeys(source_ids)),
                    "episode_id": resolution.episode_id,
                    "selected_history_turn_ids": resolution.selected_turn_ids,
                    "selected_history_relevances": tuple(
                        item.relevance for item in resolution.selected_candidates
                    ),
                    "rewrite_confidence": resolution.confidence,
                    "completed_at": datetime.now(UTC),
                }
            )
        return True

    def recent(self, conversation_id: str, *, limit: int = 8) -> tuple[ConversationTurn, ...]:
        return self.history(conversation_id, limit=limit)

    def history(
        self, conversation_id: str, *, limit: int = 500
    ) -> tuple[ConversationTurn, ...]:
        normalized = _conversation_id(conversation_id)
        with self._lock:
            values = sorted(
                (
                    item
                    for item in self._turns.values()
                    if item.conversation_id == normalized and item.status != "pending"
                ),
                key=lambda item: item.sequence,
            )
        return tuple(values[-limit:])

    def clear(self, conversation_id: str) -> int:
        normalized = _conversation_id(conversation_id)
        with self._lock:
            targets = [
                turn_id
                for turn_id, item in self._turns.items()
                if item.conversation_id == normalized
            ]
            for turn_id in targets:
                self._turns.pop(turn_id, None)
            self._workspaces.pop(normalized, None)
            for request_id in [
                run_request
                for run_request, record in self._runs.items()
                if record.conversation_id == normalized
            ]:
                self._runs.pop(request_id, None)
        return len(targets)

    def episodes(
        self, conversation_id: str, *, limit: int = 100
    ) -> tuple[ConversationEpisode, ...]:
        latest: dict[str, ConversationEpisode] = {}
        for turn in self.history(conversation_id, limit=2_000):
            if turn.episode_id is None or turn.standalone_question is None:
                continue
            latest[turn.episode_id] = ConversationEpisode(
                conversation_id=turn.conversation_id,
                episode_id=turn.episode_id,
                summary=turn.standalone_question,
                last_sequence=turn.sequence,
                updated_at=turn.completed_at or turn.created_at,
            )
        return tuple(sorted(latest.values(), key=lambda item: item.last_sequence)[-limit:])


def _conversation_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 256 or any(char.isspace() for char in normalized):
        raise ValueError("conversation_id is invalid")
    return normalized


def _request_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 256 or any(char.isspace() for char in normalized):
        raise ValueError("request_id is invalid")
    return normalized


def _question(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 10_000:
        raise ValueError("conversation question is invalid")
    return normalized


def _summary(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized[:3_000] or None
