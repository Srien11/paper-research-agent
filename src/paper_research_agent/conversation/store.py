"""SQLite-backed authoritative ledger shared by every Web route."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from paper_research_agent.agent.orchestrator.control import (
    AgentRunControl,
    PlanEdit,
    RunControlCommand,
    RunControlConflict,
    RunControlStatus,
    apply_plan_edit,
    transition_run_control,
)
from paper_research_agent.agent.orchestrator.models import (
    AgentApprovalClaim,
    AgentRunStart,
    CommitOutcome,
    ConversationWorkspace,
    MainAgentRequest,
    MainAgentResult,
)
from paper_research_agent.conversation.models import (
    ConversationEpisode,
    ConversationResolution,
    ConversationStatus,
    ConversationTurn,
)


@dataclass(frozen=True, slots=True)
class AgentCheckpointThreads:
    """Exact checkpoint IDs owned by one conversation."""

    main: tuple[str, ...] = ()
    research: tuple[str, ...] = ()


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
        self,
        *,
        request_id: str,
        conversation_id: str,
        user_question: str,
        request: MainAgentRequest | None = None,
    ) -> AgentRunStart: ...

    def load_workspace(self, conversation_id: str) -> ConversationWorkspace: ...

    def load_agent_run(self, request_id: str) -> MainAgentResult | None: ...

    def load_agent_request(self, request_id: str) -> MainAgentRequest | None: ...

    def load_agent_control(
        self, *, request_id: str | None = None, run_id: str | None = None
    ) -> AgentRunControl | None: ...

    def command_agent_run(
        self, *, request_id: str, command: RunControlCommand
    ) -> AgentRunControl: ...

    def edit_agent_plan(
        self, *, request_id: str, edit: PlanEdit
    ) -> ConversationWorkspace: ...

    def commit_agent_run(
        self,
        *,
        run_id: str,
        turn_id: str,
        expected_workspace_version: int,
        workspace: ConversationWorkspace,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None,
        source_ids: Sequence[str],
        result: MainAgentResult,
    ) -> CommitOutcome: ...

    def fail_agent_run(
        self, *, run_id: str, turn_id: str, reason_code: str
    ) -> CommitOutcome: ...

    def claim_agent_approval(
        self, *, request_id: str, approval_request_id: str
    ) -> AgentApprovalClaim | None: ...

    def agent_checkpoint_threads(
        self, conversation_id: str
    ) -> AgentCheckpointThreads: ...

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
        self,
        *,
        request_id: str,
        conversation_id: str,
        user_question: str,
        request: MainAgentRequest | None = None,
    ) -> AgentRunStart:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        normalized_request = _request_id(request_id)
        created = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT run_id, turn_id, status, result_json, conversation_id "
                "FROM main_agent_runs "
                "WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()
            if existing is not None:
                if str(existing[4]) != normalized_id:
                    connection.rollback()
                    raise ValueError("request_id belongs to another conversation")
                run_id = str(existing[0])
                turn_id = str(existing[1])
                workspace = self._workspace_row(connection, normalized_id)
                existing_status = str(existing[2])
                if existing_status == "completed":
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
                if existing_status == "failed":
                    connection.commit()
                    return AgentRunStart(
                        run_id=run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=turn_id,
                        workspace=workspace,
                        outcome="failed_cached",
                    )
                if existing_status == "waiting_approval":
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
                        outcome="waiting_approval_cached",
                        result=result,
                    )
                if existing_status in {"paused", "cancelled", "resuming"}:
                    result = (
                        MainAgentResult.model_validate_json(str(existing[3]))
                        if existing[3] is not None
                        else None
                    )
                    outcome = {
                        "paused": "paused_cached",
                        "cancelled": "cancelled_cached",
                        "resuming": "resuming",
                    }[existing_status]
                    connection.commit()
                    return AgentRunStart(
                        run_id=run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=turn_id,
                        workspace=workspace,
                        outcome=outcome,  # type: ignore[arg-type]
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
            existing_workspace = connection.execute(
                "SELECT state_json FROM conversation_workspaces WHERE conversation_id = ?",
                (normalized_id,),
            ).fetchone()
            if existing_workspace is not None:
                workspace = ConversationWorkspace.model_validate_json(
                    str(existing_workspace[0])
                )
            else:
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
                       status, request_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
                (
                    run_id,
                    normalized_request,
                    normalized_id,
                    turn_id,
                    workspace.version,
                    request.model_dump_json() if request is not None else None,
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
            connection.execute(
                """INSERT INTO main_agent_controls (
                       request_id, run_id, conversation_id, status, revision, updated_at
                   ) VALUES (?, ?, ?, 'running', 0, ?)""",
                (normalized_request, run_id, normalized_id, created.isoformat()),
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
        if row is None or row[1] is None:
            return None
        return MainAgentResult.model_validate_json(str(row[1]))

    def load_agent_request(self, request_id: str) -> MainAgentRequest | None:
        normalized = _request_id(request_id)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT r.request_json, r.conversation_id, t.user_question
                   FROM main_agent_runs r
                   JOIN conversation_turns t ON t.turn_id = r.turn_id
                   WHERE r.request_id = ?""",
                (normalized,),
            ).fetchone()
        if row is None:
            return None
        if row[0] is not None:
            return MainAgentRequest.model_validate_json(str(row[0]))
        return MainAgentRequest(
            request_id=normalized,
            conversation_id=str(row[1]),
            message=str(row[2]),
            rag_mode="preferred",
        )

    def load_agent_control(
        self, *, request_id: str | None = None, run_id: str | None = None
    ) -> AgentRunControl | None:
        column, value = _control_lookup(request_id=request_id, run_id=run_id)
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT request_id, run_id, conversation_id, status, revision, updated_at "
                f"FROM main_agent_controls WHERE {column} = ?",
                (value,),
            ).fetchone()
        return _control_row(row) if row is not None else None

    def command_agent_run(
        self, *, request_id: str, command: RunControlCommand
    ) -> AgentRunControl:
        normalized = _request_id(request_id)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_id, run_id, conversation_id, status, revision, updated_at "
                "FROM main_agent_controls WHERE request_id = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RunControlConflict("run control not found")
            updated = transition_run_control(_control_row(row), command)
            cursor = connection.execute(
                """UPDATE main_agent_controls
                   SET status = ?, revision = ?, updated_at = ?
                   WHERE request_id = ? AND revision = ?""",
                (
                    updated.status,
                    updated.revision,
                    updated.updated_at.isoformat(),
                    normalized,
                    command.expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RunControlConflict("run control revision conflict")
            if updated.status == "resuming" or (
                updated.status == "cancel_requested"
                and _control_row(row).status in {"paused", "waiting_approval"}
            ):
                connection.execute(
                    "UPDATE main_agent_runs SET status = 'resuming', updated_at = ? "
                    "WHERE request_id = ? AND status IN ('paused', 'waiting_approval')",
                    (updated.updated_at.isoformat(), normalized),
                )
            connection.commit()
        return updated

    def edit_agent_plan(
        self, *, request_id: str, edit: PlanEdit
    ) -> ConversationWorkspace:
        normalized = _request_id(request_id)
        now = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT conversation_id, status, result_json FROM main_agent_runs "
                "WHERE request_id = ?",
                (normalized,),
            ).fetchone()
            if run is None or str(run[1]) != "paused":
                connection.rollback()
                raise RunControlConflict("plan edits require a paused run")
            workspace = self._workspace_row(connection, str(run[0]))
            edited = apply_plan_edit(workspace, edit, now=now).model_copy(
                update={"version": workspace.version + 1, "updated_at": now}
            )
            connection.execute(
                "UPDATE conversation_workspaces SET version = ?, state_json = ?, updated_at = ? "
                "WHERE conversation_id = ? AND version = ?",
                (
                    edited.version,
                    edited.model_dump_json(),
                    now.isoformat(),
                    edited.conversation_id,
                    workspace.version,
                ),
            )
            if run[2] is not None:
                result = MainAgentResult.model_validate_json(str(run[2])).model_copy(
                    update={"workspace_version": edited.version}
                )
                connection.execute(
                    "UPDATE main_agent_runs SET result_json = ?, updated_at = ? WHERE request_id = ?",
                    (result.model_dump_json(), now.isoformat(), normalized),
                )
            connection.commit()
        return edited

    def commit_agent_run(
        self,
        *,
        run_id: str,
        turn_id: str,
        expected_workspace_version: int,
        workspace: ConversationWorkspace,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None,
        source_ids: Sequence[str],
        result: MainAgentResult,
    ) -> CommitOutcome:
        completed = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT status FROM main_agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="run_not_found",
                    workspace_version=expected_workspace_version,
                )
            if str(run[0]) == "completed":
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="already_completed",
                    workspace_version=expected_workspace_version,
                )
            if str(run[0]) == "failed":
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="already_failed",
                    workspace_version=expected_workspace_version,
                )
            current = self._workspace_row(connection, workspace.conversation_id)
            if current.version != expected_workspace_version:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="version_conflict",
                    workspace_version=current.version,
                )
            turn_row = connection.execute(
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
                    completed.isoformat(),
                    turn_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=current.version,
                )
            next_version = current.version + 1
            updated = workspace.model_copy(
                update={"version": next_version, "updated_at": completed}
            )
            connection.execute(
                """UPDATE conversation_workspaces
                   SET version = ?, state_json = ?, updated_at = ?
                   WHERE conversation_id = ?""",
                (
                    next_version,
                    updated.model_dump_json(),
                    completed.isoformat(),
                    workspace.conversation_id,
                ),
            )
            connection.execute(
                "UPDATE main_agent_controls SET status = ?, updated_at = ? WHERE run_id = ?",
                (_control_status_for_result(result.status), completed.isoformat(), run_id),
            )
            connection.execute(
                """UPDATE main_agent_runs
                   SET status = ?, committed_workspace_version = ?, result_json = ?,
                       updated_at = ?
                   WHERE run_id = ?""",
                (
                    result.status,
                    next_version,
                    result.model_dump_json(),
                    completed.isoformat(),
                    run_id,
                ),
            )
            if resolution.episode_id is not None and turn_row is not None:
                connection.execute(
                    """INSERT INTO conversation_episodes (
                           conversation_id, episode_id, summary, last_sequence, updated_at
                       ) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(conversation_id, episode_id) DO UPDATE SET
                           summary = excluded.summary,
                           last_sequence = excluded.last_sequence,
                           updated_at = excluded.updated_at""",
                    (
                        str(turn_row[0]),
                        resolution.episode_id,
                        resolution.standalone_question[:2_000],
                        int(turn_row[1]),
                        completed.isoformat(),
                    ),
                )
            connection.commit()
        return CommitOutcome(committed=True, reason="committed", workspace_version=next_version)

    def fail_agent_run(
        self, *, run_id: str, turn_id: str, reason_code: str
    ) -> CommitOutcome:
        failure_code = _failure_code(reason_code)
        completed = datetime.now(UTC)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT status, turn_id, conversation_id FROM main_agent_runs "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="run_not_found",
                    workspace_version=0,
                )
            workspace = self._workspace_row(connection, str(run[2]))
            status = str(run[0])
            if status == "completed":
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="already_completed",
                    workspace_version=workspace.version,
                )
            if status == "failed":
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="already_failed",
                    workspace_version=workspace.version,
                )
            if str(run[1]) != turn_id:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=workspace.version,
                )
            cursor = connection.execute(
                """UPDATE conversation_turns
                   SET route = 'main_agent', status = 'failed',
                       assistant_summary = '主 Agent 运行失败。', completed_at = ?
                   WHERE turn_id = ? AND status = 'pending'""",
                (completed.isoformat(), turn_id),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=workspace.version,
                )
            connection.execute(
                """UPDATE main_agent_runs
                   SET status = 'failed', failure_code = ?, updated_at = ?
                   WHERE run_id = ?""",
                (failure_code, completed.isoformat(), run_id),
            )
            connection.execute(
                "UPDATE main_agent_controls SET status = 'failed', updated_at = ? WHERE run_id = ?",
                (completed.isoformat(), run_id),
            )
            connection.commit()
        return CommitOutcome(
            committed=True,
            reason="failed",
            workspace_version=workspace.version,
        )

    def claim_agent_approval(
        self, *, request_id: str, approval_request_id: str
    ) -> AgentApprovalClaim | None:
        normalized_request = _request_id(request_id)
        normalized_approval = _approval_id(approval_request_id)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT turn_id, conversation_id, result_json FROM main_agent_runs "
                "WHERE request_id = ? AND status = 'waiting_approval'",
                (normalized_request,),
            ).fetchone()
            if row is None or row[2] is None:
                connection.commit()
                return None
            result = MainAgentResult.model_validate_json(str(row[2]))
            if _result_approval_id(result) != normalized_approval:
                connection.commit()
                return None
            cursor = connection.execute(
                "UPDATE main_agent_runs SET status = 'resuming', updated_at = ? "
                "WHERE request_id = ? AND status = 'waiting_approval'",
                (datetime.now(UTC).isoformat(), normalized_request),
            )
            if cursor.rowcount != 1:
                connection.commit()
                return None
            workspace = self._workspace_row(connection, str(row[1]))
            connection.commit()
        return AgentApprovalClaim(
            result=result,
            turn_id=str(row[0]),
            workspace=workspace,
        )

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
                "DELETE FROM main_agent_controls WHERE conversation_id = ?", (normalized,)
            )
            connection.execute(
                "DELETE FROM main_agent_runs WHERE conversation_id = ?", (normalized,)
            )
        return int(cursor.rowcount)

    def agent_checkpoint_threads(
        self, conversation_id: str
    ) -> AgentCheckpointThreads:
        normalized = _conversation_id(conversation_id)
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT run_id, result_json FROM main_agent_runs "
                "WHERE conversation_id = ? ORDER BY created_at, run_id",
                (normalized,),
            ).fetchall()
        main = tuple(f"main::{normalized}::{row[0]!s}" for row in rows)
        research: list[str] = []
        for run_id, result_json in rows:
            if result_json is None:
                continue
            result = MainAgentResult.model_validate_json(str(result_json))
            research.extend(
                f"{normalized}::{run_id!s}::{child.task_id}"
                for child in result.child_results
                if child.capability == "dynamic_tools"
            )
        return AgentCheckpointThreads(main=main, research=tuple(research))

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
                    failure_code TEXT,
                    result_json TEXT,
                    request_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(turn_id) REFERENCES conversation_turns(turn_id)
                );
                CREATE INDEX IF NOT EXISTS main_agent_runs_conversation_idx
                    ON main_agent_runs(conversation_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS main_agent_controls (
                    request_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES main_agent_runs(run_id)
                );
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
            run_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(main_agent_runs)")
            }
            if "failure_code" not in run_columns:
                connection.execute(
                    "ALTER TABLE main_agent_runs ADD COLUMN failure_code TEXT"
                )
            if "request_json" not in run_columns:
                connection.execute(
                    "ALTER TABLE main_agent_runs ADD COLUMN request_json TEXT"
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
    committed_workspace_version: int | None = None
    failure_code: str | None = None
    request: MainAgentRequest | None = None


class InMemoryConversationStore:
    """Process-local implementation used by tests and embedded callers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: dict[str, ConversationTurn] = {}
        self._workspaces: dict[str, ConversationWorkspace] = {}
        self._runs: dict[str, _AgentRunRecord] = {}
        self._controls: dict[str, AgentRunControl] = {}

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
        self,
        *,
        request_id: str,
        conversation_id: str,
        user_question: str,
        request: MainAgentRequest | None = None,
    ) -> AgentRunStart:
        normalized_id = _conversation_id(conversation_id)
        normalized_question = _question(user_question)
        normalized_request = _request_id(request_id)
        created = datetime.now(UTC)
        with self._lock:
            existing = self._runs.get(normalized_request)
            if existing is not None:
                if existing.conversation_id != normalized_id:
                    raise ValueError("request_id belongs to another conversation")
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
                if existing.status == "failed":
                    return AgentRunStart(
                        run_id=existing.run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=existing.turn_id,
                        workspace=workspace,
                        outcome="failed_cached",
                    )
                if existing.status == "waiting_approval":
                    return AgentRunStart(
                        run_id=existing.run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=existing.turn_id,
                        workspace=workspace,
                        outcome="waiting_approval_cached",
                        result=existing.result,
                    )
                if existing.status in {"paused", "cancelled", "resuming"}:
                    outcome = {
                        "paused": "paused_cached",
                        "cancelled": "cancelled_cached",
                        "resuming": "resuming",
                    }[existing.status]
                    return AgentRunStart(
                        run_id=existing.run_id,
                        request_id=normalized_request,
                        conversation_id=normalized_id,
                        turn_id=existing.turn_id,
                        workspace=workspace,
                        outcome=outcome,  # type: ignore[arg-type]
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
            existing_workspace = self._workspaces.get(normalized_id)
            if existing_workspace is None:
                existing_workspace = ConversationWorkspace(
                    conversation_id=normalized_id,
                    version=0,
                    updated_at=created,
                )
                self._workspaces[normalized_id] = existing_workspace
            workspace = existing_workspace
            self._runs[normalized_request] = _AgentRunRecord(
                run_id=run_id,
                request_id=normalized_request,
                conversation_id=normalized_id,
                turn_id=turn_id,
                status="running",
                base_workspace_version=workspace.version,
                request=request,
            )
            self._controls[normalized_request] = AgentRunControl(
                request_id=normalized_request,
                run_id=run_id,
                conversation_id=normalized_id,
                updated_at=created,
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
        if record is None:
            return None
        return record.result

    def load_agent_request(self, request_id: str) -> MainAgentRequest | None:
        normalized = _request_id(request_id)
        with self._lock:
            record = self._runs.get(normalized)
            if record is None:
                return None
            if record.request is not None:
                return record.request
            turn = self._turns.get(record.turn_id)
            if turn is None:
                return None
            return MainAgentRequest(
                request_id=record.request_id,
                conversation_id=record.conversation_id,
                message=turn.user_question,
                rag_mode="preferred",
            )

    def load_agent_control(
        self, *, request_id: str | None = None, run_id: str | None = None
    ) -> AgentRunControl | None:
        _, value = _control_lookup(request_id=request_id, run_id=run_id)
        with self._lock:
            if request_id is not None:
                return self._controls.get(value)
            return next(
                (item for item in self._controls.values() if item.run_id == value), None
            )

    def command_agent_run(
        self, *, request_id: str, command: RunControlCommand
    ) -> AgentRunControl:
        normalized = _request_id(request_id)
        with self._lock:
            current = self._controls.get(normalized)
            if current is None:
                raise RunControlConflict("run control not found")
            updated = transition_run_control(current, command)
            self._controls[normalized] = updated
            if updated.status == "resuming":
                record = self._runs[normalized]
                if record.status == "paused":
                    record.status = "resuming"
            elif updated.status == "cancel_requested" and current.status in {
                "paused",
                "waiting_approval",
            }:
                self._runs[normalized].status = "resuming"
            return updated

    def edit_agent_plan(
        self, *, request_id: str, edit: PlanEdit
    ) -> ConversationWorkspace:
        normalized = _request_id(request_id)
        now = datetime.now(UTC)
        with self._lock:
            record = self._runs.get(normalized)
            if record is None or record.status != "paused":
                raise RunControlConflict("plan edits require a paused run")
            workspace = self._workspaces[record.conversation_id]
            edited = apply_plan_edit(workspace, edit, now=now).model_copy(
                update={"version": workspace.version + 1, "updated_at": now}
            )
            self._workspaces[record.conversation_id] = edited
            if record.result is not None:
                record.result = record.result.model_copy(
                    update={"workspace_version": edited.version}
                )
            return edited

    def commit_agent_run(
        self,
        *,
        run_id: str,
        turn_id: str,
        expected_workspace_version: int,
        workspace: ConversationWorkspace,
        route: str,
        status: ConversationStatus,
        resolution: ConversationResolution,
        assistant_summary: str | None,
        source_ids: Sequence[str],
        result: MainAgentResult,
    ) -> CommitOutcome:
        created = datetime.now(UTC)
        with self._lock:
            record = next(
                (item for item in self._runs.values() if item.run_id == run_id), None
            )
            if record is None:
                return CommitOutcome(
                    committed=False,
                    reason="run_not_found",
                    workspace_version=expected_workspace_version,
                )
            if record.status == "completed":
                return CommitOutcome(
                    committed=False,
                    reason="already_completed",
                    workspace_version=expected_workspace_version,
                )
            if record.status == "failed":
                return CommitOutcome(
                    committed=False,
                    reason="already_failed",
                    workspace_version=expected_workspace_version,
                )
            current = self._workspaces.get(workspace.conversation_id)
            if current is None or current.version != expected_workspace_version:
                return CommitOutcome(
                    committed=False,
                    reason="version_conflict",
                    workspace_version=current.version if current is not None else 0,
                )
            turn = self._turns.get(turn_id)
            if turn is None or turn.status != "pending":
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=current.version,
                )
            next_version = current.version + 1
            self._workspaces[workspace.conversation_id] = workspace.model_copy(
                update={"version": next_version, "updated_at": created}
            )
            self._turns[turn_id] = turn.model_copy(
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
                    "completed_at": created,
                }
            )
            record.status = result.status
            record.result = result
            record.committed_workspace_version = next_version
            control = self._controls.get(record.request_id)
            if control is not None:
                self._controls[record.request_id] = control.model_copy(
                    update={
                        "status": _control_status_for_result(result.status),
                        "updated_at": created,
                    }
                )
        return CommitOutcome(committed=True, reason="committed", workspace_version=next_version)

    def fail_agent_run(
        self, *, run_id: str, turn_id: str, reason_code: str
    ) -> CommitOutcome:
        failure_code = _failure_code(reason_code)
        completed = datetime.now(UTC)
        with self._lock:
            record = next(
                (item for item in self._runs.values() if item.run_id == run_id), None
            )
            if record is None:
                return CommitOutcome(
                    committed=False,
                    reason="run_not_found",
                    workspace_version=0,
                )
            workspace = self._workspaces.get(record.conversation_id)
            workspace_version = workspace.version if workspace is not None else 0
            if record.status == "completed":
                return CommitOutcome(
                    committed=False,
                    reason="already_completed",
                    workspace_version=workspace_version,
                )
            if record.status == "failed":
                return CommitOutcome(
                    committed=False,
                    reason="already_failed",
                    workspace_version=workspace_version,
                )
            if record.turn_id != turn_id:
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=workspace_version,
                )
            turn = self._turns.get(turn_id)
            if turn is None or turn.status != "pending":
                return CommitOutcome(
                    committed=False,
                    reason="turn_conflict",
                    workspace_version=workspace_version,
                )
            self._turns[turn_id] = turn.model_copy(
                update={
                    "route": "main_agent",
                    "status": "failed",
                    "assistant_summary": "主 Agent 运行失败。",
                    "completed_at": completed,
                }
            )
            record.status = "failed"
            record.failure_code = failure_code
            control = self._controls.get(record.request_id)
            if control is not None:
                self._controls[record.request_id] = control.model_copy(
                    update={"status": "failed", "updated_at": completed}
                )
        return CommitOutcome(
            committed=True,
            reason="failed",
            workspace_version=workspace_version,
        )

    def claim_agent_approval(
        self, *, request_id: str, approval_request_id: str
    ) -> AgentApprovalClaim | None:
        normalized_request = _request_id(request_id)
        normalized_approval = _approval_id(approval_request_id)
        with self._lock:
            record = self._runs.get(normalized_request)
            if (
                record is None
                or record.status != "waiting_approval"
                or record.result is None
                or _result_approval_id(record.result) != normalized_approval
            ):
                return None
            workspace = self._workspaces.get(record.conversation_id)
            if workspace is None:
                return None
            record.status = "resuming"
            return AgentApprovalClaim(
                result=record.result,
                turn_id=record.turn_id,
                workspace=workspace,
            )

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
                self._controls.pop(request_id, None)
        return len(targets)

    def agent_checkpoint_threads(
        self, conversation_id: str
    ) -> AgentCheckpointThreads:
        normalized = _conversation_id(conversation_id)
        with self._lock:
            records = sorted(
                (
                    record
                    for record in self._runs.values()
                    if record.conversation_id == normalized
                ),
                key=lambda record: record.run_id,
            )
            main = tuple(
                f"main::{normalized}::{record.run_id}" for record in records
            )
            research = tuple(
                f"{normalized}::{record.run_id}::{child.task_id}"
                for record in records
                if record.result is not None
                for child in record.result.child_results
                if child.capability == "dynamic_tools"
            )
        return AgentCheckpointThreads(main=main, research=research)

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


def _control_lookup(
    *, request_id: str | None, run_id: str | None
) -> tuple[str, str]:
    if (request_id is None) == (run_id is None):
        raise ValueError("provide exactly one of request_id or run_id")
    if request_id is not None:
        return "request_id", _request_id(request_id)
    assert run_id is not None
    return "run_id", _request_id(run_id)


def _control_row(row: Sequence[object]) -> AgentRunControl:
    return AgentRunControl(
        request_id=str(row[0]),
        run_id=str(row[1]),
        conversation_id=str(row[2]),
        status=cast(RunControlStatus, str(row[3])),
        revision=int(str(row[4])),
        updated_at=datetime.fromisoformat(str(row[5])),
    )


def _control_status_for_result(status: str) -> RunControlStatus:
    if status in {"paused", "cancelled", "completed", "failed", "waiting_approval"}:
        return cast(RunControlStatus, status)
    return "running"


def _question(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 10_000:
        raise ValueError("conversation question is invalid")
    return normalized


def _failure_code(value: str) -> str:
    normalized = value.strip()
    if re.fullmatch(r"[a-z][a-z0-9_]{1,63}", normalized) is None:
        raise ValueError("failure reason code is invalid")
    return normalized


def _approval_id(value: str) -> str:
    normalized = value.strip()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise ValueError("approval request ID is invalid")
    return normalized


def _result_approval_id(result: MainAgentResult) -> str | None:
    pending = result.pending_approval
    if pending is None:
        return None
    value = pending.get("approval_request_id")
    return str(value) if value is not None else None


def _summary(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split()).strip()
    return normalized[:3_000] or None
