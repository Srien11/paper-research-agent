"""Approval-gated local notes, reports, and long-term memory."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Collection
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path, PurePath

from paper_research_agent.agent.tooling.approval import ApprovalManager
from paper_research_agent.agent.tooling.contracts import (
    ExportResearchReportInput,
    ManageLongTermMemoryInput,
    SaveResearchNoteInput,
    ToolExecutionResult,
)


class WorkspaceResearchTools:
    def __init__(
        self,
        project_root: Path,
        *,
        approvals: ApprovalManager,
        source_chunk_ids: Collection[str] = (),
    ):
        self._root = project_root.resolve()
        self._approvals = approvals
        self._runtime = self._root / "data" / "runtime"
        self._memory_path = self._runtime / "long-term-memory-v1.sqlite3"
        self._source_chunk_ids = frozenset(source_chunk_ids)

    def save_research_note(self, request: SaveResearchNoteInput) -> ToolExecutionResult:
        denied = self._authorize("save_research_note", request)
        if denied is not None:
            return denied
        note_id = uuid.uuid4().hex
        target = self._safe_target(Path("research-notes") / f"{note_id}.md")
        body = (
            f"# {request.title.strip()}\n\n{request.content.strip()}\n\n"
            f"<!-- source_chunk_ids: {json.dumps(request.source_chunk_ids)} -->\n"
        )
        _atomic_write(target, body, overwrite=False)
        return ToolExecutionResult(
            tool_name="save_research_note",
            items=(
                {"note_id": note_id, "relative_path": target.relative_to(self._root).as_posix()},
            ),
        )

    def export_research_report(self, request: ExportResearchReportInput) -> ToolExecutionResult:
        denied = self._authorize("export_research_report", request)
        if denied is not None:
            return denied
        relative = PurePath(request.relative_path)
        expected_suffix = ".md" if request.format == "markdown" else ".json"
        if relative.suffix.casefold() != expected_suffix:
            raise ValueError("report path suffix does not match its format")
        target = self._safe_target(Path("exports") / Path(*relative.parts))
        if request.format == "json":
            parsed = json.loads(request.content)
            content = json.dumps(parsed, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        else:
            content = request.content
        _atomic_write(target, content, overwrite=request.overwrite)
        return ToolExecutionResult(
            tool_name="export_research_report",
            items=(
                {
                    "relative_path": target.relative_to(self._root).as_posix(),
                    "bytes": len(content.encode()),
                },
            ),
        )

    def manage_long_term_memory(self, request: ManageLongTermMemoryInput) -> ToolExecutionResult:
        if request.action in {"search", "list"}:
            return self._read_memories(request)
        denied = self._authorize("manage_long_term_memory", request)
        if denied is not None:
            return denied
        self._initialize_memory()
        if request.action == "add":
            return self._add_memory(request)
        if request.action == "update":
            return self._update_memory(request)
        return self._delete_memory(request)

    def _add_memory(self, request: ManageLongTermMemoryInput) -> ToolExecutionResult:
        kind = request.kind or ""
        content = request.content or ""
        self._validate_memory_sources(kind, request.source_chunk_ids)
        now = datetime.now(UTC).isoformat()
        content_sha256 = _content_fingerprint(content)
        with closing(sqlite3.connect(self._memory_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                "SELECT memory_id FROM memories WHERE scope_id = ? AND kind = ? "
                "AND content_sha256 = ? AND status = 'active' LIMIT 1",
                (request.scope_id, kind, content_sha256),
            ).fetchone()
            if duplicate is not None:
                return ToolExecutionResult(
                    tool_name="manage_long_term_memory",
                    items=({"memory_id": duplicate[0], "action": "duplicate"},),
                    summary={"duplicate": True},
                )
            memory_id = uuid.uuid4().hex
            self._insert_memory(
                connection,
                memory_id=memory_id,
                kind=kind,
                content=content,
                source_chunk_ids=request.source_chunk_ids,
                scope_id=request.scope_id,
                content_sha256=content_sha256,
                version=1,
                created_at=now,
                updated_at=now,
                expires_at=request.expires_at,
                supersedes_memory_id=None,
            )
        return ToolExecutionResult(
            tool_name="manage_long_term_memory",
            items=({"memory_id": memory_id, "action": "added", "version": 1},),
        )

    def _update_memory(self, request: ManageLongTermMemoryInput) -> ToolExecutionResult:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._memory_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE memory_id = ? "
                "AND scope_id = ? AND status = 'active'",
                (request.memory_id, request.scope_id),
            ).fetchone()
            if row is None:
                return _memory_not_found("update")
            old = _memory_item(row)
            kind = request.kind or str(old["kind"])
            raw_source_ids = old["source_chunk_ids"]
            if not isinstance(raw_source_ids, (list, tuple)) or not all(
                isinstance(item, str) for item in raw_source_ids
            ):
                raise RuntimeError("stored memory source IDs are invalid")
            source_ids = request.source_chunk_ids or tuple(raw_source_ids)
            self._validate_memory_sources(kind, source_ids)
            connection.execute(
                "UPDATE memories SET status = 'superseded', updated_at = ? WHERE memory_id = ?",
                (now, request.memory_id),
            )
            memory_id = uuid.uuid4().hex
            raw_version = old["version"]
            if not isinstance(raw_version, int):
                raise TypeError("stored memory version is invalid")
            version = raw_version + 1
            self._insert_memory(
                connection,
                memory_id=memory_id,
                kind=kind,
                content=request.content or "",
                source_chunk_ids=source_ids,
                scope_id=request.scope_id,
                content_sha256=_content_fingerprint(request.content or ""),
                version=version,
                created_at=str(old["created_at"]),
                updated_at=now,
                expires_at=request.expires_at or old["expires_at"],
                supersedes_memory_id=str(request.memory_id),
            )
        return ToolExecutionResult(
            tool_name="manage_long_term_memory",
            items=(
                {
                    "memory_id": memory_id,
                    "action": "updated",
                    "version": version,
                    "supersedes_memory_id": request.memory_id,
                },
            ),
        )

    def _delete_memory(self, request: ManageLongTermMemoryInput) -> ToolExecutionResult:
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._memory_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE memories SET status = 'deleted', updated_at = ? "
                "WHERE memory_id = ? AND scope_id = ? AND status = 'active'",
                (now, request.memory_id, request.scope_id),
            )
        return ToolExecutionResult(
            tool_name="manage_long_term_memory",
            status="ok" if cursor.rowcount else "not_found",
            items=({"memory_id": request.memory_id, "action": "deleted"},)
            if cursor.rowcount
            else (),
        )

    def _read_memories(self, request: ManageLongTermMemoryInput) -> ToolExecutionResult:
        self._initialize_memory()
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self._memory_path)) as connection:
            rows = connection.execute(
                f"SELECT {_MEMORY_COLUMNS} FROM memories WHERE scope_id = ? "
                "AND status = 'active' AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY updated_at DESC LIMIT 200",
                (request.scope_id, now),
            ).fetchall()
        items = [_memory_item(row) for row in rows]
        if request.action == "search":
            query = request.query or ""
            ranked = sorted(
                ((_memory_relevance(query, str(item["content"])), item) for item in items),
                key=lambda pair: -pair[0],
            )
            items = [item for score, item in ranked if score > 0]
        limited = tuple(items[: request.limit])
        return ToolExecutionResult(
            tool_name="manage_long_term_memory",
            status="ok" if limited else "not_found",
            items=limited,
            summary={"count": len(limited), "action": request.action},
        )

    def _insert_memory(
        self,
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        kind: str,
        content: str,
        source_chunk_ids: tuple[str, ...],
        scope_id: str,
        content_sha256: str,
        version: int,
        created_at: str,
        updated_at: str,
        expires_at: object,
        supersedes_memory_id: str | None,
    ) -> None:
        connection.execute(
            "INSERT INTO memories "
            "(memory_id, kind, content, source_chunk_ids_json, created_at, updated_at, "
            "expires_at, scope_id, content_sha256, status, version, supersedes_memory_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (
                memory_id,
                kind,
                content,
                json.dumps(source_chunk_ids),
                created_at,
                updated_at,
                expires_at,
                scope_id,
                content_sha256,
                version,
                supersedes_memory_id,
            ),
        )

    def _validate_memory_sources(self, kind: str, source_chunk_ids: tuple[str, ...]) -> None:
        if kind == "confirmed_conclusion" and not source_chunk_ids:
            raise ValueError("confirmed conclusions require source chunks")
        unknown = set(source_chunk_ids) - self._source_chunk_ids
        if unknown:
            raise ValueError("memory source chunks are not in the immutable catalog")

    def _authorize(
        self,
        tool_name: str,
        request: SaveResearchNoteInput | ExportResearchReportInput | ManageLongTermMemoryInput,
    ) -> ToolExecutionResult | None:
        if self._approvals.consume(tool_name, request, request.approval_token):
            return None
        approval = self._approvals.request(tool_name, request)
        return ToolExecutionResult(
            tool_name=tool_name,
            status="approval_required",
            summary={
                "approval_request_id": approval.request_id,
                "arguments_sha256": approval.arguments_sha256,
                "expires_at_epoch": approval.expires_at_epoch,
            },
        )

    def _safe_target(self, relative: Path) -> Path:
        pure = PurePath(str(relative))
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("workspace target must be a safe relative path")
        target = (self._runtime / Path(*pure.parts)).resolve()
        if self._runtime.resolve() not in target.parents:
            raise ValueError("workspace target escapes data/runtime")
        return target

    def _initialize_memory(self) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._memory_path)) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_chunk_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    scope_id TEXT NOT NULL DEFAULT 'global',
                    content_sha256 TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    version INTEGER NOT NULL DEFAULT 1,
                    supersedes_memory_id TEXT
                );
                CREATE INDEX IF NOT EXISTS memories_updated ON memories(updated_at DESC);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(memories)").fetchall()
            }
            migrations = {
                "scope_id": "ALTER TABLE memories ADD COLUMN scope_id TEXT NOT NULL DEFAULT 'global'",
                "content_sha256": "ALTER TABLE memories ADD COLUMN content_sha256 TEXT",
                "status": "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
                "version": "ALTER TABLE memories ADD COLUMN version INTEGER NOT NULL DEFAULT 1",
                "supersedes_memory_id": "ALTER TABLE memories ADD COLUMN supersedes_memory_id TEXT",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_scope_status "
                "ON memories(scope_id, status, updated_at DESC)"
            )
            missing_hashes = connection.execute(
                "SELECT memory_id, content FROM memories "
                "WHERE content_sha256 IS NULL OR content_sha256 = ''"
            ).fetchall()
            connection.executemany(
                "UPDATE memories SET content_sha256 = ? WHERE memory_id = ?",
                [(_content_fingerprint(row[1]), row[0]) for row in missing_hashes],
            )


_MEMORY_COLUMNS = (
    "memory_id, kind, content, source_chunk_ids_json, created_at, updated_at, expires_at, "
    "scope_id, content_sha256, status, version, supersedes_memory_id"
)


def _memory_item(row: tuple[object, ...]) -> dict[str, object]:
    return {
        "memory_id": row[0],
        "kind": row[1],
        "content": row[2],
        "source_chunk_ids": tuple(json.loads(str(row[3]))),
        "created_at": row[4],
        "updated_at": row[5],
        "expires_at": row[6],
        "scope_id": row[7],
        "content_sha256": row[8],
        "status": row[9],
        "version": row[10],
        "supersedes_memory_id": row[11],
    }


def _memory_not_found(action: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name="manage_long_term_memory",
        status="not_found",
        summary={"action": action},
    )


def _content_fingerprint(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _memory_terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = set(re.findall(r"[a-z0-9]+", normalized))
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    for run in cjk_runs:
        words.update(run[index : index + 2] for index in range(max(1, len(run) - 1)))
    return {term for term in words if term}


def _memory_relevance(query: str, content: str) -> float:
    normalized_query = unicodedata.normalize("NFKC", query).casefold().strip()
    normalized_content = unicodedata.normalize("NFKC", content).casefold()
    query_terms = _memory_terms(normalized_query)
    if not query_terms:
        return 0
    overlap = len(query_terms & _memory_terms(normalized_content)) / len(query_terms)
    return overlap + (1.0 if normalized_query in normalized_content else 0.0)


def _atomic_write(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("workspace target already exists")
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex}")
    temporary.write_text(content, encoding="utf-8")
    if path.exists() and not overwrite:
        temporary.unlink(missing_ok=True)
        raise FileExistsError("workspace target already exists")
    temporary.replace(path)
