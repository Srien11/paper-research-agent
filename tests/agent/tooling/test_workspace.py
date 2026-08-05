from __future__ import annotations

import sqlite3
import tempfile
import unittest
import uuid
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from paper_research_agent.agent.tooling.approval import ApprovalManager
from paper_research_agent.agent.tooling.contracts import (
    ExportResearchReportInput,
    ManageLongTermMemoryInput,
    SaveResearchNoteInput,
    ToolExecutionResult,
)
from paper_research_agent.agent.tooling.workspace import WorkspaceResearchTools


class WorkspaceTests(unittest.TestCase):
    @staticmethod
    def _approved_memory(
        tools: WorkspaceResearchTools,
        approvals: ApprovalManager,
        request: ManageLongTermMemoryInput,
    ) -> ToolExecutionResult:
        pending = tools.manage_long_term_memory(request)
        token = approvals.approve(str(pending.summary["approval_request_id"]))
        return tools.manage_long_term_memory(request.model_copy(update={"approval_token": token}))

    def test_one_time_approval_gates_notes_reports_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            approvals = ApprovalManager(secret=b"x" * 32)
            tools = WorkspaceResearchTools(
                Path(directory), approvals=approvals, source_chunk_ids={"chunk-1"}
            )
            note = SaveResearchNoteInput(title="Finding", content="Confirmed result")
            pending = tools.save_research_note(note)
            self.assertEqual(pending.status, "approval_required")
            token = approvals.approve(str(pending.summary["approval_request_id"]))
            saved = tools.save_research_note(note.model_copy(update={"approval_token": token}))
            self.assertEqual(saved.status, "ok")
            self.assertTrue((Path(directory) / saved.items[0]["relative_path"]).exists())
            self.assertEqual(
                tools.save_research_note(note.model_copy(update={"approval_token": token})).status,
                "approval_required",
            )

            report = ExportResearchReportInput(
                relative_path="review.md",
                format="markdown",
                content="# Review",
            )
            pending = tools.export_research_report(report)
            report_token = approvals.approve(str(pending.summary["approval_request_id"]))
            exported = tools.export_research_report(
                report.model_copy(update={"approval_token": report_token})
            )
            self.assertTrue((Path(directory) / exported.items[0]["relative_path"]).exists())

            add = ManageLongTermMemoryInput(
                action="add",
                kind="confirmed_conclusion",
                content="The validated conclusion",
                source_chunk_ids=("chunk-1",),
            )
            pending = tools.manage_long_term_memory(add)
            memory_token = approvals.approve(str(pending.summary["approval_request_id"]))
            added = tools.manage_long_term_memory(
                add.model_copy(update={"approval_token": memory_token})
            )
            found = tools.manage_long_term_memory(
                ManageLongTermMemoryInput(action="search", query="validated")
            )
            self.assertEqual(found.items[0]["memory_id"], added.items[0]["memory_id"])

    def test_memory_update_list_soft_delete_expiry_and_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            approvals = ApprovalManager(secret=b"x" * 32)
            tools = WorkspaceResearchTools(Path(directory), approvals=approvals)
            add = ManageLongTermMemoryInput(
                action="add",
                kind="preference",
                content="偏好简洁的中文研究报告格式",
                scope_id="owner",
            )
            added = self._approved_memory(tools, approvals, add)
            duplicate = self._approved_memory(tools, approvals, add)
            self.assertEqual(duplicate.items[0]["action"], "duplicate")
            self.assertEqual(duplicate.items[0]["memory_id"], added.items[0]["memory_id"])

            found = tools.manage_long_term_memory(
                ManageLongTermMemoryInput(
                    action="search",
                    query="中文报告",
                    scope_id="owner",
                )
            )
            self.assertEqual(found.items[0]["content"], "偏好简洁的中文研究报告格式")
            self.assertEqual(found.items[0]["status"], "active")
            self.assertEqual(found.items[0]["version"], 1)

            update = ManageLongTermMemoryInput(
                action="update",
                memory_id=str(added.items[0]["memory_id"]),
                content="偏好带摘要的中文研究报告格式",
                scope_id="owner",
            )
            updated = self._approved_memory(tools, approvals, update)
            self.assertEqual(updated.items[0]["version"], 2)
            self.assertEqual(
                updated.items[0]["supersedes_memory_id"],
                added.items[0]["memory_id"],
            )
            listed = tools.manage_long_term_memory(
                ManageLongTermMemoryInput(action="list", scope_id="owner", limit=20)
            )
            self.assertEqual(len(listed.items), 1)
            self.assertEqual(listed.items[0]["memory_id"], updated.items[0]["memory_id"])

            expired = ManageLongTermMemoryInput(
                action="add",
                kind="project_context",
                content="已经过期的项目背景",
                scope_id="owner",
                expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
            )
            self._approved_memory(tools, approvals, expired)
            listed = tools.manage_long_term_memory(
                ManageLongTermMemoryInput(action="list", scope_id="owner", limit=20)
            )
            self.assertEqual(len(listed.items), 1)

            delete = ManageLongTermMemoryInput(
                action="delete",
                memory_id=str(updated.items[0]["memory_id"]),
                scope_id="owner",
            )
            self._approved_memory(tools, approvals, delete)
            empty = tools.manage_long_term_memory(
                ManageLongTermMemoryInput(action="list", scope_id="owner")
            )
            self.assertEqual(empty.status, "not_found")

    def test_migrates_existing_v1_memory_table_without_losing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "runtime" / "long-term-memory-v1.sqlite3"
            path.parent.mkdir(parents=True)
            memory_id = uuid.uuid4().hex
            now = datetime.now(UTC).isoformat()
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    "content TEXT NOT NULL, source_chunk_ids_json TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at TEXT)"
                )
                connection.execute(
                    "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, "preference", "Keep answers concise", "[]", now, now, None),
                )
            tools = WorkspaceResearchTools(
                Path(directory), approvals=ApprovalManager(secret=b"x" * 32)
            )

            listed = tools.manage_long_term_memory(ManageLongTermMemoryInput(action="list"))

            self.assertEqual(listed.items[0]["memory_id"], memory_id)
            self.assertEqual(listed.items[0]["scope_id"], "global")
            self.assertEqual(len(str(listed.items[0]["content_sha256"])), 64)

    def test_confirmed_conclusion_requires_sources_and_unknown_sources_fail(self) -> None:
        with self.assertRaises(ValidationError):
            ManageLongTermMemoryInput(
                action="add",
                kind="confirmed_conclusion",
                content="Unsupported conclusion",
            )
        with tempfile.TemporaryDirectory() as directory:
            approvals = ApprovalManager(secret=b"x" * 32)
            tools = WorkspaceResearchTools(
                Path(directory), approvals=approvals, source_chunk_ids={"known"}
            )
            request = ManageLongTermMemoryInput(
                action="add",
                kind="confirmed_conclusion",
                content="Conclusion",
                source_chunk_ids=("unknown",),
            )
            pending = tools.manage_long_term_memory(request)
            token = approvals.approve(str(pending.summary["approval_request_id"]))
            with self.assertRaisesRegex(ValueError, "immutable catalog"):
                tools.manage_long_term_memory(request.model_copy(update={"approval_token": token}))

    def test_report_path_cannot_escape_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            approvals = ApprovalManager(secret=b"x" * 32)
            tools = WorkspaceResearchTools(Path(directory), approvals=approvals)
            request = ExportResearchReportInput(
                relative_path="../escape.md", format="markdown", content="blocked"
            )
            pending = tools.export_research_report(request)
            token = approvals.approve(str(pending.summary["approval_request_id"]))
            with self.assertRaisesRegex(ValueError, "safe relative"):
                tools.export_research_report(request.model_copy(update={"approval_token": token}))


if __name__ == "__main__":
    unittest.main()
