from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from paper_research_agent.agent.observability import (
    AgentEvent,
    SQLiteAgentEventLogger,
    emit_agent_event,
    safe_fingerprint,
)


def _event(**overrides: object) -> AgentEvent:
    values: dict[str, object] = {
        "run_id": "a" * 32,
        "occurred_at": datetime(2026, 8, 4, tzinfo=UTC),
        "event_type": "tool_completed",
        "status": "succeeded",
        "component": "tool",
        "name": "search_corpus",
        "duration_ms": 12.5,
        "question_sha256": "b" * 64,
        "thread_sha256": "c" * 64,
        "hit_count": 4,
        "tool_call_count": 1,
    }
    values.update(overrides)
    return AgentEvent.model_validate(values)


class AgentEventTests(unittest.TestCase):
    def test_main_entry_event_kinds_accept_only_fixed_width_summaries(self) -> None:
        kinds = (
            ("main_runtime_built", "succeeded"),
            ("main_run_started", "started"),
            ("capability_routed", "succeeded"),
            ("child_completed", "succeeded"),
            ("main_run_paused", "succeeded"),
            ("main_commit_rejected", "failed"),
            ("main_run_completed", "succeeded"),
            ("deprecated_endpoint_used", "succeeded"),
        )
        for event_type, status in kinds:
            with self.subTest(event_type=event_type):
                event = _event(
                    event_type=event_type,
                    status=status,
                    duration_ms=None if status == "started" else 12.5,
                    component="runtime",
                    name="main_agent",
                    reason_code=(
                        "commit_validation_failed"
                        if event_type == "main_commit_rejected"
                        else None
                    ),
                    capability=(
                        "local_rag"
                        if event_type in {"capability_routed", "child_completed"}
                        else None
                    ),
                    child_status=(
                        "completed" if event_type == "child_completed" else None
                    ),
                    endpoint=(
                        "ask" if event_type == "deprecated_endpoint_used" else None
                    ),
                )
                self.assertEqual(event.event_type, event_type)

    def test_main_event_contract_rejects_raw_private_fields(self) -> None:
        forbidden = {
            "message": "private prompt",
            "answer": "private answer",
            "chunk_id": "chunk-private",
            "attachment_filename": "C:\\private\\paper.pdf",
            "tool_arguments": {"token": "secret"},
        }
        for field, value in forbidden.items():
            with self.subTest(field=field), self.assertRaises(ValidationError):
                AgentEvent.model_validate(
                    {
                        **_event().model_dump(),
                        field: value,
                    }
                )

    def test_contract_rejects_unknown_payload_and_raw_identifiers(self) -> None:
        with self.assertRaises(ValidationError):
            AgentEvent.model_validate(
                {
                    **_event().model_dump(),
                    "query": "raw private query",
                }
            )
        with self.assertRaises(ValidationError):
            _event(thread_sha256="session-1")

    def test_fingerprint_is_stable_without_retaining_plaintext(self) -> None:
        digest = safe_fingerprint("  Sensitive Question  ")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, safe_fingerprint("Sensitive Question"))
        self.assertNotIn("Sensitive", digest)

    def test_sqlite_logger_persists_queryable_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime" / "agent-events.sqlite3"
            logger = SQLiteAgentEventLogger(path)

            self.assertTrue(logger.write(_event()))
            self.assertTrue(
                logger.write(
                    _event(
                        event_type="run_completed",
                        component="runtime",
                        name="research_agent",
                        termination_reason="evidence_sufficient",
                    )
                )
            )

            with closing(sqlite3.connect(path)) as connection:
                rows = connection.execute(
                    "SELECT event_type, name, hit_count, termination_reason "
                    "FROM agent_events ORDER BY event_id"
                ).fetchall()
            self.assertEqual(
                rows,
                [
                    ("tool_completed", "search_corpus", 4, None),
                    (
                        "run_completed",
                        "research_agent",
                        4,
                        "evidence_sufficient",
                    ),
                ],
            )

    def test_sqlite_logger_migrates_existing_v1_table_for_main_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent-events.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    "CREATE TABLE agent_events ("
                    "event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "schema_version TEXT NOT NULL, run_id TEXT NOT NULL, "
                    "occurred_at TEXT NOT NULL, event_type TEXT NOT NULL, "
                    "status TEXT NOT NULL, component TEXT NOT NULL, name TEXT NOT NULL, "
                    "duration_ms REAL, question_sha256 TEXT, thread_sha256 TEXT, "
                    "step_id_sha256 TEXT, query_sha256 TEXT, error_type TEXT, "
                    "reason_code TEXT, termination_reason TEXT, degraded INTEGER, "
                    "evidence_sufficient INTEGER, hit_count INTEGER, "
                    "requested_count INTEGER, returned_count INTEGER, "
                    "evidence_count INTEGER, tool_call_count INTEGER, "
                    "replan_count INTEGER, max_steps INTEGER, max_tool_calls INTEGER, "
                    "timeout_seconds REAL)"
                )
                connection.execute("PRAGMA user_version = 1")

            logger = SQLiteAgentEventLogger(path)
            event = _event(
                event_type="deprecated_endpoint_used",
                status="succeeded",
                component="runtime",
                name="compatibility",
                endpoint="ask",
            )

            self.assertTrue(logger.write(event))
            with closing(sqlite3.connect(path)) as connection:
                row = connection.execute(
                    "SELECT endpoint FROM agent_events"
                ).fetchone()
                version = connection.execute("PRAGMA user_version").fetchone()
            self.assertEqual(row, ("ask",))
            self.assertEqual(version, (2,))

    def test_best_effort_emit_never_breaks_the_agent(self) -> None:
        class BrokenSink:
            def write(self, event: AgentEvent) -> bool:
                del event
                raise sqlite3.Error("disk unavailable")

        self.assertFalse(emit_agent_event(BrokenSink(), _event()))
        self.assertFalse(emit_agent_event(None, _event()))


if __name__ == "__main__":
    unittest.main()
