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

    def test_best_effort_emit_never_breaks_the_agent(self) -> None:
        class BrokenSink:
            def write(self, event: AgentEvent) -> bool:
                del event
                raise sqlite3.Error("disk unavailable")

        self.assertFalse(emit_agent_event(BrokenSink(), _event()))
        self.assertFalse(emit_agent_event(None, _event()))


if __name__ == "__main__":
    unittest.main()
