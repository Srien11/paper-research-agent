from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from paper_research_agent.agent.orchestrator.models import MainAgentResult
from paper_research_agent.conversation.store import (
    InMemoryConversationStore,
    SQLiteConversationStore,
)


def _cached_result(request_id: str, run_id: str) -> MainAgentResult:
    return MainAgentResult(
        run_id=run_id,
        request_id=request_id,
        conversation_id="conversation-1",
        status="completed",
        answer="已缓存回答",
        route_trace=("local_rag",),
        workspace_version=1,
    )


class AgentRunRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.sqlite = SQLiteConversationStore(Path(self._directory.name) / "agent.sqlite3")
        self.memory = InMemoryConversationStore()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _stores(self) -> tuple[tuple[str, object], ...]:
        return ("sqlite", self.sqlite), ("memory", self.memory)

    def _pending_turn_count(self, store: object, conversation_id: str) -> int:
        if isinstance(store, SQLiteConversationStore):
            with closing(sqlite3.connect(store.path)) as connection:
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()[0]
                )
        if isinstance(store, InMemoryConversationStore):
            return sum(
                1
                for item in store._turns.values()
                if item.conversation_id == conversation_id
            )
        raise AssertionError("unknown store")

    def _mark_completed(self, store: object, start: object, result: MainAgentResult) -> None:
        if isinstance(store, SQLiteConversationStore):
            run_id = start.run_id
            with closing(sqlite3.connect(store.path)) as connection:
                connection.execute(
                    "UPDATE main_agent_runs SET status = 'completed', result_json = ?, "
                    "updated_at = ? WHERE run_id = ?",
                    (result.model_dump_json(), "2026-08-07T00:00:00+00:00", run_id),
                )
                connection.commit()
            return
        if isinstance(store, InMemoryConversationStore):
            request_id = start.request_id
            record = store._runs[request_id]
            record.status = "completed"
            record.result = result
            return
        raise AssertionError("unknown store")

    def test_new_run_initializes_workspace_version_zero(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                start = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                self.assertEqual(start.outcome, "created")
                self.assertEqual(start.workspace.version, 0)
                self.assertEqual(start.workspace.conversation_id, "conversation-1")
                loaded = store.load_workspace("conversation-1")  # type: ignore[attr-defined]
                self.assertEqual(loaded.version, 0)

    def test_same_request_reuses_run_and_single_turn(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                first = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                second = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                self.assertEqual(first.run_id, second.run_id)
                self.assertEqual(first.turn_id, second.turn_id)
                self.assertEqual(second.outcome, "running_reused")
                self.assertEqual(self._pending_turn_count(store, "conversation-1"), 1)

    def test_completed_request_returns_cached_result(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                first = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                self._mark_completed(store, first, _cached_result("request-1", first.run_id))
                reused = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                self.assertEqual(reused.outcome, "completed_cached")
                self.assertIsNotNone(reused.result)
                self.assertEqual(reused.result.answer, "已缓存回答")
                self.assertEqual(reused.run_id, first.run_id)

    def test_load_agent_run_returns_cached_completed_result(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                self.assertIsNone(store.load_agent_run("request-1"))  # type: ignore[attr-defined]
                first = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                self._mark_completed(store, first, _cached_result("request-1", first.run_id))
                self.assertIsNotNone(store.load_agent_run("request-1"))  # type: ignore[attr-defined]

    def test_clear_removes_workspace_and_runs(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                store.clear("conversation-1")
                with self.assertRaises(ValueError):
                    store.load_workspace("conversation-1")  # type: ignore[attr-defined]
                self.assertIsNone(store.load_agent_run("request-1"))  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
