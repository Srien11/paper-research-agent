from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from paper_research_agent.agent.orchestrator.models import MainAgentResult
from paper_research_agent.conversation.models import ConversationResolution
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


def _resolution() -> ConversationResolution:
    return ConversationResolution(
        original_question="比较 RAG 与 GraphRAG",
        standalone_question="比较 RAG 与 GraphRAG",
        chinese_query="比较 RAG 与 GraphRAG",
        confidence=1,
        episode_id="a" * 16,
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

    def _commit_args(self, start: object) -> dict[str, object]:
        workspace = start.workspace.model_copy(update={"summary": "更新后的摘要"})
        return {
            "run_id": start.run_id,
            "turn_id": start.turn_id,
            "expected_workspace_version": start.workspace.version,
            "workspace": workspace,
            "route": "local_rag",
            "status": "completed",
            "resolution": _resolution(),
            "assistant_summary": "已给出回答",
            "source_ids": ("source-1",),
            "result": _cached_result(start.request_id, start.run_id),
        }

    def test_commit_agent_run_updates_turn_workspace_and_run_atomically(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                start = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                args = self._commit_args(start)
                outcome = store.commit_agent_run(**args)  # type: ignore[attr-defined]
                self.assertTrue(outcome.committed)
                self.assertEqual(outcome.workspace_version, 1)
                loaded = store.load_workspace("conversation-1")  # type: ignore[attr-defined]
                self.assertEqual(loaded.version, 1)
                self.assertEqual(loaded.summary, "更新后的摘要")
                history = store.history("conversation-1")
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0].status, "completed")
                self.assertEqual(len(store.episodes("conversation-1")), 1)
                cached = store.load_agent_run("request-1")  # type: ignore[attr-defined]
                self.assertIsNotNone(cached)
                self.assertEqual(cached.answer, "已缓存回答")

    def test_commit_with_stale_workspace_version_is_rejected(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                start = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                args = self._commit_args(start)
                args["expected_workspace_version"] = 5
                outcome = store.commit_agent_run(**args)  # type: ignore[attr-defined]
                self.assertFalse(outcome.committed)
                self.assertEqual(outcome.reason, "version_conflict")
                self.assertEqual(store.load_workspace("conversation-1").version, 0)  # type: ignore[attr-defined]
                self.assertEqual(store.history("conversation-1"), ())
                self.assertIsNone(store.load_agent_run("request-1"))  # type: ignore[attr-defined]

    def test_injected_failure_rolls_back_entire_commit(self) -> None:
        start = self.sqlite.begin_agent_run(
            request_id="request-1",
            conversation_id="conversation-1",
            user_question="比较 RAG 与 GraphRAG",
        )
        with closing(sqlite3.connect(self.sqlite.path)) as connection:
            connection.execute(
                "CREATE TRIGGER boom BEFORE UPDATE ON conversation_workspaces "
                "BEGIN SELECT RAISE(ABORT, 'boom'); END;"
            )
            connection.commit()
        with self.assertRaises(sqlite3.Error):
            self.sqlite.commit_agent_run(**self._commit_args(start))
        self.assertEqual(self.sqlite.load_workspace("conversation-1").version, 0)
        self.assertEqual(self.sqlite.history("conversation-1"), ())
        self.assertIsNone(self.sqlite.load_agent_run("request-1"))

    def test_duplicate_commit_is_idempotent(self) -> None:
        for label, store in self._stores():
            with self.subTest(store=label):
                start = store.begin_agent_run(  # type: ignore[attr-defined]
                    request_id="request-1",
                    conversation_id="conversation-1",
                    user_question="比较 RAG 与 GraphRAG",
                )
                args = self._commit_args(start)
                first = store.commit_agent_run(**args)  # type: ignore[attr-defined]
                second = store.commit_agent_run(**args)  # type: ignore[attr-defined]
                self.assertTrue(first.committed)
                self.assertFalse(second.committed)
                self.assertEqual(second.reason, "already_completed")
                self.assertEqual(len(store.history("conversation-1")), 1)
                self.assertEqual(len(store.episodes("conversation-1")), 1)
                self.assertEqual(store.load_workspace("conversation-1").version, 1)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
