from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.hydrator import ContextHydrator
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ConversationWorkspace,
    GoalState,
    MainAgentRequest,
    TaskPlan,
)
from paper_research_agent.conversation.models import ConversationResolution
from paper_research_agent.conversation.store import InMemoryConversationStore


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _resolution(question: str) -> ConversationResolution:
    return ConversationResolution(
        original_question=question,
        standalone_question=question,
        chinese_query=question,
        confidence=1,
    )


def _goal(objective: str = "降低 RAG 幻觉评测指标") -> GoalState:
    return GoalState(
        goal_id="a" * 32,
        objective=objective,
        status="active",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )


def _task(title: str, objective: str) -> AgentTask:
    return AgentTask(
        task_id="collect-evidence",
        goal_id="a" * 32,
        title=title,
        objective=objective,
        success_criteria=("找到证据",),
        capability="local_rag",
        status="pending",
    )


def _plan(tasks: tuple[AgentTask, ...]) -> TaskPlan:
    return TaskPlan(
        plan_id="c" * 32,
        goal_id="a" * 32,
        revision=1,
        tasks=tasks,
        created_at=_utc(),
        updated_at=_utc(),
    )


class _RecordingProvider:
    def __init__(
        self,
        calls: list[bool],
        *,
        memories: tuple[dict[str, object], ...] = (),
        fail: bool = False,
    ) -> None:
        self.calls = calls
        self.memories = memories
        self.fail = fail

    async def search(self, query: str, *, limit: int = 5) -> tuple[dict[str, object], ...]:
        del query
        self.calls.append(True)
        if self.fail:
            raise OSError("memory backend unavailable")
        return tuple(self.memories[:limit])


class ContextHydratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore()
        self.calls: list[bool] = []
        self.provider = _RecordingProvider(self.calls)
        self.hydrator = ContextHydrator(self.store, memory_provider=self.provider)

    def _request(
        self, message: str = "比较 RAG 与 GraphRAG", rag_mode: str = "preferred"
    ) -> MainAgentRequest:
        return MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message=message,
            rag_mode=rag_mode,
        )

    def _workspace(self, **overrides: object) -> ConversationWorkspace:
        values: dict[str, object] = {
            "conversation_id": "conversation-1",
            "version": 0,
            "updated_at": _utc(),
        }
        values.update(overrides)
        return ConversationWorkspace(**values)

    def _hydrate(
        self,
        request: MainAgentRequest,
        workspace: ConversationWorkspace,
        *,
        hydrator: ContextHydrator | None = None,
    ) -> object:
        target = hydrator or self.hydrator
        return asyncio.run(target.hydrate(request, workspace, turn_id="a" * 32))

    def _populate(
        self, conversation_id: str, questions: list[str], summary: str | None = None
    ) -> list[str]:
        turn_ids: list[str] = []
        for question in questions:
            turn = self.store.begin_turn(conversation_id, question)
            store = self.store
            store.complete_turn(
                turn.turn_id,
                route="normal_chat",
                status="completed",
                resolution=_resolution(question),
                assistant_summary=summary,
            )
            turn_ids.append(turn.turn_id)
        return turn_ids

    def test_recent_six_turns_always_loaded_without_reference(self) -> None:
        self._populate("conversation-1", [f"无关问题{i}" for i in range(1, 9)])
        envelope = self._hydrate(self._request("一个全新主题"), self._workspace())
        self.assertEqual(len(envelope.recent_messages), 6)
        self.assertIn("无关问题8", envelope.recent_messages[-1].content)

    def test_recall_query_includes_active_goal_pending_task_and_unresolved(self) -> None:
        early_ids = self._populate(
            "conversation-1", ["早期关于幻觉评测的讨论", *[f"其他内容{i}" for i in range(8)]]
        )
        workspace = self._workspace(
            active_goal=_goal(),
            task_plan=_plan(
                tasks=(_task("检索幻觉相关论文", "找到关于幻觉评测的本地论文"),)
            ),
            unresolved_questions=("如何校准评测？",),
        )
        envelope = self._hydrate(self._request("继续"), workspace)
        recalled_ids = {item.source_id for item in envelope.recalled_context}
        self.assertIn(early_ids[0], recalled_ids)

    def test_long_term_memory_read_before_any_routing(self) -> None:
        self.provider.memories = (
            {
                "memory_id": "a" * 32,
                "content": "用户偏好用中文回答",
                "kind": "preference",
            },
        )
        envelope = self._hydrate(self._request(), self._workspace())
        self.assertEqual(self.calls, [True])
        memory_items = [
            item for item in envelope.recalled_context if item.kind == "long_term_memory"
        ]
        self.assertEqual(len(memory_items), 1)
        self.assertEqual(memory_items[0].trust, "research_context")
        self.assertEqual(memory_items[0].source_id, "a" * 32)
        self.assertEqual(memory_items[0].memory_kind, "preference")

    def test_remote_recall_across_short_and_long_histories(self) -> None:
        for total in (5, 20, 100):
            with self.subTest(total=total):
                store = InMemoryConversationStore()
                hydrator = ContextHydrator(store)
                questions = [f"无关主题{i}" for i in range(total)]
                insert_at = max(0, total // 3)
                questions[insert_at] = "早期关于幻觉评测的讨论"
                ids = self._populate_via(store, "conversation-1", questions)
                request = self._request("幻觉评测")
                request = request.model_copy(
                    update={"conversation_id": "conversation-1"}
                )
                workspace = self._workspace()
                workspace = workspace.model_copy(update={"conversation_id": "conversation-1"})
                envelope = asyncio.run(
                    hydrator.hydrate(request, workspace, turn_id="a" * 32)
                )
                recalled = [
                    item
                    for item in envelope.recalled_context
                    if item.kind == "conversation_turn"
                ]
                recent_ids = set(ids[max(0, total - 6) :])
                if insert_at < total - 6:
                    self.assertTrue(any(item.source_id == ids[insert_at] for item in recalled))
                    self.assertTrue(all(item.source_id not in recent_ids for item in recalled))
                else:
                    self.assertFalse(any(item.source_id == ids[insert_at] for item in recalled))

    def _populate_via(
        self, store: InMemoryConversationStore, conversation_id: str, questions: list[str]
    ) -> list[str]:
        turn_ids: list[str] = []
        for question in questions:
            turn = store.begin_turn(conversation_id, question)
            store.complete_turn(
                turn.turn_id,
                route="normal_chat",
                status="completed",
                resolution=_resolution(question),
            )
            turn_ids.append(turn.turn_id)
        return turn_ids

    def test_hydration_is_isolated_per_conversation(self) -> None:
        self._populate("conversation-1", ["甲主题"] * 6)
        self._populate("conversation-2", ["乙主题"] * 6)
        envelope = self._hydrate(self._request(), self._workspace())
        for message in envelope.recent_messages:
            self.assertNotIn("乙主题", message.content)

    def test_budget_trims_remote_then_oldest_recent_then_low_priority_memory(self) -> None:
        self.provider.memories = (
            {"memory_id": "m1", "content": "高优先级记忆内容", "kind": "preference", "relevance": 0.9},
            {"memory_id": "m2", "content": "低优先级记忆内容", "kind": "preference", "relevance": 0.1},
        )
        questions = [f"远距低相关{i}" + "内容" * 60 for i in range(8)]
        self._populate("conversation-1", questions)
        envelope = self._hydrate(self._request(), self._workspace())
        memory_items = [
            item for item in envelope.recalled_context if item.kind == "long_term_memory"
        ]
        if memory_items:
            self.assertEqual(memory_items[0].source_id, "m1")
        recalled_turns = [
            item for item in envelope.recalled_context if item.kind == "conversation_turn"
        ]
        self.assertLessEqual(len(recalled_turns), 5)

    def test_old_assistant_content_is_non_evidence(self) -> None:
        self._populate("conversation-1", ["一个问题"], summary="旧助手摘要")
        envelope = self._hydrate(self._request(), self._workspace())
        assistant_messages = [
            item for item in envelope.recent_messages if item.role == "assistant"
        ]
        self.assertTrue(assistant_messages)
        for message in assistant_messages:
            self.assertEqual(message.trust, "non_evidence")

    def test_memory_recall_failure_degrades_to_empty(self) -> None:
        self.provider.fail = True
        envelope = self._hydrate(self._request(), self._workspace())
        self.assertFalse(
            any(item.kind == "long_term_memory" for item in envelope.recalled_context)
        )


if __name__ == "__main__":
    unittest.main()
