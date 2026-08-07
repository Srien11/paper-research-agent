from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ContextMessage,
    ConversationWorkspace,
    GoalState,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.planner import GoalReconciler


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal(**overrides: object) -> GoalState:
    values: dict[str, object] = {
        "goal_id": "a" * 32,
        "objective": "比较 RAG 与 GraphRAG",
        "status": "active",
        "origin_turn_id": "b" * 32,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return GoalState(**values)


def _envelope(**overrides: object) -> AgentContextEnvelope:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "request_id": "request-1",
        "turn_id": "c" * 32,
        "current_message": "继续",
        "rag_mode": "preferred",
        "workspace": ConversationWorkspace(
            conversation_id="conversation-1",
            version=0,
            active_goal=_goal(),
            updated_at=_utc(),
        ),
        "recent_messages": (
            ContextMessage(turn_id="t1", sequence=1, role="user", content="之前的问题"),
        ),
        "recalled_context": (),
        "prepared_at": _utc(),
    }
    values.update(overrides)
    return AgentContextEnvelope(**values)


def _interpretation(relation: str, **overrides: object) -> TurnInterpretationV2:
    values: dict[str, Any] = {
        "relation": relation,
        "resolved_request": "继续比较 RAG 与 GraphRAG",
        "confidence": 0.9,
    }
    values.update(overrides)
    return TurnInterpretationV2(**values)


class _FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def with_structured_output(self, schema: object, method: str = "function_calling") -> object:
        del schema, method
        return self

    async def ainvoke(self, messages: object) -> object:
        del messages
        self.calls += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def _reconcile(
    reconciler: GoalReconciler,
    envelope: AgentContextEnvelope,
    interpretation: TurnInterpretationV2,
) -> object:
    return asyncio.run(reconciler.reconcile(envelope, interpretation))


class GoalReconcilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reconciler = GoalReconciler()

    def test_continue_reuses_goal_id(self) -> None:
        envelope = _envelope()
        decision = _reconcile(self.reconciler, envelope, _interpretation("continue_goal"))
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_resume_after_approval_reuses_goal_id(self) -> None:
        decision = _reconcile(
            self.reconciler, _envelope(), _interpretation("resume_after_approval")
        )
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_refine_reuses_goal_id_and_updates_objective(self) -> None:
        interpretation = _interpretation(
            "refine_goal", goal_change_summary="只比较推理成本和准确率"
        )
        decision = _reconcile(self.reconciler, _envelope(), interpretation)
        self.assertEqual(decision.action, "revise")
        self.assertEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.objective, "只比较推理成本和准确率")

    def test_new_goal_creates_new_goal_id(self) -> None:
        decision = _reconcile(self.reconciler, _envelope(), _interpretation("new_goal"))
        self.assertEqual(decision.action, "create")
        self.assertNotEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.objective, "继续比较 RAG 与 GraphRAG")

    def test_cancel_abandons_active_goal(self) -> None:
        decision = _reconcile(self.reconciler, _envelope(), _interpretation("cancel_goal"))
        self.assertEqual(decision.action, "abandon")
        self.assertEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.status, "abandoned")

    def test_answer_within_goal_keeps_goal(self) -> None:
        decision = _reconcile(
            self.reconciler, _envelope(), _interpretation("answer_within_goal")
        )
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_meta_conversation_does_not_create_goal(self) -> None:
        workspace = ConversationWorkspace(
            conversation_id="conversation-1", version=0, updated_at=_utc()
        )
        envelope = _envelope(workspace=workspace)
        decision = _reconcile(
            self.reconciler, envelope, _interpretation("meta_conversation")
        )
        self.assertEqual(decision.action, "keep")
        self.assertIsNone(decision.goal)

    def test_no_active_goal_creates_goal(self) -> None:
        workspace = ConversationWorkspace(
            conversation_id="conversation-1", version=0, updated_at=_utc()
        )
        envelope = _envelope(workspace=workspace)
        decision = _reconcile(
            self.reconciler, envelope, _interpretation("continue_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.origin_turn_id, "c" * 32)

    def test_model_completes_acceptance_criteria_for_create(self) -> None:
        fake = _FakeModel(
            [
                {
                    "objective": "比较 RAG 与 GraphRAG 并给出选型建议",
                    "acceptance_criteria": (
                        {
                            "criterion_id": "c1",
                            "description": "给出两项指标对比",
                        },
                    ),
                    "constraints": ("只依据本地论文",),
                }
            ]
        )
        reconciler = GoalReconciler(model=fake)
        decision = _reconcile(
            reconciler, _envelope(), _interpretation("new_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.objective, "比较 RAG 与 GraphRAG 并给出选型建议")
        self.assertEqual(len(decision.goal.acceptance_criteria), 1)

    def test_model_failure_falls_back_to_deterministic(self) -> None:
        fake = _FakeModel([RuntimeError("down")])
        reconciler = GoalReconciler(model=fake)
        decision = _reconcile(
            reconciler, _envelope(), _interpretation("new_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.objective, "继续比较 RAG 与 GraphRAG")
        self.assertIsNotNone(decision.goal.goal_id)


if __name__ == "__main__":
    unittest.main()
