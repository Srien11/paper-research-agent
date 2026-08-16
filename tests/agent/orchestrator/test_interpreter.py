from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any

from paper_research_agent.agent.orchestrator.interpreter import TurnInterpreter
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ContextMessage,
    ConversationWorkspace,
    GoalState,
    RecalledContext,
    TurnInterpretationV2,
)


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal() -> GoalState:
    return GoalState(
        goal_id="a" * 32,
        objective="比较 RAG 与 GraphRAG",
        status="active",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )


def _envelope(**overrides: object) -> AgentContextEnvelope:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "request_id": "request-1",
        "turn_id": "a" * 32,
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
        "recalled_context": (
            RecalledContext(
                source_id="t5",
                kind="conversation_turn",
                content="远距相关内容",
                relevance=0.5,
                trust="non_evidence",
            ),
        ),
        "prepared_at": _utc(),
    }
    values.update(overrides)
    return AgentContextEnvelope(**values)


def _interpretation(relation: str, **overrides: object) -> dict[str, Any]:
    values: dict[str, Any] = {
        "relation": relation,
        "resolved_request": "继续当前任务",
        "confidence": 0.9,
    }
    values.update(overrides)
    return values


class _FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.messages: list[object] = []

    def with_structured_output(self, schema: object, method: str = "function_calling") -> object:
        del schema, method
        return self

    async def ainvoke(self, messages: object) -> object:
        self.messages.append(messages)
        self.calls += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def _interpret(model: _FakeModel, envelope: AgentContextEnvelope) -> TurnInterpretationV2:
    interpreter = TurnInterpreter(model)
    return asyncio.run(interpreter.interpret(envelope))


class TurnInterpreterTests(unittest.TestCase):
    def test_prompt_contains_bounded_context_content_as_untrusted_data(self) -> None:
        memory_id = "m" * 32
        envelope = _envelope(
            recalled_context=(
                RecalledContext(
                    source_id="t5",
                    kind="conversation_turn",
                    content="远距相关内容",
                    relevance=0.5,
                    trust="non_evidence",
                ),
                RecalledContext(
                    source_id=memory_id,
                    kind="long_term_memory",
                    content="用户偏好用中文回答",
                    relevance=0.8,
                    trust="research_context",
                ),
            )
        )
        model = _FakeModel(
            [_interpretation("continue_goal", selected_context_ids=(memory_id,))]
        )

        result = _interpret(model, envelope)

        user_content = model.messages[0][1].content
        system_content = model.messages[0][0].content
        self.assertIn("远距相关内容", user_content)
        self.assertIn("用户偏好用中文回答", user_content)
        self.assertIn(memory_id, user_content)
        self.assertIn("untrusted data", user_content)
        self.assertNotIn("用户偏好用中文回答", system_content)
        self.assertEqual(result.selected_context_ids, (memory_id,))

    def test_interpret_new_goal(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("new_goal")]), _envelope()
        )
        self.assertEqual(result.relation, "new_goal")

    def test_interpret_continue_goal(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("continue_goal")]), _envelope()
        )
        self.assertEqual(result.relation, "continue_goal")

    def test_interpret_refine_goal(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("refine_goal", goal_change_summary="只比较指标")]),
            _envelope(),
        )
        self.assertEqual(result.relation, "refine_goal")
        self.assertEqual(result.goal_change_summary, "只比较指标")

    def test_interpret_answer_within_goal(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("answer_within_goal")]), _envelope()
        )
        self.assertEqual(result.relation, "answer_within_goal")

    def test_interpret_cancel_goal(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("cancel_goal")]), _envelope()
        )
        self.assertEqual(result.relation, "cancel_goal")

    def test_interpret_resume_after_approval(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("resume_after_approval")]), _envelope()
        )
        self.assertEqual(result.relation, "resume_after_approval")

    def test_interpret_meta_conversation(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("meta_conversation")]), _envelope()
        )
        self.assertEqual(result.relation, "meta_conversation")

    def test_unknown_context_id_triggers_retry_then_success(self) -> None:
        model = _FakeModel(
            [
                _interpretation("continue_goal", selected_context_ids=("unknown",)),
                _interpretation("continue_goal", selected_context_ids=("t1",)),
            ]
        )
        result = _interpret(model, _envelope())
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.selected_context_ids, ("t1",))

    def test_unknown_context_id_twice_falls_back(self) -> None:
        model = _FakeModel(
            [
                _interpretation("new_goal", selected_context_ids=("unknown",)),
                _interpretation("new_goal", selected_context_ids=("unknown",)),
            ]
        )
        result = _interpret(model, _envelope())
        self.assertEqual(result.relation, "continue_goal")
        self.assertEqual(result.confidence, 0.0)

    def test_low_confidence_goal_change_requires_clarification(self) -> None:
        model = _FakeModel([_interpretation("refine_goal", confidence=0.4)])
        result = _interpret(model, _envelope())
        self.assertTrue(result.needs_clarification)
        self.assertIsNotNone(result.clarification_question)

    def test_high_confidence_goal_change_needs_no_clarification(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("new_goal", confidence=0.9)]), _envelope()
        )
        self.assertFalse(result.needs_clarification)

    def test_retry_after_first_model_failure(self) -> None:
        model = _FakeModel([RuntimeError("model down"), _interpretation("continue_goal")])
        result = _interpret(model, _envelope())
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.relation, "continue_goal")

    def test_two_failures_falls_back_to_continue(self) -> None:
        model = _FakeModel([RuntimeError("down"), RuntimeError("down")])
        result = _interpret(model, _envelope())
        self.assertEqual(result.relation, "continue_goal")
        self.assertEqual(result.confidence, 0.0)

    def test_fallback_creates_goal_when_none_active(self) -> None:
        envelope = _envelope(
            workspace=ConversationWorkspace(
                conversation_id="conversation-1",
                version=0,
                updated_at=_utc(),
            )
        )
        model = _FakeModel([RuntimeError("down"), RuntimeError("down")])
        result = _interpret(model, envelope)
        self.assertEqual(result.relation, "new_goal")

    def test_interpreter_never_outputs_route_or_capability(self) -> None:
        result = _interpret(
            _FakeModel([_interpretation("continue_goal")]), _envelope()
        )
        self.assertFalse(hasattr(result, "route"))
        self.assertFalse(hasattr(result, "capability"))


if __name__ == "__main__":
    unittest.main()
