from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationContextSnapshot,
    TurnInterpretation,
)
from paper_research_agent.conversation.resolver import resolution_from_interpretation


def candidate(turn_id: str, question: str, *, relevance: float) -> ConversationCandidate:
    return ConversationCandidate(
        turn_id=turn_id,
        sequence=1,
        user_question=question,
        standalone_question=question,
        route="normal_chat",
        assistant_summary="有限的历史结论。",
        status="completed",
        episode_id="a" * 16,
        relevance=relevance,
    )


class TurnInterpretationContractTests(unittest.TestCase):
    def test_interpretation_can_plan_local_and_dynamic_research_together(self) -> None:
        value = TurnInterpretation(
            depends_on_history=True,
            selected_history_turn_ids=("1" * 32,),
            standalone_question="基于本地论文与外部资料分析大模型测评。",
            chinese_query="大模型测评 方法 指标 基准",
            confidence=0.95,
            route="web_research",
            use_local_papers=True,
            use_web_research=True,
            use_dynamic_tools=True,
            research_mode="planned",
            reason="需要本地论文和外部研究共同取证",
        )

        self.assertTrue(value.use_local_papers)
        self.assertTrue(value.use_dynamic_tools)
        self.assertEqual(value.route, "web_research")

    def test_clarification_requires_a_question(self) -> None:
        with self.assertRaises(ValidationError):
            TurnInterpretation(
                depends_on_history=True,
                standalone_question="继续讨论测评。",
                chinese_query="测评",
                confidence=0.3,
                needs_clarification=True,
                route="normal_chat",
                reason="多个主题接近",
            )

    def test_model_cannot_select_turn_outside_supplied_context(self) -> None:
        known = candidate("1" * 32, "大模型测评", relevance=0.96)
        snapshot = ConversationContextSnapshot(
            original_question="参考一下知识库再说一次",
            recent_turns=(known,),
            recalled_turns=(),
            episodes=(),
            prepared_at=datetime.now(UTC),
        )
        interpretation = TurnInterpretation(
            depends_on_history=True,
            selected_history_turn_ids=("2" * 32,),
            standalone_question="分析不存在的主题。",
            chinese_query="不存在的主题",
            confidence=0.9,
            route="local_rag",
            use_local_papers=True,
            reason="采用历史",
        )

        with self.assertRaisesRegex(ValueError, "unknown conversation turn"):
            resolution_from_interpretation(snapshot, interpretation)


if __name__ == "__main__":
    unittest.main()
