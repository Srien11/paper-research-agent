from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.memory.context import (
    contextualize_retrieval_query,
    to_context_memory,
)
from paper_research_agent.memory.models import ShortTermMemoryTurn


def turn(question: str, answer: str, number: int = 1) -> ShortTermMemoryTurn:
    now = datetime.now(UTC)
    return ShortTermMemoryTurn(
        turn_id=f"{number:032x}",
        session_id="session-1",
        created_at=now,
        expires_at=now + timedelta(hours=24),
        user_question=question,
        standalone_question=question,
        assistant_claims=(answer,),
        status="answered",
    )


class MemoryContextTests(unittest.TestCase):
    def test_only_context_dependent_follow_up_uses_previous_question(self) -> None:
        history = (turn("BEIR 基准包含哪些任务？", "它包含多种检索任务。"),)
        resolved = contextualize_retrieval_query("它与 MTEB 有什么区别？", history)
        standalone = contextualize_retrieval_query("什么是 SelfCheckGPT？", history)
        self.assertIn("BEIR 基准包含哪些任务", resolved)
        self.assertIn("它与 MTEB 有什么区别", resolved)
        self.assertEqual(standalone, "什么是 SelfCheckGPT？")

    def test_context_memory_keeps_validated_answer_without_evidence_body(self) -> None:
        history = (turn("问题", "回答。"),)
        context_turns = to_context_memory(history)
        self.assertEqual(context_turns[0].user_question, "问题")
        self.assertEqual(context_turns[0].assistant_claims, ("回答。",))
        self.assertFalse(hasattr(context_turns[0], "evidence_text"))

    def test_third_follow_up_keeps_the_resolved_topic_anchor(self) -> None:
        previous = turn("它和 MTEB 有什么区别？", "有若干区别。")
        previous = previous.model_copy(
            update={
                "standalone_question": (
                    "当前问题：它和 MTEB 有什么区别？\n上一轮研究问题：BEIR 基准包含哪些任务？"
                )
            }
        )
        resolved = contextualize_retrieval_query("它的评测规模多大？", (previous,))
        self.assertIn("BEIR 基准包含哪些任务", resolved)
        self.assertIn("它的评测规模多大", resolved)
