from __future__ import annotations

import unittest
from datetime import UTC, datetime

from paper_research_agent.conversation.models import ConversationTurn
from paper_research_agent.conversation.resolver import (
    build_conversation_context,
    resolve_conversation_question,
)


def turn(
    sequence: int,
    question: str,
    *,
    route: str = "normal_chat",
    episode_id: str | None = None,
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=f"{sequence:032x}",
        conversation_id="conversation-a",
        sequence=sequence,
        user_question=question,
        standalone_question=question,
        route=route,
        status="completed",
        assistant_summary=f"关于{question}的有限结论。",
        episode_id=episode_id,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


class ConversationResolverTests(unittest.TestCase):
    def test_context_preparation_always_includes_recent_turn_without_trigger_rules(self) -> None:
        snapshot = build_conversation_context(
            "参考一下知识库再说一次",
            (turn(1, "大模型测评", route="normal_chat"),),
        )

        self.assertEqual(
            tuple(item.user_question for item in snapshot.recent_turns),
            ("大模型测评",),
        )

    def test_context_preparation_keeps_far_semantic_candidate_outside_recent_window(self) -> None:
        history = [turn(1, "大模型测评的指标和基准", episode_id="a" * 16)]
        history.extend(turn(index, f"无关主题 {index}") for index in range(2, 102))

        snapshot = build_conversation_context("回到大模型测评的指标", history)

        self.assertNotIn(f"{1:032x}", {item.turn_id for item in snapshot.recent_turns})
        self.assertIn(f"{1:032x}", {item.turn_id for item in snapshot.recalled_turns})

    def test_generic_knowledge_base_follow_up_uses_previous_cross_route_topic(self) -> None:
        resolution = resolve_conversation_question(
            "结合一下知识库",
            (turn(1, "大模型测评", route="normal_chat"),),
        )

        self.assertEqual(resolution.selected_turn_ids, (f"{1:032x}",))
        self.assertIn("大模型测评", resolution.standalone_question)
        self.assertIn("本地论文知识库", resolution.standalone_question)
        self.assertFalse(resolution.needs_clarification)

    def test_reference_knowledge_base_and_repeat_uses_previous_topic(self) -> None:
        for question in (
            "参考一下知识库再说一次",
            "按知识库再回答一次",
            "用本地论文知识库重新解释一下",
            "再结合知识库说一遍",
            "根据知识库补充一下",
        ):
            with self.subTest(question=question):
                resolution = resolve_conversation_question(
                    question,
                    (turn(1, "大模型测评", route="normal_chat"),),
                )

                self.assertEqual(resolution.selected_turn_ids, (f"{1:032x}",))
                self.assertIn("大模型测评", resolution.standalone_question)
                self.assertFalse(resolution.needs_clarification)

    def test_explicit_new_topic_does_not_inherit_old_topic(self) -> None:
        resolution = resolve_conversation_question(
            "结合知识库解释 RAG",
            (turn(1, "大模型测评"),),
        )
        self.assertEqual(resolution.standalone_question, "结合知识库解释 RAG")
        self.assertEqual(resolution.selected_turn_ids, ())
        self.assertFalse(resolution.needs_clarification)

    def test_old_topic_can_be_recalled_after_five_twenty_or_one_hundred_turns(self) -> None:
        for gap in (5, 20, 100):
            with self.subTest(gap=gap):
                history = [turn(1, "大模型测评的指标和基准", episode_id="a" * 16)]
                history.extend(turn(index, f"无关主题 {index}") for index in range(2, gap + 2))

                resolution = resolve_conversation_question("回到之前的大模型测评", history)

                self.assertEqual(resolution.selected_turn_ids, (f"{1:032x}",))
                self.assertIn("大模型测评", resolution.standalone_question)

    def test_close_historical_topics_require_clarification(self) -> None:
        history = (
            turn(1, "大模型测评方法", episode_id="a" * 16),
            turn(2, "RAG 测评方法", episode_id="b" * 16),
        )
        resolution = resolve_conversation_question("回到之前的测评方法", history)
        self.assertTrue(resolution.needs_clarification)
        self.assertIn("还是", resolution.clarification_question or "")

    def test_conversation_ids_never_cross_recall_boundary(self) -> None:
        own_history = (turn(1, "RAG"),)
        resolution = resolve_conversation_question("结合一下知识库", own_history)
        self.assertNotIn("大模型测评", resolution.standalone_question)


if __name__ == "__main__":
    unittest.main()
