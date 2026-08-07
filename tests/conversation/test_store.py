from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationResolution,
)
from paper_research_agent.conversation.store import SQLiteConversationStore


class ConversationStoreTests(unittest.TestCase):
    def test_turn_lifecycle_is_isolated_and_clear_removes_every_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "conversation.sqlite3")
            first = store.begin_turn("conversation-a", "大模型测评")
            resolution = ConversationResolution(
                original_question="大模型测评",
                standalone_question="大模型测评",
                chinese_query="大模型测评",
                confidence=1,
                episode_id="a" * 16,
            )
            self.assertTrue(
                store.complete_turn(
                    first.turn_id,
                    route="normal_chat",
                    status="completed",
                    resolution=resolution,
                    assistant_summary="讨论了常见评测维度。",
                )
            )
            other = store.begin_turn("conversation-b", "RAG")
            store.complete_turn(
                other.turn_id,
                route="local_rag",
                status="insufficient_evidence",
                resolution=resolution.model_copy(
                    update={
                        "original_question": "RAG",
                        "standalone_question": "RAG",
                        "chinese_query": "RAG",
                    }
                ),
            )

            self.assertEqual(store.history("conversation-a")[0].user_question, "大模型测评")
            self.assertEqual(store.episodes("conversation-a")[0].summary, "大模型测评")
            self.assertEqual(store.history("conversation-b")[0].user_question, "RAG")
            self.assertEqual(store.clear("conversation-a"), 1)
            self.assertEqual(store.history("conversation-a"), ())
            self.assertEqual(store.episodes("conversation-a"), ())
            self.assertEqual(len(store.history("conversation-b")), 1)

    def test_pending_turn_is_not_recalled_or_completed_after_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "conversation.sqlite3")
            pending = store.begin_turn("conversation-a", "尚未完成")
            self.assertEqual(store.history("conversation-a"), ())
            self.assertEqual(store.clear("conversation-a"), 1)
            resolution = ConversationResolution(
                original_question="尚未完成",
                standalone_question="尚未完成",
                chinese_query="尚未完成",
                confidence=1,
            )
            self.assertFalse(
                store.complete_turn(
                    pending.turn_id,
                    route="normal_chat",
                    status="completed",
                    resolution=resolution,
                )
            )

    def test_selected_history_turn_and_relevance_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteConversationStore(Path(directory) / "conversation.sqlite3")
            anchor = store.begin_turn("conversation-a", "大模型测评")
            base = ConversationResolution(
                original_question="大模型测评",
                standalone_question="大模型测评",
                chinese_query="大模型测评",
                confidence=1,
                episode_id="a" * 16,
            )
            store.complete_turn(
                anchor.turn_id,
                route="normal_chat",
                status="completed",
                resolution=base,
            )
            follow_up = store.begin_turn("conversation-a", "结合一下知识库")
            candidate = ConversationCandidate(
                turn_id=anchor.turn_id,
                sequence=1,
                user_question="大模型测评",
                standalone_question="大模型测评",
                route="normal_chat",
                status="completed",
                episode_id="a" * 16,
                relevance=0.96,
            )
            resolution = ConversationResolution(
                original_question="结合一下知识库",
                standalone_question="请基于本地论文知识库，继续分析大模型测评。",
                chinese_query="请基于本地论文知识库，继续分析大模型测评。",
                candidates=(candidate,),
                selected_turn_ids=(anchor.turn_id,),
                confidence=0.96,
                episode_id="a" * 16,
            )
            store.complete_turn(
                follow_up.turn_id,
                route="local_rag",
                status="insufficient_evidence",
                resolution=resolution,
            )

            saved = store.history("conversation-a")[-1]
            self.assertEqual(saved.selected_history_turn_ids, (anchor.turn_id,))
            self.assertEqual(saved.selected_history_relevances, (0.96,))
