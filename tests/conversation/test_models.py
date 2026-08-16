from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from paper_research_agent.conversation.models import ConversationTurn


def _turn(source_ids: tuple[str, ...]) -> ConversationTurn:
    return ConversationTurn(
        turn_id="0" * 32,
        conversation_id="conversation",
        sequence=1,
        user_question="question",
        status="completed",
        source_ids=source_ids,
        created_at=datetime.now(UTC),
    )


class ConversationModelTests(unittest.TestCase):
    def test_aggregate_source_ids_match_twelve_child_task_budget(self) -> None:
        source_ids = tuple(f"source-{index}" for index in range(1_200))
        self.assertEqual(len(_turn(source_ids).source_ids), 1_200)
        with self.assertRaises(ValidationError):
            _turn(source_ids + ("overflow",))


if __name__ == "__main__":
    unittest.main()
