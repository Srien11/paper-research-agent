from __future__ import annotations

import unittest

from paper_research_agent.agent.orchestrator.identifiers import (
    child_session_id,
    dynamic_thread_id,
)
from paper_research_agent.memory.models import normalize_session_id


class InternalIdentifierTests(unittest.TestCase):
    def test_preserves_legacy_child_identifier_when_valid(self) -> None:
        expected = "research::conversation::run::task"
        self.assertEqual(
            child_session_id("research", "conversation", "run", "task"),
            expected,
        )

    def test_hashes_overlong_child_identifier_without_truncating_components(self) -> None:
        value = child_session_id("research", "c" * 48, "r" * 32, "t" * 35)
        self.assertTrue(value.startswith("research:h1:"))
        self.assertLessEqual(len(value), 128)
        self.assertEqual(normalize_session_id(value), value)
        self.assertEqual(
            value,
            child_session_id("research", "c" * 48, "r" * 32, "t" * 35),
        )

    def test_hashes_child_identifier_with_unsafe_component_characters(self) -> None:
        value = child_session_id("chat", "会话 id", "run", "task")
        self.assertTrue(value.startswith("chat:h1:"))
        self.assertEqual(normalize_session_id(value), value)

    def test_preserves_legacy_dynamic_thread_when_bounded(self) -> None:
        self.assertEqual(
            dynamic_thread_id("conversation", "run", "task"),
            "conversation::run::task",
        )

    def test_hashes_overlong_dynamic_thread_deterministically(self) -> None:
        value = dynamic_thread_id("c" * 256, "r" * 32, "t" * 64)
        self.assertTrue(value.startswith("dynamic:h1:"))
        self.assertLessEqual(len(value), 240)
        self.assertEqual(value, dynamic_thread_id("c" * 256, "r" * 32, "t" * 64))


if __name__ == "__main__":
    unittest.main()
