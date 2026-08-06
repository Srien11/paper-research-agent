from __future__ import annotations

import unittest

from pydantic import ValidationError

from paper_research_agent.web.models import QuestionRequest


class QuestionRequestTests(unittest.TestCase):
    def test_rag_mode_defaults_to_disabled(self) -> None:
        request = QuestionRequest(question="hello")

        self.assertEqual(request.rag_mode, "disabled")

    def test_rag_mode_accepts_all_supported_values(self) -> None:
        for mode in ("disabled", "preferred", "required"):
            with self.subTest(mode=mode):
                request = QuestionRequest(question="hello", rag_mode=mode)
                self.assertEqual(request.rag_mode, mode)

    def test_legacy_local_only_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            QuestionRequest(question="hello", local_only=True)


if __name__ == "__main__":
    unittest.main()
