from __future__ import annotations

import unittest

from pydantic import ValidationError

from paper_research_agent.web.models import QuestionRequest, SafeEvidenceSource


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


class SafeEvidenceSourceTests(unittest.TestCase):
    def test_excerpt_matches_runtime_configuration_boundary(self) -> None:
        values = {
            "citation_id": "E1",
            "chunk_id": "chunk",
            "corpus_id": "C001",
            "title": "title",
            "official_url": None,
            "section_id": None,
            "page_start": 1,
            "page_end": 1,
            "evidence_type": "text",
            "storage_class": "redistributable",
            "final_rank": 1,
        }
        source = SafeEvidenceSource(**values, excerpt="e" * 2_000)
        self.assertEqual(len(source.excerpt), 2_000)
        with self.assertRaises(ValidationError):
            SafeEvidenceSource(**values, excerpt="e" * 2_001)


if __name__ == "__main__":
    unittest.main()
