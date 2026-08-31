from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import GoldQuestion
from paper_research_agent.evaluation.silver_quality import (
    SilverQuarantineManifest,
    apply_silver_quarantine,
)


def _question(identifier: int) -> GoldQuestion:
    quote = "private evidence"
    return GoldQuestion.model_validate(
        {
            "question_id": f"GQ{identifier:03d}",
            "split": "dev",
            "language": "zh",
            "task_type": "definition_scope",
            "difficulty": "medium",
            "question": "private question",
            "answerable": True,
            "must_have_claims": [{"claim_id": "M1", "text": "private claim"}],
            "evidence_spans": [
                {
                    "span_id": "S001",
                    "paper_id": "C001",
                    "evidence_version_id": "asset-1",
                    "page": 1,
                    "element_id": "element-1",
                    "raw_span_start": 0,
                    "raw_span_end": len(quote),
                    "raw_quote": quote,
                    "span_hash": hashlib.sha256(quote.encode()).hexdigest(),
                    "support_role": "required",
                    "projected_chunk_ids": ["chunk-1"],
                }
            ],
            "citation_relations": [{"claim_id": "M1", "span_id": "S001", "relation": "supports"}],
            "annotation_status": "silver_generated",
            "corpus_version": "corpus-v1",
            "knowledge_cutoff": "2026-07-26",
        }
    )


class SilverQuarantineTests(unittest.TestCase):
    def test_quarantine_removes_ids_and_returns_body_free_counts(self) -> None:
        manifest = SilverQuarantineManifest.model_validate(
            {
                "dataset_id": "test-v1",
                "scope_question_ids": ["GQ001", "GQ002"],
                "entries": [
                    {
                        "question_id": "GQ002",
                        "reason_code": "incomplete_reference",
                        "review_status": "confirmed",
                    }
                ],
            }
        )
        retained, report = apply_silver_quarantine((_question(1), _question(2)), manifest)
        self.assertEqual([question.question_id for question in retained], ["GQ001"])
        self.assertEqual(report.excluded_question_ids, ("GQ002",))
        self.assertEqual(report.scope_question_count, 2)
        self.assertEqual(report.outside_scope_question_count, 0)
        self.assertEqual(report.reason_counts, {"incomplete_reference": 1})
        self.assertNotIn("private", report.model_dump_json())

    def test_quarantine_rejects_unknown_and_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            SilverQuarantineManifest.model_validate(
                {
                    "dataset_id": "test-v1",
                    "scope_question_ids": ["GQ001"],
                    "entries": [
                        {
                            "question_id": "GQ001",
                            "reason_code": "incomplete_reference",
                            "review_status": "confirmed",
                        },
                        {
                            "question_id": "GQ001",
                            "reason_code": "degraded_ocr_evidence",
                            "review_status": "high_risk",
                        },
                    ],
                }
            )
        manifest = SilverQuarantineManifest.model_validate(
            {
                "dataset_id": "test-v1",
                "scope_question_ids": ["GQ001", "GQ999"],
                "entries": [
                    {
                        "question_id": "GQ999",
                        "reason_code": "incomplete_reference",
                        "review_status": "confirmed",
                    }
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown question IDs"):
            apply_silver_quarantine((_question(1),), manifest)


if __name__ == "__main__":
    unittest.main()
