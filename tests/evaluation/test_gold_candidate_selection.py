from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_selection import (
    ANSWERABLE_DIFFICULTY_QUOTAS,
    ANSWERABLE_EVIDENCE_QUOTAS,
    ANSWERABLE_LANGUAGE_QUOTAS,
    ANSWERABLE_TASK_QUOTAS,
    UNANSWERABLE_REASON_QUOTAS,
    build_candidate_blueprint,
)


def _papers() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prefix, count, split in (("C", 60, "core"), ("T", 20, "challenge")):
        for index in range(1, count + 1):
            rows.append(
                {
                    "corpus_id": f"{prefix}{index:03d}",
                    "title": f"Evaluation paper {prefix}{index:03d}",
                    "dataset_split": split,
                    "storage_class": (
                        "redistributable" if index % 2 else "internal_research_only"
                    ),
                }
            )
    return rows


class GoldCandidateSelectionTests(unittest.TestCase):
    def test_blueprint_satisfies_frozen_answerable_quotas(self) -> None:
        rows = build_candidate_blueprint(_papers(), seed=20260806)
        answerable = [row for row in rows if row.answerable]
        self.assertEqual(len(rows), 80)
        self.assertEqual(len(answerable), 60)
        self.assertEqual(Counter(row.task_type for row in answerable), ANSWERABLE_TASK_QUOTAS)
        self.assertEqual(Counter(row.language for row in answerable), ANSWERABLE_LANGUAGE_QUOTAS)
        self.assertEqual(Counter(row.difficulty for row in answerable), ANSWERABLE_DIFFICULTY_QUOTAS)
        self.assertEqual(
            Counter(row.evidence_source for row in answerable), ANSWERABLE_EVIDENCE_QUOTAS
        )
        primary_splits = Counter(row.primary_split for row in answerable)
        self.assertEqual(primary_splits, {"core": 42, "challenge": 18})
        primary_papers = Counter(row.target_paper_ids[0] for row in answerable)
        self.assertGreaterEqual(len(primary_papers), 45)
        self.assertLessEqual(max(primary_papers.values()), 2)

    def test_blueprint_has_strict_unanswerable_taxonomy(self) -> None:
        rows = build_candidate_blueprint(_papers(), seed=20260806)
        negatives = [row for row in rows if not row.answerable]
        self.assertEqual(len(negatives), 20)
        self.assertEqual(
            Counter(row.unanswerable_reason for row in negatives),
            UNANSWERABLE_REASON_QUOTAS,
        )
        self.assertTrue(all(row.expected_status == "insufficient_evidence" for row in negatives))
        self.assertTrue(all(row.nearest_distractor_paper_ids for row in negatives))
        self.assertTrue(all(row.unanswerable_reason is None for row in rows if row.answerable))

    def test_blueprint_is_deterministic_and_does_not_use_governance_prompts(self) -> None:
        papers = _papers()
        for item in papers:
            item["challenge_question_ideas"] = ["LEAKED TEST IDEA"]
            item["challenge_tags"] = ["LEAKED_TAG"]
        first = build_candidate_blueprint(papers, seed=7)
        second = build_candidate_blueprint(list(reversed(papers)), seed=7)
        self.assertEqual(first, second)
        serialized = "\n".join(row.model_dump_json() for row in first)
        self.assertNotIn("LEAKED TEST IDEA", serialized)
        self.assertNotIn("LEAKED_TAG", serialized)


if __name__ == "__main__":
    unittest.main()
