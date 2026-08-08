from __future__ import annotations

import unittest

from paper_research_agent.evaluation.candidate_runner import (
    CandidatePaperEvaluationCase,
    summarize_candidate_paper_cases,
)


class CandidatePaperGoldRunnerTests(unittest.TestCase):
    def test_primary_metrics_exclude_full_text_detail_questions(self) -> None:
        cases = (
            CandidatePaperEvaluationCase(
                question_id="CPG001",
                split="dev",
                clue_scope="title_abstract",
                relevant_paper_ids=("C001", "C002"),
                candidate_paper_ids=("C001", "C009", "C002"),
                rewrite_status="fallback_original",
            ),
            CandidatePaperEvaluationCase(
                question_id="CPG002",
                split="dev",
                clue_scope="title_abstract",
                relevant_paper_ids=("C003", "C004"),
                candidate_paper_ids=("C003", "C009", "C008"),
                rewrite_status="success",
            ),
            CandidatePaperEvaluationCase(
                question_id="CPG003",
                split="sealed_test",
                clue_scope="full_text_detail",
                relevant_paper_ids=("C010",),
                candidate_paper_ids=("C099",),
                rewrite_status="success",
            ),
        )

        summary = summarize_candidate_paper_cases(cases, cutoffs=(1, 3, 8))

        self.assertEqual(summary["primary_question_count"], 2)
        self.assertEqual(summary["primary"]["recall_at_3_macro"], 0.75)
        self.assertEqual(summary["primary"]["all_target_hit_at_3"], 0.5)
        self.assertEqual(summary["primary"]["recall_at_1_macro"], 0.5)
        self.assertEqual(summary["diagnostic_full_text_question_count"], 1)
        self.assertEqual(summary["rewrite_fallback_count"], 1)
        self.assertEqual(summary["rewrite_success"]["recall_at_3_macro"], 0.5)
        self.assertEqual(summary["rewrite_fallback"]["recall_at_3_macro"], 1.0)

    def test_rejects_duplicate_candidate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate paper IDs must be unique"):
            CandidatePaperEvaluationCase(
                question_id="CPG001",
                split="dev",
                clue_scope="title_abstract",
                relevant_paper_ids=("C001",),
                candidate_paper_ids=("C001", "C001"),
                rewrite_status="success",
            )


if __name__ == "__main__":
    unittest.main()
