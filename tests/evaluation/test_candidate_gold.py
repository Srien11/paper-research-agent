from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_research_agent.evaluation.candidate_gold import (
    CandidatePaperGoldQuestion,
    candidate_gold_summary,
    load_candidate_paper_gold,
)


def _record(**updates):
    value = {
        "schema_version": "candidate-paper-gold-v1",
        "question_id": "CPG001",
        "split": "dev",
        "language": "zh",
        "task_type": "multi_paper_comparison",
        "difficulty": "medium",
        "clue_scope": "title_abstract",
        "question": "比较两类事实性评估方法分别如何衡量原子事实与采样一致性。",
        "relevant_paper_ids": ["C007", "C014"],
        "nearest_distractor_paper_ids": ["C049"],
        "annotation_reasons": {
            "C007": "摘要将长文本拆成原子事实并计算事实精度。",
            "C014": "摘要通过多次采样的一致性检测幻觉。",
        },
        "annotation_status": "delegated_expert_reviewed",
        "reviewer_id": "codex-local-reviewer",
        "review_passes": ["authorship", "reverse_verification"],
        "corpus_version": "llm-eval-reliability-v1.0.0-2026-07-26",
    }
    value.update(updates)
    return value


class CandidatePaperGoldQuestionTests(unittest.TestCase):
    def test_accepts_reviewed_multi_paper_question(self) -> None:
        question = CandidatePaperGoldQuestion.model_validate(_record())

        self.assertEqual(question.relevant_paper_ids, ("C007", "C014"))
        self.assertEqual(question.clue_scope, "title_abstract")

    def test_rejects_explicit_corpus_identifier_in_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain corpus IDs"):
            CandidatePaperGoldQuestion.model_validate(
                _record(question="比较 C007 与 C014。")
            )

    def test_rejects_reason_keys_that_do_not_match_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "reasons must match"):
            CandidatePaperGoldQuestion.model_validate(
                _record(annotation_reasons={"C007": "only one reason"})
            )

    def test_rejects_duplicate_or_overlapping_paper_ids(self) -> None:
        with self.assertRaises(ValueError):
            CandidatePaperGoldQuestion.model_validate(
                _record(relevant_paper_ids=["C007", "C007"])
            )
        with self.assertRaisesRegex(ValueError, "distractors must not overlap"):
            CandidatePaperGoldQuestion.model_validate(
                _record(nearest_distractor_paper_ids=["C014"])
            )

    def test_load_rejects_duplicate_question_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.jsonl"
            path.write_text(
                "\n".join(json.dumps(_record(), ensure_ascii=False) for _ in range(2)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "question_id values must be unique"):
                load_candidate_paper_gold(path)

    def test_summary_contains_counts_but_not_question_bodies(self) -> None:
        questions = (
            CandidatePaperGoldQuestion.model_validate(_record()),
            CandidatePaperGoldQuestion.model_validate(
                _record(
                    question_id="CPG002",
                    split="sealed_test",
                    clue_scope="full_text_detail",
                )
            ),
        )

        summary = candidate_gold_summary(questions)

        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(summary["title_abstract_count"], 1)
        self.assertEqual(summary["sealed_test_count"], 1)
        self.assertNotIn("questions", summary)


if __name__ == "__main__":
    unittest.main()
