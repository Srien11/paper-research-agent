from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.metrics import (
    candidate_paper_recall,
    evidence_hit_at_k,
    explicit_corpus_id_accuracy,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self) -> None:
        ranking = ["C002", "C001", "C003"]
        relevant = {"C001", "C003"}
        self.assertEqual(recall_at_k(ranking, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(ranking, relevant), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranking, relevant, 3), 0.6934264036172708)
        self.assertEqual(evidence_hit_at_k(["x", "y"], {"y"}, 2), 1.0)

    def test_ndcg_counts_each_relevant_document_once(self) -> None:
        self.assertEqual(ndcg_at_k(["C001", "C001"], {"C001"}, 2), 1.0)

    def test_candidate_paper_recall_is_independent_of_chunk_recall(self) -> None:
        self.assertEqual(
            candidate_paper_recall(["C001", "T999", "C001"], {"C001", "T001"}),
            0.5,
        )
        self.assertIsNone(candidate_paper_recall(["C001"], set()))

    def test_explicit_corpus_id_accuracy_is_exact_and_ordered(self) -> None:
        self.assertEqual(
            explicit_corpus_id_accuracy(
                [("C001", "T001"), ("C001",)],
                [("C001", "T001"), ("T001",)],
            ),
            0.5,
        )
        self.assertIsNone(explicit_corpus_id_accuracy([], []))
        with self.assertRaisesRegex(ValueError, "counts differ"):
            explicit_corpus_id_accuracy([("C001",)], [])
