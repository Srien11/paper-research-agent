from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.metrics import (
    evidence_hit_at_k,
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
