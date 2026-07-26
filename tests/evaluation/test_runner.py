from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.metrics import evidence_hit_at_k


class EvidenceMetricCoverageTests(unittest.TestCase):
    def test_unlabeled_query_is_not_counted_as_a_miss(self) -> None:
        self.assertIsNone(evidence_hit_at_k(["chunk-a"], set(), 1))
