from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def test_checked_in_dataset_has_30_unique_single_reviewer_queries(self) -> None:
        queries = load_dataset(PROJECT_ROOT / "evaluation/datasets/dev-silver-v1.jsonl")
        self.assertEqual(len(queries), 30)
        self.assertEqual(len({query.query_id for query in queries}), 30)
        self.assertTrue(all(query.reviewer_count == 1 for query in queries))

    def test_duplicate_query_ids_are_rejected(self) -> None:
        line = (
            '{"query_id":"Q001","query":"q","relevant_paper_ids":["C001"],'
            '"relevant_chunk_ids":[],"answerable":true,'
            '"annotation_status":"silver_single_reviewer","reviewer_count":1}'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)
