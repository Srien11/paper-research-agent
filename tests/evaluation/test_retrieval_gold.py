from __future__ import annotations

import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import GoldQuestion
from paper_research_agent.evaluation.retrieval_gold import (
    CATEGORY_QUOTAS,
    RetrievalGoldQuery,
    build_retrieval_gold,
)


def _question(index: int, category: str) -> GoldQuestion:
    paper_ids = (f"C{index:03d}",) if category != "cross_paper" else (f"C{index:03d}", "T001")
    span_count = 1 if category == "single_one_span" else 2
    task_type = "figure_table_explanation" if category == "figure" else "definition_scope"
    spans = []
    relations = []
    claims = []
    for span_index in range(span_count):
        paper_id = paper_ids[min(span_index, len(paper_ids) - 1)]
        quote = f"evidence {index} {span_index}"
        span_id = f"S{span_index + 1:03d}"
        claim_id = f"M{span_index + 1}"
        spans.append(
            {
                "span_id": span_id,
                "paper_id": paper_id,
                "evidence_version_id": f"asset-{paper_id}",
                "page": 1,
                "element_id": f"element-{index}-{span_index}",
                "raw_span_start": 0,
                "raw_span_end": len(quote),
                "raw_quote": quote,
                "span_hash": hashlib.sha256(quote.encode()).hexdigest(),
                "support_role": "required",
                "projected_chunk_ids": [f"chk-{index}-{span_index}"],
            }
        )
        claims.append({"claim_id": claim_id, "text": f"claim {index} {span_index}"})
        relations.append({"claim_id": claim_id, "span_id": span_id, "relation": "supports"})
    return GoldQuestion.model_validate(
        {
            "question_id": f"GQ{index:03d}",
            "split": "dev",
            "language": ("zh", "en", "mixed")[index % 3],
            "task_type": task_type,
            "difficulty": ("easy", "medium", "hard")[index % 3],
            "question": f"question {index}",
            "answerable": True,
            "must_have_claims": claims,
            "evidence_spans": spans,
            "citation_relations": relations,
            "annotation_status": "silver_generated",
            "corpus_version": "corpus-v1",
            "knowledge_cutoff": "2026-07-26",
        }
    )


class RetrievalGoldTests(unittest.TestCase):
    def test_selects_exact_category_mix_and_preserves_silver_status(self) -> None:
        rows = []
        index = 1
        for category, count in CATEGORY_QUOTAS.items():
            for _ in range(count):
                rows.append(_question(index, category))
                index += 1
        selected = build_retrieval_gold(rows)
        self.assertEqual(len(selected), 30)
        self.assertEqual(Counter(item.category for item in selected), CATEGORY_QUOTAS)
        self.assertTrue(all(item.annotation_status == "silver_generated" for item in selected))

    def test_contract_rejects_chunk_projection_mismatch(self) -> None:
        source = _question(1, "single_one_span")
        with self.assertRaises(ValueError):
            RetrievalGoldQuery.model_validate(
                {
                    "query_id": "RG001",
                    "source_question_id": source.question_id,
                    "category": "single_one_span",
                    "query": source.question,
                    "language": source.language,
                    "difficulty": source.difficulty,
                    "relevant_paper_ids": ["C001"],
                    "evidence_spans": [source.evidence_spans[0].model_dump(mode="json")],
                    "required_span_groups": [["S001"]],
                    "relevant_chunk_ids": ["wrong"],
                    "annotation_status": "silver_generated",
                    "corpus_version": "corpus-v1",
                }
            )


if __name__ == "__main__":
    unittest.main()
