from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import GoldQuestion
from paper_research_agent.evaluation.rag_gold_runner import (
    RAGJudgeResult,
    evaluate_rag_gold,
)


def _question(identifier: int, *, answerable: bool) -> GoldQuestion:
    quote = "gold evidence"
    base = {
        "question_id": f"GQ{identifier:03d}",
        "split": "dev",
        "language": "zh",
        "task_type": "definition_scope",
        "difficulty": "medium",
        "question": f"private question {identifier}",
        "answerable": answerable,
        "must_have_claims": ([{"claim_id": "M1", "text": "gold claim"}] if answerable else []),
        "forbidden_claims": ([{"claim_id": "F1", "text": "false claim"}] if answerable else []),
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
                "support_role": "required" if answerable else "distractor",
                "projected_chunk_ids": ["chk-1"],
            }
        ],
        "citation_relations": (
            [{"claim_id": "M1", "span_id": "S001", "relation": "supports"}]
            if answerable
            else []
        ),
        "unanswerable_reason": None if answerable else "no evidence",
        "nearest_distractor_paper_ids": [] if answerable else ["C001"],
        "annotation_status": "silver_generated",
        "corpus_version": "corpus-v1",
        "knowledge_cutoff": "2026-07-26",
    }
    return GoldQuestion.model_validate(base)


class _Runtime:
    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: str = "single",
    ) -> object:
        del session_id
        del research_mode
        if question.endswith("1"):
            return {
                "answer": {
                    "status": "answered",
                    "claims": [{"text": "private answer", "citation_ids": ["E1"]}],
                },
                "sources": [{"chunk_id": "chk-1", "corpus_id": "C001"}],
                "generation": {"input_tokens": 10, "output_tokens": 3},
            }
        return {
            "answer": {"status": "insufficient_evidence", "claims": []},
            "sources": [],
            "generation": {"input_tokens": 5, "output_tokens": 1},
        }


class _Judge:
    async def score(self, question: GoldQuestion, answer: object, sources: object) -> RAGJudgeResult:
        del answer, sources
        return RAGJudgeResult.model_validate(
            {
                "must_have": [
                    {"claim_id": "M1", "satisfied": True, "citation_supported": True}
                ],
                "forbidden": [{"claim_id": "F1", "present": False}],
                "supported_answer_claim_count": 1,
                "citation_supported_answer_claim_count": 1,
            }
        )


class RAGGoldRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scores_answer_and_refusal_without_persisting_text(self) -> None:
        questions = (_question(1, answerable=True), _question(2, answerable=False))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_rag_gold(_Runtime(), _Judge(), questions, output)
            saved = output.read_text(encoding="utf-8")
        self.assertEqual(result["aggregates"]["claim_f1"], 1.0)
        self.assertEqual(result["aggregates"]["citation_f1"], 1.0)
        self.assertEqual(result["aggregates"]["refusal_f1"], 1.0)
        self.assertEqual(result["aggregates"]["span_recall"], 1.0)
        self.assertEqual(result["records"][1]["paper_total"], 0)
        for forbidden in ("private question", "private answer", "gold evidence", "gold claim"):
            self.assertNotIn(forbidden, saved)
        json.loads(saved)


if __name__ == "__main__":
    unittest.main()
