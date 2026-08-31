from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import GoldQuestion
from paper_research_agent.evaluation.ragas_eval import (
    RagasCaseInput,
    RagasEvaluationConfig,
    evaluate_ragas_gold,
    merge_ragas_results,
    paper_id_precision_recall,
    write_ragas_report,
)


def _question(identifier: int, *, answerable: bool) -> GoldQuestion:
    quote = "private gold evidence"
    return GoldQuestion.model_validate(
        {
            "question_id": f"GQ{identifier:03d}",
            "split": "dev",
            "language": "zh",
            "task_type": "definition_scope",
            "difficulty": "medium",
            "question": f"private question {identifier}",
            "answerable": answerable,
            "must_have_claims": (
                [{"claim_id": "M1", "text": "private gold claim"}] if answerable else []
            ),
            "forbidden_claims": [],
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
                },
                *(
                    [
                        {
                            "span_id": "S002",
                            "paper_id": "C002",
                            "evidence_version_id": "asset-2",
                            "page": 2,
                            "element_id": "element-2",
                            "raw_span_start": 0,
                            "raw_span_end": len("unlinked supporting evidence"),
                            "raw_quote": "unlinked supporting evidence",
                            "span_hash": hashlib.sha256(
                                b"unlinked supporting evidence"
                            ).hexdigest(),
                            "support_role": "supporting",
                            "projected_chunk_ids": ["chk-2"],
                        }
                    ]
                    if answerable
                    else []
                ),
            ],
            "citation_relations": (
                [{"claim_id": "M1", "span_id": "S001", "relation": "supports"}]
                if answerable
                else []
            ),
            "unanswerable_reason": None if answerable else "private refusal reason",
            "nearest_distractor_paper_ids": [] if answerable else ["C001"],
            "annotation_status": "silver_generated",
            "corpus_version": "corpus-v1",
            "knowledge_cutoff": "2026-07-26",
        }
    )


class _Runtime:
    async def ask(
        self, question: str, *, session_id: str, research_mode: str = "single"
    ) -> object:
        del session_id, research_mode
        if question.endswith("1"):
            return {
                "answer": {
                    "status": "answered",
                    "answer_markdown": "private answer",
                    "claims": [{"text": "private answer claim", "citation_ids": ["E1"]}],
                },
                "sources": [
                    {
                        "citation_id": "E1",
                        "chunk_id": "chk-1",
                        "corpus_id": "C001",
                        "excerpt": "private source excerpt",
                    }
                ],
            }
        return {
            "answer": {
                "status": "insufficient_evidence",
                "answer_markdown": "private refusal",
                "claims": [],
            },
            "sources": [],
        }


class _Scorer:
    model_id = "judge-v1"
    embedding_model_id = "embedding-v1"
    ragas_version = "0.4.3"

    def __init__(self) -> None:
        self.samples: list[RagasCaseInput] = []

    async def score(self, sample: RagasCaseInput):
        self.samples.append(sample)
        return {
            "selected_context_precision": 0.8,
            "selected_context_recall": 0.7,
            "response_faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "citation_faithfulness": 1.0,
        }


class _AlwaysAnsweredRuntime:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def ask(
        self, question: str, *, session_id: str, research_mode: str = "single"
    ) -> object:
        del question, session_id, research_mode
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            return {
                "answer": {
                    "status": "answered",
                    "answer_markdown": "private concurrent answer",
                    "claims": [
                        {"text": "private concurrent claim", "citation_ids": ["E1"]}
                    ],
                },
                "sources": [
                    {
                        "citation_id": "E1",
                        "chunk_id": "chk-1",
                        "corpus_id": "C001",
                        "excerpt": "private short excerpt",
                    }
                ],
            }
        finally:
            self.active -= 1


class _ConcurrencyScorer(_Scorer):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.started = 0
        self.first_batch_ready = asyncio.Event()

    async def score(self, sample: RagasCaseInput):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started += 1
        try:
            if self.started <= 4:
                if self.started == 4:
                    self.first_batch_ready.set()
                await asyncio.wait_for(self.first_batch_ready.wait(), timeout=1)
            return await super().score(sample)
        finally:
            self.active -= 1


class RagasEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_config_rejects_missing_embedding_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedding model"):
            RagasEvaluationConfig(
                judge_model="judge-v1",
                embedding_model=None,
                enabled_metrics=("answer_relevancy",),
            )
        with self.assertRaisesRegex(ValueError, "less than or equal to 4"):
            RagasEvaluationConfig(
                judge_model="judge-v1",
                embedding_model="embedding-v1",
                max_concurrency=5,
            )

    def test_paper_id_metrics_deduplicate_ids(self) -> None:
        precision, recall = paper_id_precision_recall(
            ["C001", "C001", "C002"], ["C001", "C003"]
        )
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)
        self.assertEqual(paper_id_precision_recall([], ["C001"]), (None, 0.0))

    async def test_runner_scores_answers_and_never_persists_private_bodies(self) -> None:
        scorer = _Scorer()
        questions = (_question(1, answerable=True), _question(2, answerable=False))
        full_chunk = "x" * 400 + " private support after excerpt boundary"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            report = Path(directory) / "report.md"
            merged_output = Path(directory) / "merged.json"
            result = await evaluate_ragas_gold(
                _Runtime(),
                scorer,
                questions,
                output,
                chunk_text_by_id={"chk-1": full_chunk},
            )
            write_ragas_report(
                result,
                report,
                thresholds={"response_faithfulness": 0.9},
            )
            saved = output.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")
            merged = merge_ragas_results(
                (
                    {
                        **result,
                        "fingerprint_sha256": "a" * 64,
                        "records": [result["records"][0]],
                    },
                    {
                        **result,
                        "fingerprint_sha256": "b" * 64,
                        "records": [result["records"][1]],
                    },
                ),
                merged_output,
                evaluation_context={"process_isolation_batch_size": 4},
            )
            merged_saved = merged_output.read_text(encoding="utf-8")

        self.assertEqual(len(scorer.samples), 1)
        self.assertEqual(scorer.samples[0].retrieved_contexts, (full_chunk,))
        self.assertEqual(
            scorer.samples[0].citation_claims[0].retrieved_contexts,
            (full_chunk,),
        )
        self.assertEqual(result["aggregates"]["response_faithfulness"], 0.9)
        self.assertEqual(result["aggregates"]["paper_id_recall"], 1.0)
        self.assertEqual(result["records"][0]["citation_validity"], 1.0)
        self.assertFalse(result["records"][1]["ragas_scored"])
        self.assertEqual(merged["question_count"], 2)
        self.assertEqual(merged["evaluation_context"]["process_isolation_batch_size"], 4)
        self.assertIn("达标", report_text)
        for private_text in (
            "private question",
            "private answer",
            "private gold evidence",
            "private gold claim",
            "private source excerpt",
            "private support after excerpt boundary",
            "private refusal",
        ):
            self.assertNotIn(private_text, saved)
            self.assertNotIn(private_text, report_text)
            self.assertNotIn(private_text, merged_saved)
        json.loads(saved)

    async def test_runner_scores_four_cases_concurrently_without_parallel_agent_calls(
        self,
    ) -> None:
        runtime = _AlwaysAnsweredRuntime()
        scorer = _ConcurrencyScorer()
        questions = tuple(_question(index, answerable=True) for index in range(1, 6))
        progress: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_ragas_gold(
                runtime,
                scorer,
                questions,
                output,
                chunk_text_by_id={"chk-1": "private full concurrent chunk"},
                case_concurrency=4,
                progress_callback=lambda completed, total: progress.append((completed, total)),
            )

        self.assertEqual(runtime.max_active, 1)
        self.assertEqual(scorer.max_active, 4)
        self.assertEqual(len(scorer.samples), 5)
        self.assertEqual(progress, [(4, 5), (5, 5)])
        self.assertEqual(
            [record["question_id"] for record in result["records"]],
            [f"GQ{index:03d}" for index in range(1, 6)],
        )


if __name__ == "__main__":
    unittest.main()
