from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paper_research_agent.evaluation.dataset import DiagnosticQuery
from paper_research_agent.evaluation.end_to_end import evaluate_end_to_end


class _FakeRuntime:
    async def ask(self, question: str, *, session_id: str) -> object:
        del question, session_id
        return SimpleNamespace(
            answer=SimpleNamespace(
                status="answered",
                claims=(SimpleNamespace(),),
                citations=(SimpleNamespace(),),
            ),
            sources=(
                SimpleNamespace(corpus_id="C001", chunk_id="chunk-gold"),
                SimpleNamespace(corpus_id="C002", chunk_id="chunk-other"),
            ),
            generation=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                attempts=1,
            ),
        )


class EndToEndEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_aggregates_quality_and_omits_sensitive_text(self) -> None:
        query = DiagnosticQuery(
            query_id="Q001",
            query="sensitive research question",
            relevant_paper_ids=("C001",),
            relevant_chunk_ids=("chunk-gold",),
            answerable=True,
            annotation_status="silver_single_reviewer",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_end_to_end(
                _FakeRuntime(),
                [query],
                output,
                evaluation_context={"index_id": "idx-test", "model": "model-test"},
            )
            saved = output.read_text(encoding="utf-8")

        self.assertEqual(result["aggregates"]["run_success_rate"], 1.0)
        self.assertEqual(result["aggregates"]["answer_status_accuracy"], 1.0)
        self.assertEqual(result["aggregates"]["paper_recall"], 1.0)
        self.assertEqual(result["aggregates"]["evidence_hit_rate"], 1.0)
        self.assertEqual(result["aggregates"]["citation_structure_rate"], 1.0)
        self.assertEqual(result["answerable_query_count"], 1)
        self.assertEqual(result["unanswerable_query_count"], 0)
        self.assertEqual(result["evaluation_context"]["index_id"], "idx-test")
        self.assertNotIn("sensitive research question", saved)
        self.assertNotIn("answer_markdown", saved)
        self.assertNotIn("excerpt", saved)
        self.assertEqual(json.loads(saved)["records"][0]["source_count"], 2)

    async def test_evaluation_records_safe_error_type_and_continues(self) -> None:
        class FailingRuntime:
            async def ask(self, question: str, *, session_id: str) -> object:
                del question, session_id
                raise TimeoutError("provider payload must not be saved")

        query = DiagnosticQuery(
            query_id="Q002",
            query="private question",
            relevant_paper_ids=("C002",),
            answerable=True,
            annotation_status="silver_single_reviewer",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_end_to_end(FailingRuntime(), [query], output)
            saved = output.read_text(encoding="utf-8")

        self.assertEqual(result["aggregates"]["run_success_rate"], 0.0)
        self.assertEqual(result["records"][0]["error_type"], "TimeoutError")
        self.assertNotIn("provider payload", saved)


if __name__ == "__main__":
    unittest.main()
