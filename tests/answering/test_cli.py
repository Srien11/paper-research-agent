from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from answer import _optional_audit, run_answer

from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


class FakeGenerator:
    model_id = "qwen3.7-plus-2026-05-26"
    prompt_version = "rag-answer-json-v1"

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        del request
        return GenerationResult(
            content=(
                '{"status":"answered","claims":[{"text":"CLI结论。",'
                '"citation_ids":["E1"]}],"insufficient_reason":null}'
            ),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            attempts=1,
        )


def assembled() -> AssembledContext:
    return AssembledContext(
        messages=(
            PromptMessage(role="system", content="trusted"),
            PromptMessage(role="user", content="PRIVATE_CONTEXT_SENTINEL"),
        ),
        citations=(
            CitationRef(
                citation_id="E1",
                chunk_id="chunk-1",
                corpus_id="C001",
                asset_id="asset-1",
                page_start=1,
                page_end=1,
                text_sha256=hashlib.sha256(b"evidence").hexdigest(),
                storage_class="internal_research_only",
            ),
        ),
        estimated_tokens=100,
        token_budget=2000,
        output_reserve_tokens=1200,
        omitted_evidence_count=0,
    )


class AnswerCliTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_json_becomes_minimal_answer_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context_path = root / "context.json"
            output_path = root / "answer.json"
            context_path.write_text(assembled().model_dump_json(indent=2), encoding="utf-8")
            result = await run_answer(
                context_path,
                output_path=output_path,
                generator=FakeGenerator(),
                audit=None,
            )
            saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(result.status, "answered")
        self.assertEqual(saved["answer_markdown"], "CLI结论。[E1]")
        serialized = json.dumps(saved, ensure_ascii=False)
        self.assertNotIn("PRIVATE_CONTEXT_SENTINEL", serialized)
        self.assertNotIn("messages", saved)
        self.assertNotIn("context", saved)

    async def test_empty_context_needs_no_api_key(self) -> None:
        empty = AssembledContext(
            messages=(
                PromptMessage(role="system", content="trusted"),
                PromptMessage(role="user", content="question"),
            ),
            citations=(),
            estimated_tokens=100,
            token_budget=2000,
            output_reserve_tokens=1200,
            omitted_evidence_count=1,
            evidence_insufficient=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            context_path = Path(directory) / "empty.json"
            output_path = Path(directory) / "answer.json"
            context_path.write_text(empty.model_dump_json(), encoding="utf-8")
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False):
                result = await run_answer(context_path, output_path=output_path, audit=None)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.attempts, 0)

    def test_audit_initialization_failure_is_best_effort(self) -> None:
        with patch("answer.SQLiteAnswerAuditLogger", side_effect=sqlite3.DatabaseError):
            self.assertIsNone(_optional_audit(Path("data/runtime/unavailable.sqlite3")))


if __name__ == "__main__":
    unittest.main()
