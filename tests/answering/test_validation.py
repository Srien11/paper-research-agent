from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.models import (
    AnswerClaim,
    AnswerRequest,
    GenerationResult,
    ProviderAnswer,
)
from paper_research_agent.answering.validation import AnswerValidationError, validate_and_render
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


def citation(identifier: str, chunk_id: str, storage_class: str) -> CitationRef:
    return CitationRef(
        citation_id=identifier,
        chunk_id=chunk_id,
        corpus_id="C001" if identifier == "E1" else "T001",
        asset_id=f"asset-{identifier}",
        page_start=1,
        page_end=2,
        text_sha256=hashlib.sha256(chunk_id.encode()).hexdigest(),
        storage_class=storage_class,
    )


def request() -> AnswerRequest:
    return AnswerRequest(
        context=AssembledContext(
            messages=(
                PromptMessage(role="system", content="trusted"),
                PromptMessage(role="user", content="question and evidence"),
            ),
            citations=(
                citation("E1", "chunk-1", "internal_research_only"),
                citation("E2", "chunk-2", "redistributable"),
            ),
            estimated_tokens=100,
            token_budget=2000,
            output_reserve_tokens=1200,
            omitted_evidence_count=0,
        )
    )


def generation(content: str = "{}") -> GenerationResult:
    return GenerationResult(
        content=content,
        requested_model="qwen3.7-plus-2026-05-26",
        actual_model="qwen3.7-plus-2026-05-26",
        prompt_version="rag-answer-json-v1",
        input_tokens=100,
        output_tokens=20,
        latency_ms=12.5,
        attempts=1,
    )


class AnswerValidationTests(unittest.TestCase):
    def test_valid_claims_are_rendered_with_trusted_citation_metadata(self) -> None:
        draft = ProviderAnswer(
            status="answered",
            claims=(
                AnswerClaim(text="第一项发现。", citation_ids=("E2", "E1")),
                AnswerClaim(text="第二项发现。", citation_ids=("E1",)),
            ),
        )
        result = validate_and_render(draft, request(), generation())
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.answer_markdown, "第一项发现。[E2][E1]\n\n第二项发现。[E1]")
        self.assertEqual([item.citation_id for item in result.citations], ["E1", "E2"])
        self.assertEqual(result.citations[0].storage_class, "internal_research_only")
        serialized = result.model_dump_json()
        self.assertNotIn("question and evidence", serialized)
        self.assertNotIn("figure", serialized)

    def test_unknown_citation_fails_closed(self) -> None:
        draft = ProviderAnswer(
            status="answered",
            claims=(AnswerClaim(text="伪造事实。", citation_ids=("E99",)),),
        )
        with self.assertRaisesRegex(AnswerValidationError, "unknown citation"):
            validate_and_render(draft, request(), generation())

    def test_provider_can_report_insufficient_evidence(self) -> None:
        draft = ProviderAnswer(
            status="insufficient_evidence",
            claims=(),
            insufficient_reason="现有证据无法回答该问题。",
        )
        result = validate_and_render(draft, request(), generation())
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, ())
        self.assertEqual(
            result.answer_markdown,
            "当前检索上下文没有足够证据，无法可靠回答该问题。",
        )

    def test_provider_insufficient_reason_is_never_echoed(self) -> None:
        draft = ProviderAnswer(
            status="insufficient_evidence",
            claims=(),
            insufficient_reason="PRIVATE_EVIDENCE [E999] accuracy 99%",
        )
        result = validate_and_render(draft, request(), generation())
        self.assertNotIn("PRIVATE_EVIDENCE", result.answer_markdown)
        self.assertNotIn("E999", result.answer_markdown)


if __name__ == "__main__":
    unittest.main()
