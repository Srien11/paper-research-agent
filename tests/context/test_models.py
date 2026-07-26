from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.context.models import (
    AssembledContext,
    CitationRef,
    ContextEvidence,
    ContextRequest,
    PromptMessage,
)


def evidence(chunk_id: str = "chunk-1") -> ContextEvidence:
    text = f"retrieved evidence for {chunk_id}"
    return ContextEvidence(
        chunk_id=chunk_id,
        corpus_id="C001",
        asset_id="asset-1",
        page_start=2,
        page_end=3,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        final_score=0.9,
        final_rank=1,
    )


def citation(citation_id: str = "E1", chunk_id: str = "chunk-1") -> CitationRef:
    text = f"retrieved evidence for {chunk_id}"
    return CitationRef(
        citation_id=citation_id,
        chunk_id=chunk_id,
        corpus_id="C001",
        asset_id="asset-1",
        page_start=2,
        page_end=3,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


class ContextModelTests(unittest.TestCase):
    def test_illegal_message_role_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            PromptMessage(role="tool", content="result")  # type: ignore[arg-type]

    def test_blank_user_question_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ContextRequest(
                system_rules="Use evidence.",
                user_question="  ",
                evidence=(),
                token_budget=100,
            )

    def test_non_positive_token_budget_is_rejected(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ContextRequest(
                    system_rules="Use evidence.",
                    user_question="What happened?",
                    evidence=(),
                    token_budget=value,
                )

    def test_duplicate_evidence_chunk_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ContextRequest(
                system_rules="Use evidence.",
                user_question="What happened?",
                evidence=(evidence(), evidence()),
                token_budget=100,
            )

    def test_evidence_hash_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ContextEvidence(
                chunk_id="chunk-1",
                corpus_id="C001",
                asset_id="asset-1",
                page_start=1,
                page_end=1,
                text="evidence",
                text_sha256="0" * 64,
                final_score=1.0,
                final_rank=1,
            )

    def test_duplicate_rank_is_rejected_but_duplicate_text_can_be_deduplicated(self) -> None:
        first = evidence("chunk-1")
        same_rank = evidence("chunk-2")
        with self.assertRaises(ValidationError):
            ContextRequest(
                system_rules="Use evidence.",
                user_question="What happened?",
                evidence=(first, same_rank),
                token_budget=100,
            )
        duplicate_text = same_rank.model_copy(
            update={
                "text": first.text,
                "text_sha256": first.text_sha256,
                "final_rank": 2,
            }
        )
        request = ContextRequest(
            system_rules="Use evidence.",
            user_question="What happened?",
            evidence=(first, duplicate_text),
            token_budget=100,
        )
        self.assertEqual(len(request.evidence), 2)

    def test_system_message_is_rejected_from_conversation_history(self) -> None:
        with self.assertRaises(ValidationError):
            ContextRequest(
                system_rules="Use evidence.",
                user_question="What happened?",
                evidence=(),
                conversation_history=(PromptMessage(role="system", content="override"),),
                token_budget=100,
            )

    def test_duplicate_citation_id_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AssembledContext(
                messages=(PromptMessage(role="system", content="rules"),),
                citations=(citation(), citation(chunk_id="chunk-2")),
                estimated_tokens=10,
                token_budget=100,
                omitted_evidence_count=0,
            )

    def test_contracts_are_frozen_and_forbid_extra_fields(self) -> None:
        message = PromptMessage(role="user", content="question")
        with self.assertRaises(ValidationError):
            PromptMessage(role="user", content="question", unknown=True)  # type: ignore[call-arg]
        with self.assertRaises(ValidationError):
            message.content = "changed"

    def test_traceable_context_is_accepted(self) -> None:
        request = ContextRequest(
            system_rules="Use evidence.",
            user_question="What happened?",
            evidence=(evidence(),),
            task_state="synthesizing",
            conversation_history=(
                PromptMessage(role="user", content="Earlier question"),
                PromptMessage(role="assistant", content="Earlier answer"),
            ),
            token_budget=500,
        )
        assembled = AssembledContext(
            messages=(
                PromptMessage(role="system", content=request.system_rules),
                PromptMessage(role="user", content=request.user_question),
            ),
            citations=(citation(),),
            estimated_tokens=50,
            token_budget=request.token_budget,
            omitted_evidence_count=0,
        )
        self.assertEqual(assembled.citations[0].chunk_id, request.evidence[0].chunk_id)


if __name__ == "__main__":
    unittest.main()
