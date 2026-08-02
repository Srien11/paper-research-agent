from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.models import (
    AnswerClaim,
    AnswerRequest,
    ProviderAnswer,
)
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


def context(*, storage_class: str | None = "internal_research_only") -> AssembledContext:
    citation = CitationRef(
        citation_id="E1",
        chunk_id="chunk-1",
        corpus_id="C001",
        asset_id="asset-1",
        page_start=1,
        page_end=1,
        text_sha256=hashlib.sha256(b"evidence").hexdigest(),
        storage_class=storage_class,
    )
    return AssembledContext(
        messages=(
            PromptMessage(role="system", content="trusted rules"),
            PromptMessage(role="user", content="untrusted question and evidence"),
        ),
        citations=(citation,),
        estimated_tokens=50,
        token_budget=2000,
        output_reserve_tokens=1200,
        omitted_evidence_count=0,
    )


class AnsweringModelTests(unittest.TestCase):
    def test_claim_requires_unique_citations_and_rejects_inline_markers(self) -> None:
        with self.assertRaises(ValidationError):
            AnswerClaim(text="事实 [E1]", citation_ids=("E1",))
        with self.assertRaises(ValidationError):
            AnswerClaim(text="事实", citation_ids=("E1", "E1"))
        with self.assertRaises(ValidationError):
            AnswerClaim(text="事实", citation_ids=())

    def test_provider_answer_state_is_consistent(self) -> None:
        claim = AnswerClaim(text="一个有证据的事实。", citation_ids=("E1",))
        answered = ProviderAnswer(status="answered", claims=(claim,))
        self.assertEqual(answered.claims, (claim,))
        insufficient = ProviderAnswer(
            status="insufficient_evidence",
            claims=(),
            insufficient_reason="现有证据不足。",
        )
        self.assertEqual(insufficient.claims, ())
        invalid = (
            {"status": "answered", "claims": ()},
            {
                "status": "answered",
                "claims": (claim,),
                "insufficient_reason": "不应存在",
            },
            {"status": "insufficient_evidence", "claims": (claim,)},
            {"status": "insufficient_evidence", "claims": ()},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ProviderAnswer(**values)  # type: ignore[arg-type]

    def test_request_requires_one_leading_system_message_and_known_rights(self) -> None:
        request = AnswerRequest(context=context())
        self.assertEqual(request.output_mode, "private_research")
        with self.assertRaises(ValidationError):
            AnswerRequest(context=context(storage_class=None))
        bad = context().model_copy(
            update={
                "messages": (
                    PromptMessage(role="user", content="question"),
                    PromptMessage(role="system", content="late rules"),
                )
            }
        )
        with self.assertRaises(ValidationError):
            AnswerRequest(context=bad)
        with self.assertRaises(ValidationError):
            AnswerRequest(context=context(), output_mode="public_export")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
