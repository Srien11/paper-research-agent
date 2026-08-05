from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from paper_research_agent.agent.dynamic.memory import (
    LangChainMemoryProposer,
    MemoryProposal,
    has_explicit_memory_intent,
)
from paper_research_agent.agent.dynamic.models import ToolObservation
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult


def _citation_observation() -> ToolObservation:
    return ToolObservation(
        sequence=1,
        decision_fingerprint="a" * 64,
        tool_name="verify_claim",
        purpose="Verify the conclusion",
        result=ToolExecutionResult(
            tool_name="verify_claim",
            trust="citation_evidence",
            items=({"chunk_id": "chunk-1", "text": "Evidence"},),
        ),
    )


class MemoryProposalTests(unittest.IsolatedAsyncioTestCase):
    def test_explicit_intent_gate_is_conservative(self) -> None:
        self.assertTrue(has_explicit_memory_intent("请记住我偏好简洁回答"))
        self.assertTrue(has_explicit_memory_intent("Remember that I prefer Chinese"))
        self.assertFalse(has_explicit_memory_intent("简要解释这篇论文"))

    def test_strict_proposal_contract_rejects_unsupported_conclusion(self) -> None:
        with self.assertRaises(ValidationError):
            MemoryProposal(
                action="add",
                kind="confirmed_conclusion",
                content="Unsupported",
                rationale="Save it",
            )

    async def test_ordinary_question_returns_none_without_calling_model(self) -> None:
        model = Mock()
        structured = AsyncMock()
        model.with_structured_output.return_value = structured
        proposer = LangChainMemoryProposer(model)

        proposal = await proposer.propose("解释 RAG", (), ())

        self.assertEqual(proposal.action, "none")
        structured.ainvoke.assert_not_awaited()

    async def test_explicit_conclusion_uses_only_available_citation_ids(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = MemoryProposal(
            action="add",
            kind="confirmed_conclusion",
            content="The evidence supports the conclusion.",
            source_chunk_ids=("chunk-1",),
            rationale="The user explicitly asked to remember it.",
        )
        model.with_structured_output.return_value = structured
        proposer = LangChainMemoryProposer(model)

        proposal = await proposer.propose(
            "请记住这个确认过的结论",
            (),
            (_citation_observation(),),
        )

        self.assertEqual(proposal.source_chunk_ids, ("chunk-1",))
        messages = structured.ainvoke.await_args.args[0]
        self.assertIn("chunk-1", messages[0].content)

    async def test_update_must_target_a_recalled_memory(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = MemoryProposal(
            action="update",
            memory_id="b" * 32,
            content="New preference",
            rationale="Update it",
        )
        model.with_structured_output.return_value = structured
        proposer = LangChainMemoryProposer(model)

        with self.assertRaisesRegex(ValueError, "unrecalled"):
            await proposer.propose("更新这条记忆", (), ())


if __name__ == "__main__":
    unittest.main()
