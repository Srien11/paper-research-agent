from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from paper_research_agent.agent.orchestrator.artifacts import (
    DynamicToolArtifact,
    LocalRAGArtifact,
    LocalRAGTrace,
)
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ChildTaskResult,
    ConversationWorkspace,
    GoalState,
)
from paper_research_agent.agent.orchestrator.synthesizer import AnswerSynthesizer
from paper_research_agent.answering.models import (
    AnswerCitation,
    AnswerClaim,
    RAGAnswer,
)


def _utc() -> datetime:
    return datetime(2026, 8, 10, tzinfo=UTC)


def _context() -> AgentContextEnvelope:
    goal = GoalState(
        goal_id="a" * 32,
        objective="比较本地论文结论并核验外部状态",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )
    workspace = ConversationWorkspace(
        conversation_id="conversation-1",
        active_goal=goal,
        updated_at=_utc(),
    )
    return AgentContextEnvelope(
        conversation_id="conversation-1",
        request_id="request-1",
        turn_id="c" * 32,
        current_message="给出综合结论",
        rag_mode="preferred",
        workspace=workspace,
        prepared_at=_utc(),
    )


def _local_child() -> ChildTaskResult:
    answer = RAGAnswer(
        status="answered",
        answer_markdown="本地论文结论 [E1]。",
        claims=(AnswerClaim(text="本地论文结论。", citation_ids=("E1",)),),
        citations=(
            AnswerCitation(
                citation_id="E1",
                chunk_id="chunk-1",
                corpus_id="C001",
                asset_id="asset-1",
                page_start=1,
                page_end=1,
                text_sha256="d" * 64,
                storage_class="redistributable",
            ),
        ),
        requested_model="test-model",
        actual_model="test-model",
        prompt_version="test-prompt",
        latency_ms=0,
        attempts=1,
    )
    artifact = LocalRAGArtifact(
        text=answer.answer_markdown,
        source_ids=("chunk-1",),
        answer=answer,
        retrieval=LocalRAGTrace(
            index_id="index-v1",
            resolved_question_sha256="e" * 64,
            hit_count=1,
        ),
    )
    return ChildTaskResult(
        child_run_id="run-local",
        task_id="local",
        capability="local_rag",
        status="completed",
        summary="被压缩过的摘要",
        source_ids=("chunk-1",),
        citation_kind="local_paper",
        artifact=artifact,
    )


def _dynamic_child() -> ChildTaskResult:
    artifact = DynamicToolArtifact(
        text="外部状态已核验。",
        source_ids=("external-1",),
        tool_names=("paper_status",),
    )
    return ChildTaskResult(
        child_run_id="run-dynamic",
        task_id="dynamic",
        capability="dynamic_tools",
        status="completed",
        summary="外部摘要",
        source_ids=("external-1",),
        citation_kind="external",
        artifact=artifact,
    )


class AnswerSynthesizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_child_results_cannot_be_reported_as_completed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no child results"):
            await AnswerSynthesizer().synthesize(_context(), ())

    async def test_single_local_rag_result_is_not_rewritten(self) -> None:
        child = _local_child()
        synthesizer = AnswerSynthesizer()

        answer = await synthesizer.synthesize(_context(), (child,))

        self.assertEqual(answer.text, child.artifact.answer.answer_markdown)
        self.assertEqual(answer.source_ids, ("chunk-1",))
        self.assertEqual(answer.sections[0].source_kind, "local_paper")

    async def test_mixed_answer_keeps_provenance_separate(self) -> None:
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "sections": [
                {
                    "task_id": "local",
                    "text": "本地研究部分。",
                    "source_ids": ["chunk-1"],
                },
                {
                    "task_id": "dynamic",
                    "text": "外部核验部分。",
                    "source_ids": ["external-1"],
                },
            ]
        }
        model = Mock()
        model.with_structured_output.return_value = structured
        synthesizer = AnswerSynthesizer(model)

        answer = await synthesizer.synthesize(
            _context(), (_local_child(), _dynamic_child())
        )

        self.assertEqual(answer.sections[0].source_kind, "local_paper")
        self.assertEqual(answer.sections[1].source_kind, "external")
        self.assertEqual(answer.source_ids, ("chunk-1", "external-1"))

    async def test_synthesizer_cannot_create_new_source_ids(self) -> None:
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "sections": [
                {
                    "task_id": "local",
                    "text": "本地研究部分。",
                    "source_ids": ["invented-source"],
                },
                {
                    "task_id": "dynamic",
                    "text": "外部核验部分。",
                    "source_ids": ["external-1"],
                },
            ]
        }
        model = Mock()
        model.with_structured_output.return_value = structured
        synthesizer = AnswerSynthesizer(model)

        with self.assertRaisesRegex(ValueError, "unknown source"):
            await synthesizer.synthesize(
                _context(), (_local_child(), _dynamic_child())
            )

    async def test_result_source_ids_must_match_artifact(self) -> None:
        child = _local_child().model_copy(update={"source_ids": ("other-chunk",)})

        with self.assertRaisesRegex(ValueError, "do not match artifact"):
            await AnswerSynthesizer().synthesize(_context(), (child,))

    async def test_model_failure_uses_deterministic_sections(self) -> None:
        structured = AsyncMock()
        structured.ainvoke.side_effect = RuntimeError("provider unavailable")
        model = Mock()
        model.with_structured_output.return_value = structured
        synthesizer = AnswerSynthesizer(model)

        answer = await synthesizer.synthesize(
            _context(), (_local_child(), _dynamic_child())
        )

        self.assertIn("本地论文结论", answer.text)
        self.assertIn("外部状态已核验", answer.text)
        self.assertEqual(
            tuple(section.task_id for section in answer.sections),
            ("local", "dynamic"),
        )


if __name__ == "__main__":
    unittest.main()
