from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paper_research_agent.agent.orchestrator.models import (
    ChildTaskRequest,
    ContextMessage,
)
from paper_research_agent.answering.models import AnswerCitation, AnswerClaim, RAGAnswer
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.child_executors import (
    ConversationChildExecutor,
    RAGRuntimeChildExecutor,
)
from paper_research_agent.web.files import AttachmentStore
from paper_research_agent.web.run_event_bus import RunEventBus


async def _chunks(*values: bytes):
    for value in values:
        yield value


def _request(**overrides: object) -> ChildTaskRequest:
    values: dict[str, object] = {
        "run_id": "run-1",
        "request_id": "req_child_12345678901",
        "conversation_id": "conversation-1",
        "turn_id": "b" * 32,
        "goal_id": "a" * 32,
        "goal_objective": "比较 RAG 与 GraphRAG",
        "task_id": "task-1",
        "objective": "回答当前问题",
        "success_criteria": ("完成回答",),
        "capability": "direct_chat",
        "current_message": "继续",
        "conversation_summary": "用户正在比较两种方法",
        "constraints": (),
        "recent_messages": (
            ContextMessage(
                turn_id="turn-1",
                sequence=1,
                role="user",
                content="比较 RAG 与 GraphRAG",
            ),
        ),
        "rag_mode": "disabled",
    }
    values.update(overrides)
    return ChildTaskRequest(**values)


class _FakeConversationRuntime:
    def __init__(self) -> None:
        self.direct_request = None
        self.attachment_texts: tuple[str, ...] = ()
        self.file_texts: tuple[str, ...] = ()

    async def stream_contextual_chat(self, request):
        self.direct_request = request
        yield {"type": "delta", "text": "上下文"}
        yield {"type": "delta", "text": "回答"}
        yield {
            "type": "done",
            "metrics": {
                "elapsed_ms": 1250,
                "first_token_ms": 180,
                "input_tokens": 240,
                "output_tokens": 36,
                "total_tokens": 276,
            },
        }

    async def stream_attachment_chat(
        self, question: str, *, attachment_texts: tuple[str, ...], session_id: str
    ):
        del question, session_id
        self.attachment_texts = attachment_texts
        yield {"type": "delta", "text": "附件结论"}

    async def stream_file_edit(
        self, instruction: str, *, attachment_texts: tuple[str, ...], session_id: str
    ):
        del instruction, session_id
        self.file_texts = attachment_texts
        yield {"type": "delta", "text": "# 修改后的内容"}


class _FakeRAGRuntime:
    async def ask(self, question: str, **_kwargs: object) -> object:
        del question
        citation = AnswerCitation(
            citation_id="E1",
            chunk_id="chunk-1",
            corpus_id="C001",
            asset_id="asset-1",
            page_start=3,
            page_end=4,
            text_sha256="a" * 64,
            evidence_type="text",
            storage_class="internal_research_only",
        )
        answer = RAGAnswer(
            status="answered",
            answer_markdown="可验证结论 [E1]",
            claims=(AnswerClaim(text="可验证结论", citation_ids=("E1",)),),
            citations=(citation,),
            requested_model="test-model",
            actual_model="test-model",
            prompt_version="test-v1",
            input_tokens=10,
            output_tokens=5,
            latency_ms=8,
            attempts=1,
        )
        return SimpleNamespace(
            answer=answer,
            retrieval=SimpleNamespace(
                index_id="idx-test",
                resolved_question="测试问题",
                degraded=False,
                hits=(SimpleNamespace(),),
            ),
            context=SimpleNamespace(
                estimated_tokens=100,
                token_budget=1000,
                output_reserve_tokens=100,
            ),
            sources=(
                SimpleNamespace(
                    citation_id="E1",
                    chunk_id="chunk-1",
                    corpus_id="C001",
                    title="测试论文",
                    official_url="https://example.com/paper",
                    section_id=None,
                    page_start=3,
                    page_end=4,
                    evidence_type="text",
                    storage_class="internal_research_only",
                    excerpt="受控证据预览",
                    final_rank=1,
                ),
            ),
        )


class ConversationChildExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_rag_answer_completion_persists_safe_citations(self) -> None:
        store = InMemoryConversationStore()
        started = store.begin_agent_run(
            request_id="req_child_12345678901",
            conversation_id="conversation-1",
            user_question="测试问题",
        )
        bus = RunEventBus(store)
        executor = RAGRuntimeChildExecutor(
            _FakeRAGRuntime(),
            run_event_publisher=bus.publisher,
        )

        await executor.answer(
            _request(
                capability="local_rag",
                rag_mode="required",
                objective="测试问题",
                run_id=started.run_id,
                turn_id=started.turn_id,
            )
        )
        completed = next(
            item.to_stream_event()
            for item in store.run_events(started.request_id)
            if item.event_type == "answer_completed"
        )

        self.assertEqual(completed.detail.citations[0].citation_id, "E1")
        self.assertEqual(completed.detail.citations[0].title, "测试论文")
        self.assertEqual(completed.detail.citations[0].excerpt, "受控证据预览")
        await bus.aclose()

    async def test_provider_deltas_are_persisted_before_child_returns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = InMemoryConversationStore()
            started = store.begin_agent_run(
                request_id="req_child_12345678901",
                conversation_id="conversation-1",
                user_question="继续",
            )
            bus = RunEventBus(store)
            executor = ConversationChildExecutor(
                runtime=_FakeConversationRuntime(),
                attachments=AttachmentStore(Path(directory)),
                run_event_publisher=bus.publisher,
            )

            artifact = await executor.answer(
                _request(run_id=started.run_id, turn_id=started.turn_id)
            )
            events = [item.to_stream_event() for item in store.run_events(started.request_id)]

            self.assertEqual(artifact.text, "上下文回答")
            self.assertEqual(
                [item.type for item in events],
                ["answer_started", "answer_delta", "answer_delta", "answer_completed"],
            )
            self.assertTrue(
                all(
                    item.detail.delivery_mode == "provider_live"
                    for item in events
                )
            )
            await bus.aclose()

    async def test_direct_chat_uses_explicit_orchestrator_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = _FakeConversationRuntime()
            executor = ConversationChildExecutor(
                runtime=runtime,
                attachments=AttachmentStore(Path(directory)),
            )

            artifact = await executor.answer(_request())

            self.assertEqual(artifact.metrics.elapsed_ms, 1250)
            self.assertEqual(artifact.metrics.input_tokens, 240)
            self.assertEqual(artifact.metrics.output_tokens, 36)
            self.assertEqual(artifact.metrics.total_tokens, 276)
            self.assertEqual(artifact.text, "上下文回答")
            self.assertEqual(runtime.direct_request.recent_messages[0].content, "比较 RAG 与 GraphRAG")
            self.assertEqual(runtime.direct_request.active_goal, "比较 RAG 与 GraphRAG")
            self.assertEqual(runtime.direct_request.active_task, "回答当前问题")

    async def test_attachment_and_file_paths_use_session_scoped_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AttachmentStore(Path(directory))
            conversation_store = InMemoryConversationStore()
            started = conversation_store.begin_agent_run(
                request_id="req_child_12345678901",
                conversation_id="conversation-1",
                user_question="继续",
            )
            bus = RunEventBus(conversation_store)
            source = await store.save(
                session_id="conversation-1",
                filename="notes.md",
                content_type="text/markdown",
                chunks=_chunks("旧内容".encode()),
            )
            runtime = _FakeConversationRuntime()
            executor = ConversationChildExecutor(
                runtime=runtime,
                attachments=store,
                run_event_publisher=bus.publisher,
            )

            attachment = await executor.answer_attachment(
                _request(
                    capability="attachment_qa",
                    attachment_ids=(source.attachment_id,),
                    run_id=started.run_id,
                    turn_id=started.turn_id,
                )
            )
            edited = await executor.edit(
                _request(
                    capability="file_edit",
                    task_id="edit-notes",
                    attachment_ids=(source.attachment_id,),
                    run_id=started.run_id,
                    turn_id=started.turn_id,
                )
            )

            self.assertEqual(attachment.text, "附件结论")
            self.assertIn("旧内容", runtime.attachment_texts[0])
            self.assertIn("旧内容", runtime.file_texts[0])
            self.assertIn(
                "修改后的内容",
                store.extract("conversation-1", edited.output_attachment_ids)[0],
            )
            self.assertNotEqual(source.attachment_id, edited.output_attachment_ids[0])
            file_events = [
                item.to_stream_event()
                for item in conversation_store.run_events(started.request_id)
                if item.event_type == "file_created"
            ]
            self.assertEqual(len(file_events), 1)
            self.assertEqual(
                file_events[0].detail.output_attachment_id,
                edited.output_attachment_ids[0],
            )
            self.assertNotIn("修改后的内容", file_events[0].model_dump_json())
            await bus.aclose()


if __name__ == "__main__":
    unittest.main()
