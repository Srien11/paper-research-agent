from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_research_agent.agent.orchestrator.models import (
    ChildTaskRequest,
    ContextMessage,
)
from paper_research_agent.web.child_executors import ConversationChildExecutor
from paper_research_agent.web.files import AttachmentStore


async def _chunks(*values: bytes):
    for value in values:
        yield value


def _request(**overrides: object) -> ChildTaskRequest:
    values: dict[str, object] = {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
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
        yield {"type": "delta", "text": "上下文回答"}
        yield {"type": "done", "metrics": {}}

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


class ConversationChildExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_chat_uses_explicit_orchestrator_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = _FakeConversationRuntime()
            executor = ConversationChildExecutor(
                runtime=runtime,
                attachments=AttachmentStore(Path(directory)),
            )

            artifact = await executor.answer(_request())

            self.assertEqual(artifact.text, "上下文回答")
            self.assertEqual(runtime.direct_request.recent_messages[0].content, "比较 RAG 与 GraphRAG")
            self.assertEqual(runtime.direct_request.active_goal, "比较 RAG 与 GraphRAG")
            self.assertEqual(runtime.direct_request.active_task, "回答当前问题")

    async def test_attachment_and_file_paths_use_session_scoped_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AttachmentStore(Path(directory))
            source = await store.save(
                session_id="conversation-1",
                filename="notes.md",
                content_type="text/markdown",
                chunks=_chunks("旧内容".encode()),
            )
            runtime = _FakeConversationRuntime()
            executor = ConversationChildExecutor(runtime=runtime, attachments=store)

            attachment = await executor.answer_attachment(
                _request(
                    capability="attachment_qa",
                    attachment_ids=(source.attachment_id,),
                )
            )
            edited = await executor.edit(
                _request(
                    capability="file_edit",
                    task_id="edit-notes",
                    attachment_ids=(source.attachment_id,),
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


if __name__ == "__main__":
    unittest.main()
