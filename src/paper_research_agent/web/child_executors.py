"""Web-runtime adapters implementing strict main-Agent child executor contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.agent.observability import safe_fingerprint
from paper_research_agent.agent.orchestrator.artifacts import (
    AttachmentArtifact,
    ChatArtifact,
    FileArtifact,
    LocalRAGArtifact,
    LocalRAGTrace,
)
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest
from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.web.chat_runtime import DirectResponseRequest
from paper_research_agent.web.files import AttachmentStore


class RAGRuntimeLike(Protocol):
    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: Literal["single", "planned"] = "single",
    ) -> object: ...


class ConversationStreamingRuntime(Protocol):
    def stream_contextual_chat(
        self, request: DirectResponseRequest
    ) -> AsyncIterator[dict[str, object]]: ...

    def stream_attachment_chat(
        self,
        question: str,
        *,
        attachment_texts: tuple[str, ...],
        session_id: str,
    ) -> AsyncIterator[dict[str, object]]: ...

    def stream_file_edit(
        self,
        instruction: str,
        *,
        attachment_texts: tuple[str, ...],
        session_id: str,
    ) -> AsyncIterator[dict[str, object]]: ...


class RAGRuntimeChildExecutor:
    """Call the full RAG runtime so validated answers and citations survive dispatch."""

    def __init__(self, runtime: RAGRuntimeLike) -> None:
        self._runtime = runtime

    async def answer(self, request: ChildTaskRequest) -> LocalRAGArtifact:
        result = await self._runtime.ask(
            request.objective,
            session_id=_child_thread_id("research", request),
            research_mode=(
                "planned" if requires_research_planning(request.objective) else "single"
            ),
        )
        answer = RAGAnswer.model_validate(_value(result, "answer"))
        retrieval = _value(result, "retrieval")
        resolved_question = str(
            _value(retrieval, "resolved_question", request.objective)
        )
        citations = answer.citations
        return LocalRAGArtifact(
            text=answer.answer_markdown,
            source_ids=tuple(item.chunk_id for item in citations),
            answer=answer,
            retrieval=LocalRAGTrace(
                index_id=_required_text(retrieval, "index_id"),
                resolved_question_sha256=safe_fingerprint(resolved_question),
                degraded=bool(_value(retrieval, "degraded", False)),
                hit_count=len(tuple(_value(retrieval, "hits", ()))),
            ),
        )


class ConversationChildExecutor:
    """Adapt explicit main-Agent context to chat, attachment, and file capabilities."""

    def __init__(
        self,
        *,
        runtime: ConversationStreamingRuntime,
        attachments: AttachmentStore,
    ) -> None:
        self._runtime = runtime
        self._attachments = attachments

    async def answer(self, request: ChildTaskRequest) -> ChatArtifact:
        direct_request = DirectResponseRequest(
            session_id=_child_thread_id("chat", request),
            current_message=request.current_message,
            recent_messages=request.recent_messages,
            summary=request.conversation_summary,
            active_goal=request.goal_objective or None,
            active_task=request.objective,
            recalled_context=request.selected_context,
        )
        text = await _collect_text(self._runtime.stream_contextual_chat(direct_request))
        return ChatArtifact(text=text)

    async def answer_attachment(self, request: ChildTaskRequest) -> AttachmentArtifact:
        attachment_texts = self._attachments.extract(
            request.conversation_id,
            request.attachment_ids,
        )
        text = await _collect_text(
            self._runtime.stream_attachment_chat(
                request.objective,
                attachment_texts=attachment_texts,
                session_id=_child_thread_id("attachment", request),
            )
        )
        return AttachmentArtifact(
            text=text,
            source_ids=request.attachment_ids,
            source_attachment_ids=request.attachment_ids,
        )

    async def edit(self, request: ChildTaskRequest) -> FileArtifact:
        attachment_texts = self._attachments.extract(
            request.conversation_id,
            request.attachment_ids,
        )
        text = await _collect_text(
            self._runtime.stream_file_edit(
                request.objective,
                attachment_texts=attachment_texts,
                session_id=_child_thread_id("file", request),
            )
        )
        attachment = await self._attachments.save_generated_text(
            session_id=request.conversation_id,
            filename=f"edited-{request.task_id}.md",
            text=text,
        )
        return FileArtifact(
            text="已生成修改后的文件。",
            source_ids=request.attachment_ids,
            output_attachment_ids=(attachment.attachment_id,),
        )


async def _collect_text(source: AsyncIterator[dict[str, object]]) -> str:
    parts: list[str] = []
    async for event in source:
        if event.get("type") == "delta" and isinstance(event.get("text"), str):
            parts.append(str(event["text"]))
    text = "".join(parts).strip()
    if not text:
        raise ValueError("child chat runtime returned an empty answer")
    return text


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _required_text(source: object, name: str) -> str:
    value = _value(source, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"child runtime result is missing {name}")
    return value.strip()


def _child_thread_id(kind: str, request: ChildTaskRequest) -> str:
    return f"{kind}::{request.conversation_id}::{request.run_id}::{request.task_id}"
