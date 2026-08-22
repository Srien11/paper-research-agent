"""Web-runtime adapters implementing strict main-Agent child executor contracts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.agent.observability import safe_fingerprint
from paper_research_agent.agent.orchestrator.artifacts import (
    AttachmentArtifact,
    ChatArtifact,
    ChildExecutionMetrics,
    FileArtifact,
    LocalRAGArtifact,
    LocalRAGTrace,
)
from paper_research_agent.agent.orchestrator.identifiers import child_session_id
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest
from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.context.models import ContextLongTermMemory
from paper_research_agent.web.chat_runtime import DirectResponseRequest
from paper_research_agent.web.events import (
    AgentStreamEventDraft,
    AgentStreamEventType,
    RunNodeStatus,
    SafeRunEventDetail,
)
from paper_research_agent.web.files import AttachmentStore
from paper_research_agent.web.models import SafeEvidenceSource
from paper_research_agent.web.run_event_bus import RunEventPublisher


class RAGRuntimeLike(Protocol):
    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: Literal["single", "planned"] = "single",
        long_term_memory: tuple[ContextLongTermMemory, ...] = (),
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

    def __init__(
        self,
        runtime: RAGRuntimeLike,
        *,
        run_event_publisher: RunEventPublisher | None = None,
    ) -> None:
        self._runtime = runtime
        self._run_event_publisher = run_event_publisher

    async def answer(self, request: ChildTaskRequest) -> LocalRAGArtifact:
        long_term_memory: list[ContextLongTermMemory] = []
        for item in request.selected_context:
            if item.kind != "long_term_memory" or item.memory_kind is None:
                continue
            try:
                long_term_memory.append(
                    ContextLongTermMemory(
                        memory_id=item.source_id,
                        kind=item.memory_kind,
                        content=item.content,
                        relevance=item.relevance,
                    )
                )
            except ValidationError:
                continue
        child_started = time.perf_counter()
        result = await self._runtime.ask(
            request.objective,
            session_id=_child_session_id("research", request),
            research_mode=(
                "planned" if requires_research_planning(request.objective) else "single"
            ),
            long_term_memory=tuple(long_term_memory),
        )
        child_elapsed_ms = (time.perf_counter() - child_started) * 1_000
        answer = RAGAnswer.model_validate(_value(result, "answer"))
        retrieval = _value(result, "retrieval")
        resolved_question = str(
            _value(retrieval, "resolved_question", request.objective)
        )
        citations = answer.citations
        context = _value(result, "context", {})
        artifact = LocalRAGArtifact(
            text=answer.answer_markdown,
            source_ids=tuple(item.chunk_id for item in citations),
            answer=answer,
            retrieval=LocalRAGTrace(
                index_id=_required_text(retrieval, "index_id"),
                resolved_question_sha256=safe_fingerprint(resolved_question),
                degraded=bool(_value(retrieval, "degraded", False)),
                hit_count=len(tuple(_value(retrieval, "hits", ()))),
            ),
            metrics=ChildExecutionMetrics(
                elapsed_ms=max(0, round(child_elapsed_ms)),
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                total_tokens=answer.input_tokens + answer.output_tokens,
                estimated_context_tokens=_safe_counter(
                    _value(context, "estimated_tokens", 0)
                ),
                token_budget=_safe_counter(_value(context, "token_budget", 0)),
                output_reserve_tokens=_safe_counter(
                    _value(context, "output_reserve_tokens", 0)
                ),
            ),
        )
        if self._run_event_publisher is not None:
            safe_sources = _answer_sources(result, answer)
            await self._run_event_publisher.publish(
                _task_event(
                    request,
                    "retrieval_completed",
                    node_id=f"retrieval:{request.task_id}",
                    status="completed",
                    title="本地论文检索完成",
                    summary=f"找到 {len(citations)} 条来源",
                    detail=SafeRunEventDetail(
                        capability="local_rag",
                        delivery_mode="event_only",
                        source_count=len(citations),
                        degraded=artifact.retrieval.degraded,
                    ),
                ),
                idempotency_key=_task_event_key(request, "retrieval:completed"),
            )
            answer_publisher = _TaskAnswerPublisher(
                self._run_event_publisher,
                request,
                delivery_mode="validated_replay",
            )
            for chunk in _text_chunks(artifact.text):
                await answer_publisher.publish(chunk)
            await answer_publisher.complete(artifact.metrics, citations=safe_sources)
        return artifact


class ConversationChildExecutor:
    """Adapt explicit main-Agent context to chat, attachment, and file capabilities."""

    def __init__(
        self,
        *,
        runtime: ConversationStreamingRuntime,
        attachments: AttachmentStore,
        run_event_publisher: RunEventPublisher | None = None,
    ) -> None:
        self._runtime = runtime
        self._attachments = attachments
        self._run_event_publisher = run_event_publisher

    async def answer(self, request: ChildTaskRequest) -> ChatArtifact:
        direct_request = DirectResponseRequest(
            session_id=_child_session_id("chat", request),
            current_message=request.current_message,
            recent_messages=request.recent_messages,
            summary=request.conversation_summary,
            active_goal=request.goal_objective or None,
            active_task=request.objective,
            recalled_context=request.selected_context,
        )
        answer = _TaskAnswerPublisher(
            self._run_event_publisher, request, delivery_mode="provider_live"
        )
        text, metrics = await _collect_text(
            self._runtime.stream_contextual_chat(direct_request),
            on_delta=answer.publish,
        )
        await answer.complete(metrics)
        return ChatArtifact(text=text, metrics=metrics)

    async def answer_attachment(self, request: ChildTaskRequest) -> AttachmentArtifact:
        attachment_texts = self._attachments.extract(
            request.conversation_id,
            request.attachment_ids,
        )
        answer = _TaskAnswerPublisher(
            self._run_event_publisher, request, delivery_mode="provider_live"
        )
        text, metrics = await _collect_text(
            self._runtime.stream_attachment_chat(
                request.objective,
                attachment_texts=attachment_texts,
                session_id=_child_session_id("attachment", request),
            ),
            on_delta=answer.publish,
        )
        await answer.complete(metrics)
        return AttachmentArtifact(
            text=text,
            source_ids=request.attachment_ids,
            source_attachment_ids=request.attachment_ids,
            metrics=metrics,
        )

    async def edit(self, request: ChildTaskRequest) -> FileArtifact:
        attachment_texts = self._attachments.extract(
            request.conversation_id,
            request.attachment_ids,
        )
        answer = _TaskAnswerPublisher(
            self._run_event_publisher, request, delivery_mode="provider_live"
        )
        text, metrics = await _collect_text(
            self._runtime.stream_file_edit(
                request.objective,
                attachment_texts=attachment_texts,
                session_id=_child_session_id("file", request),
            ),
            on_delta=answer.publish,
        )
        await answer.complete(metrics)
        attachment = await self._attachments.save_generated_text(
            session_id=request.conversation_id,
            filename=f"edited-{request.task_id}.md",
            text=text,
        )
        if self._run_event_publisher is not None:
            await self._run_event_publisher.publish(
                _task_event(
                    request,
                    "file_created",
                    node_id=f"file:{request.task_id}:{attachment.attachment_id}",
                    status="completed",
                    title="文件已生成",
                    summary="修改后的文件已保存，可从附件下载",
                    detail=SafeRunEventDetail(
                        capability="file_edit",
                        delivery_mode="event_only",
                        output_attachment_id=attachment.attachment_id,
                    ),
                ),
                idempotency_key=_task_event_key(
                    request, f"file:created:{attachment.attachment_id}"
                ),
            )
        return FileArtifact(
            text="已生成修改后的文件。",
            source_ids=request.attachment_ids,
            output_attachment_ids=(attachment.attachment_id,),
            metrics=metrics,
        )


async def _collect_text(
    source: AsyncIterator[dict[str, object]],
    *,
    on_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, ChildExecutionMetrics]:
    parts: list[str] = []
    metrics = ChildExecutionMetrics()
    coalescer = _DeltaCoalescer(on_delta)
    async for event in source:
        if event.get("type") == "delta" and isinstance(event.get("text"), str):
            delta = str(event["text"])
            parts.append(delta)
            await coalescer.add(delta)
        payload = event.get("metrics")
        if event.get("type") == "done" and isinstance(payload, dict):
            metrics = ChildExecutionMetrics(
                elapsed_ms=_safe_counter(payload.get("elapsed_ms")),
                first_token_ms=_safe_counter(payload.get("first_token_ms")),
                input_tokens=_safe_counter(payload.get("input_tokens")),
                output_tokens=_safe_counter(payload.get("output_tokens")),
                total_tokens=_safe_counter(payload.get("total_tokens")),
            )
    await coalescer.close()
    text = "".join(parts).strip()
    if not text:
        raise ValueError("child chat runtime returned an empty answer")
    return text, metrics


class _DeltaCoalescer:
    def __init__(
        self,
        callback: Callable[[str], Awaitable[None]] | None,
        *,
        flush_seconds: float = 0.04,
        chunk_size: int = 192,
    ) -> None:
        self._callback = callback
        self._flush_seconds = flush_seconds
        self._chunk_size = chunk_size
        self._pending = ""
        self._first = True
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[None] | None = None

    async def add(self, delta: str) -> None:
        if self._callback is None or not delta:
            return
        if self._first:
            self._first = False
            await self._callback(delta)
            return
        flush_now = False
        async with self._lock:
            self._pending += delta
            flush_now = len(self._pending) >= self._chunk_size
            if not flush_now and self._timer is None:
                self._timer = asyncio.create_task(self._flush_later())
        if flush_now:
            await self._flush()

    async def close(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
        await self._flush()

    async def _flush_later(self) -> None:
        await asyncio.sleep(self._flush_seconds)
        await self._flush()

    async def _flush(self) -> None:
        callback = self._callback
        if callback is None:
            return
        async with self._lock:
            payload = self._pending
            self._pending = ""
            self._timer = None
        if payload:
            await callback(payload)


class _TaskAnswerPublisher:
    def __init__(
        self,
        publisher: RunEventPublisher | None,
        request: ChildTaskRequest,
        *,
        delivery_mode: Literal["provider_live", "validated_replay"],
    ) -> None:
        self._publisher = publisher
        self._request = request
        self._delivery_mode = delivery_mode
        self._started = False
        self._index = 0

    async def publish(self, delta: str) -> None:
        if self._publisher is None or not delta:
            return
        if not self._started:
            self._started = True
            await self._publisher.publish(
                _task_event(
                    self._request,
                    "answer_started",
                    node_id=f"answer:{self._request.task_id}",
                    status="running",
                    title="生成回答",
                    detail=SafeRunEventDetail(delivery_mode=self._delivery_mode),
                ),
                idempotency_key=_task_event_key(self._request, "answer:start"),
            )
        await self._publisher.publish(
            _task_event(
                self._request,
                "answer_delta",
                node_id=f"answer:{self._request.task_id}",
                status="running",
                detail=SafeRunEventDetail(delivery_mode=self._delivery_mode),
                delta=delta,
            ),
            idempotency_key=_task_event_key(
                self._request, f"answer:delta:{self._index}"
            ),
        )
        self._index += 1

    async def complete(
        self,
        metrics: ChildExecutionMetrics,
        *,
        citations: tuple[SafeEvidenceSource, ...] = (),
    ) -> None:
        if self._publisher is None or not self._started:
            return
        await self._publisher.publish(
            _task_event(
                self._request,
                "answer_completed",
                node_id=f"answer:{self._request.task_id}",
                status="completed",
                title="回答生成完成",
                detail=SafeRunEventDetail(
                    delivery_mode=self._delivery_mode,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    total_tokens=metrics.total_tokens,
                    first_token_ms=metrics.first_token_ms,
                    citations=citations,
                ),
            ),
            idempotency_key=_task_event_key(self._request, "answer:completed"),
        )


def _answer_sources(
    result: object,
    answer: RAGAnswer,
) -> tuple[SafeEvidenceSource, ...]:
    sources = tuple(
        SafeEvidenceSource.model_validate(item)
        for item in tuple(_value(result, "sources", ()))
    )
    by_citation_id = {item.citation_id: item for item in sources}
    missing = [
        citation.citation_id
        for citation in answer.citations
        if citation.citation_id not in by_citation_id
    ]
    if missing:
        raise RuntimeError(f"answer citations are missing safe source metadata: {missing}")
    return tuple(by_citation_id[item.citation_id] for item in answer.citations)


def _task_event(
    request: ChildTaskRequest,
    event_type: AgentStreamEventType,
    *,
    node_id: str,
    status: RunNodeStatus,
    title: str | None = None,
    summary: str | None = None,
    detail: SafeRunEventDetail | None = None,
    delta: str | None = None,
) -> AgentStreamEventDraft:
    return AgentStreamEventDraft(
        type=event_type,
        occurred_at=datetime.now(UTC),
        request_id=request.request_id,
        run_id=request.run_id,
        turn_id=request.turn_id,
        node_id=node_id,
        parent_node_id=f"task:{request.task_id}",
        task_id=request.task_id,
        status=status,
        title=title,
        summary=summary,
        detail=detail or SafeRunEventDetail(delivery_mode="event_only"),
        delta=delta,
    )


def _task_event_key(request: ChildTaskRequest, suffix: str) -> str:
    return f"task:{request.task_id}:attempt:{request.attempt_count}:{suffix}"


def _text_chunks(text: str, chunk_size: int = 192) -> tuple[str, ...]:
    return tuple(text[index : index + chunk_size] for index in range(0, len(text), chunk_size))


def _safe_counter(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, round(value))
    return 0


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _required_text(source: object, name: str) -> str:
    value = _value(source, name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"child runtime result is missing {name}")
    return value.strip()


def _child_session_id(kind: str, request: ChildTaskRequest) -> str:
    return child_session_id(
        kind,
        request.conversation_id,
        request.run_id,
        request.task_id,
    )
