"""Unified child-graph adapters projecting ChildTaskRequest in and ChildTaskResult out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from paper_research_agent.agent.orchestrator.artifacts import (
    AttachmentArtifact,
    ChatArtifact,
    DynamicToolArtifact,
    FileArtifact,
    LocalRAGArtifact,
)
from paper_research_agent.agent.orchestrator.identifiers import dynamic_thread_id
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest, ChildTaskResult

if TYPE_CHECKING:
    from paper_research_agent.agent.dynamic.models import DynamicResearchResult
class LocalRagChildExecutor(Protocol):
    async def answer(self, request: ChildTaskRequest) -> LocalRAGArtifact: ...


class DirectChatChildExecutor(Protocol):
    async def answer(self, request: ChildTaskRequest) -> ChatArtifact: ...


class AttachmentChildExecutor(Protocol):
    async def answer_attachment(
        self, request: ChildTaskRequest
    ) -> AttachmentArtifact: ...


class FileEditChildExecutor(Protocol):
    async def edit(self, request: ChildTaskRequest) -> FileArtifact: ...


class DynamicToolsChildExecutor(Protocol):
    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        memory_context: tuple[dict[str, object], ...] = (),
        child_context: dict[str, object] | None = None,
    ) -> DynamicResearchResult: ...

    async def resume(
        self,
        *,
        thread_id: str,
        approved: bool,
    ) -> DynamicResearchResult: ...


class ChildGraphDispatcher:
    """Dispatches one child task and projects the result; never relaxes child safety."""

    def __init__(
        self,
        *,
        direct_chat: DirectChatChildExecutor | None = None,
        local_rag: LocalRagChildExecutor | None = None,
        dynamic_tools: DynamicToolsChildExecutor | None = None,
        attachment_qa: AttachmentChildExecutor | None = None,
        file_edit: FileEditChildExecutor | None = None,
    ) -> None:
        self.direct_chat = direct_chat
        self.local_rag = local_rag
        self.dynamic_tools = dynamic_tools
        self.attachment_qa = attachment_qa
        self.file_edit = file_edit

    async def dispatch(self, request: ChildTaskRequest) -> ChildTaskResult:
        if request.capability == "direct_chat":
            return await self._dispatch_direct_chat(request)
        if request.capability == "local_rag":
            return await self._dispatch_local_rag(request)
        if request.capability == "dynamic_tools":
            return await self._dispatch_dynamic_tools(request)
        if request.capability == "attachment_qa":
            return await self._dispatch_attachment(request)
        if request.capability == "file_edit":
            return await self._dispatch_file_edit(request)
        raise ValueError(f"child dispatch does not support {request.capability}")

    async def resume_dynamic_tools(
        self, request: ChildTaskRequest, *, approved: bool
    ) -> ChildTaskResult:
        if request.capability != "dynamic_tools" or self.dynamic_tools is None:
            return _failed(request, "dynamic_tools_unavailable")
        resume_task = getattr(self.dynamic_tools, "resume_task", None)
        if callable(resume_task):
            result = await resume_task(request, approved=approved)
        else:
            result = await self.dynamic_tools.resume(
                thread_id=_dynamic_thread_id(request),
                approved=approved,
            )
        return _dynamic_result(request, result)

    async def _dispatch_direct_chat(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.direct_chat is None:
            return _failed(request, "direct_chat_unavailable")
        artifact = await self.direct_chat.answer(request)
        return _completed(request, artifact=artifact, citation_kind="none")

    async def _dispatch_local_rag(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.local_rag is None:
            return _failed(request, "local_rag_unavailable")
        artifact = await self.local_rag.answer(request)
        if artifact.answer.status == "compiler_failed":
            return ChildTaskResult(
                child_run_id=request.run_id,
                task_id=request.task_id,
                capability="local_rag",
                status="failed",
                summary=artifact.text,
                source_ids=artifact.source_ids,
                citation_kind="local_paper",
                error_code="comparison_compiler_failed",
                artifact=artifact,
            )
        return ChildTaskResult(
            child_run_id=request.run_id,
            task_id=request.task_id,
            capability="local_rag",
            status=(
                "completed"
                if artifact.answer.status == "answered"
                else "insufficient_evidence"
            ),
            summary=artifact.text,
            source_ids=artifact.source_ids,
            citation_kind="local_paper",
            artifact=artifact,
        )

    async def _dispatch_dynamic_tools(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.dynamic_tools is None:
            return _failed(request, "dynamic_tools_unavailable")
        question = request.objective or request.current_message
        run_task = getattr(self.dynamic_tools, "run_task", None)
        if callable(run_task):
            result = await run_task(request)
        else:
            result = await self.dynamic_tools.run(
                question,
                thread_id=_dynamic_thread_id(request),
                memory_context=_memory_context_from_request(request),
                child_context=_child_context_from_request(request),
            )
        return _dynamic_result(request, result)

    async def _dispatch_attachment(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.attachment_qa is None:
            return _failed(request, "attachment_qa_unavailable")
        artifact = await self.attachment_qa.answer_attachment(request)
        return _completed(request, artifact=artifact, citation_kind="none")

    async def _dispatch_file_edit(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.file_edit is None:
            return _failed(request, "file_edit_unavailable")
        artifact = await self.file_edit.edit(request)
        return _completed(request, artifact=artifact, citation_kind="none")


def _memory_context_from_request(request: ChildTaskRequest) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "memory_id": item.source_id,
            "content": item.content,
            "kind": "long_term_memory",
            "trust": "research_context",
        }
        for item in request.selected_context
        if item.kind == "long_term_memory"
    )


def _child_context_from_request(request: ChildTaskRequest) -> dict[str, object]:
    return {
        "goal_id": request.goal_id,
        "goal_objective": request.goal_objective,
        "task_id": request.task_id,
        "objective": request.objective,
        "success_criteria": list(request.success_criteria),
        "constraints": list(request.constraints),
    }


def _dynamic_result(
    request: ChildTaskRequest, result: DynamicResearchResult
) -> ChildTaskResult:
    if result.status == "approval_required":
        pending = result.pending_approval
        pending_payload = pending.model_dump(mode="json") if pending is not None else None
        if pending_payload is not None:
            pending_payload["task_id"] = request.task_id
        return ChildTaskResult(
            child_run_id=result.run_id,
            task_id=request.task_id,
            capability="dynamic_tools",
            status="waiting_approval",
            summary="等待敏感工具审批",
            pending_approval=pending_payload,
            citation_kind="none",
        )
    if result.termination_reason in {"approval_denied", "approval_expired"}:
        return ChildTaskResult(
            child_run_id=result.run_id,
            task_id=request.task_id,
            capability="dynamic_tools",
            status="failed",
            summary=result.final_summary or "审批未执行",
            citation_kind="none",
            error_code=result.termination_reason,
        )
    return ChildTaskResult(
        child_run_id=result.run_id,
        task_id=request.task_id,
        capability="dynamic_tools",
        status="completed",
        summary=result.final_summary or "动态研究已完成",
        citation_kind="external",
        artifact=DynamicToolArtifact(
            text=result.final_summary or "动态研究已完成",
            tool_names=tuple(
                dict.fromkeys(item.tool_name for item in result.observations)
            ),
        ),
    )


def _completed(
    request: ChildTaskRequest,
    *,
    artifact: ChatArtifact | AttachmentArtifact | FileArtifact,
    citation_kind: Literal["none", "local_paper", "external"],
) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=request.run_id,
        task_id=request.task_id,
        capability=request.capability,
        status="completed",
        summary=artifact.text,
        source_ids=artifact.source_ids,
        citation_kind=citation_kind,
        artifact=artifact,
    )


def _failed(request: ChildTaskRequest, error_code: str) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=request.run_id,
        task_id=request.task_id,
        capability=request.capability,
        status="failed",
        error_code=error_code,
    )


def _dynamic_thread_id(request: ChildTaskRequest) -> str:
    return dynamic_thread_id(
        request.conversation_id,
        request.run_id,
        request.task_id,
    )
