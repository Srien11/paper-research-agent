"""Unified child-graph adapters projecting ChildTaskRequest in and ChildTaskResult out."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest, ChildTaskResult

if TYPE_CHECKING:
    from paper_research_agent.agent.dynamic.models import DynamicResearchResult
    from paper_research_agent.agent.runtime import ResearchRuntimeResult


class LocalRagChildExecutor(Protocol):
    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        planning_required: bool = False,
    ) -> ResearchRuntimeResult: ...


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
        local_rag: LocalRagChildExecutor | None = None,
        dynamic_tools: DynamicToolsChildExecutor | None = None,
    ) -> None:
        self.local_rag = local_rag
        self.dynamic_tools = dynamic_tools

    async def dispatch(self, request: ChildTaskRequest) -> ChildTaskResult:
        if request.capability == "local_rag":
            return await self._dispatch_local_rag(request)
        if request.capability == "dynamic_tools":
            return await self._dispatch_dynamic_tools(request)
        raise ValueError(f"child dispatch does not support {request.capability}")

    async def _dispatch_local_rag(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.local_rag is None:
            return _failed(request, "local_rag_unavailable")
        result = await self.local_rag.run(
            request.objective,
            thread_id=_child_thread_id("research", request),
            planning_required=requires_research_planning(request.objective),
        )
        return ChildTaskResult(
            child_run_id=result.run_id,
            task_id=request.task_id,
            capability="local_rag",
            status="completed" if result.evidence_sufficient else "insufficient_evidence",
            summary=_local_summary(result),
            source_ids=tuple(item.chunk_id for item in result.evidence),
            citation_kind="local_paper",
        )

    async def _dispatch_dynamic_tools(self, request: ChildTaskRequest) -> ChildTaskResult:
        if self.dynamic_tools is None:
            return _failed(request, "dynamic_tools_unavailable")
        question = request.objective or request.current_message
        result = await self.dynamic_tools.run(
            question,
            thread_id=_child_thread_id("dynamic", request),
            memory_context=_memory_context_from_request(request),
            child_context=_child_context_from_request(request),
        )
        return _dynamic_result(request, result)


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
        "task_id": request.task_id,
        "objective": request.objective,
        "success_criteria": list(request.success_criteria),
        "constraints": list(request.constraints),
    }


def _local_summary(result: ResearchRuntimeResult) -> str:
    objectives = [observation.objective for observation in result.observations]
    if objectives:
        return "；".join(objectives)[:5000]
    return result.question[:5000]


def _dynamic_result(
    request: ChildTaskRequest, result: DynamicResearchResult
) -> ChildTaskResult:
    if result.status == "approval_required":
        pending = result.pending_approval
        return ChildTaskResult(
            child_run_id=result.run_id,
            task_id=request.task_id,
            capability="dynamic_tools",
            status="waiting_approval",
            summary="等待敏感工具审批",
            pending_approval=pending.model_dump(mode="json") if pending is not None else None,
            citation_kind="none",
        )
    return ChildTaskResult(
        child_run_id=result.run_id,
        task_id=request.task_id,
        capability="dynamic_tools",
        status="completed",
        summary=result.final_summary or "动态研究已完成",
        citation_kind="external",
    )


def _failed(request: ChildTaskRequest, error_code: str) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=request.run_id,
        task_id=request.task_id,
        capability=request.capability,
        status="failed",
        error_code=error_code,
    )


def _child_thread_id(kind: str, request: ChildTaskRequest) -> str:
    return f"{kind}::{request.conversation_id}::{request.run_id}::{request.task_id}"
