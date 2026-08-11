"""Unified dispatch, timeout, approval, and audit boundary for 18 research tools."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from paper_research_agent.agent.models import GetEvidenceInput, SearchCorpusInput
from paper_research_agent.agent.observability import AgentEvent, AgentEventSink, emit_agent_event
from paper_research_agent.agent.service import ResearchToolService
from paper_research_agent.agent.tooling.analysis import AnalysisResearchTools
from paper_research_agent.agent.tooling.approval import ApprovalManager
from paper_research_agent.agent.tooling.catalog import ExtendedToolPolicy, effective_tool_spec
from paper_research_agent.agent.tooling.content import ContentResearchTools
from paper_research_agent.agent.tooling.contracts import (
    AdjacentChunksInput,
    AnalyzeExperimentDataInput,
    CalculateInput,
    ChunkIdsInput,
    CitationGraphInput,
    CorpusInput,
    ElementLookupInput,
    ExportResearchReportInput,
    IdentifierInput,
    ManageLongTermMemoryInput,
    PaperMetadataInput,
    SaveResearchNoteInput,
    ScholarlySearchInput,
    ToolExecutionResult,
    VerifyClaimInput,
)
from paper_research_agent.agent.tooling.local import LocalResearchTools
from paper_research_agent.agent.tooling.registry import (
    RegisteredTool,
    ToolRegistrySnapshot,
    builtin_registry_snapshot,
)
from paper_research_agent.agent.tooling.scholarly import ScholarlyResearchTools
from paper_research_agent.agent.tooling.workspace import WorkspaceResearchTools


class BuiltinToolProvider:
    provider_id = "builtin"

    def __init__(
        self,
        dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolExecutionResult]],
    ) -> None:
        self._dispatch = dispatch

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult:
        del run_id
        return await self._dispatch(tool.remote_name, arguments)


class ExtendedResearchToolkit:
    def __init__(
        self,
        *,
        local: LocalResearchTools,
        content: ContentResearchTools,
        analysis: AnalysisResearchTools,
        scholarly: ScholarlyResearchTools,
        workspace: WorkspaceResearchTools,
        rag: ResearchToolService,
        event_sink: AgentEventSink | None = None,
        approvals: ApprovalManager | None = None,
        policy: ExtendedToolPolicy | None = None,
        registry: ToolRegistrySnapshot | None = None,
    ) -> None:
        self.local = local
        self.content = content
        self.analysis = analysis
        self.scholarly = scholarly
        self.workspace = workspace
        self.rag = rag
        self.event_sink = event_sink
        self.approvals = approvals
        self.policy = policy or ExtendedToolPolicy()
        self.registry = registry or builtin_registry_snapshot(BuiltinToolProvider(self._dispatch))

    def approve(self, request_id: str) -> str:
        if self.approvals is None:
            raise RuntimeError("extended toolkit approval manager is unavailable")
        return self.approvals.approve(request_id)

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> ToolExecutionResult:
        resolved_run_id = run_id or uuid.uuid4().hex
        try:
            tool, provider = self.registry.resolve(tool_name)
        except PermissionError:
            self._event(
                resolved_run_id,
                "runtime_intercepted",
                "intercepted",
                "unknown_tool",
                reason_code="tool_not_registered",
                error_type="PermissionError",
            )
            raise PermissionError(f"unknown extended research tool: {tool_name}")
        validated_arguments = self.registry.validate_arguments(tool_name, arguments)
        effective_spec = effective_tool_spec(tool.spec, validated_arguments)
        try:
            self.policy.authorize(effective_spec)
        except PermissionError:
            self._event(
                resolved_run_id,
                "runtime_intercepted",
                "intercepted",
                tool_name,
                reason_code="tool_risk_disabled",
                error_type="PermissionError",
            )
            raise
        started = time.perf_counter()
        self._event(resolved_run_id, "tool_started", "started", tool_name)
        try:
            async with asyncio.timeout(effective_spec.timeout_seconds):
                result = await provider.execute(
                    tool,
                    validated_arguments,
                    run_id=resolved_run_id,
                )
                result = result.model_copy(
                    update={"tool_name": tool.public_name, "trust": effective_spec.trust}
                )
        except TimeoutError:
            self._event(
                resolved_run_id,
                "runtime_intercepted",
                "intercepted",
                tool_name,
                duration_ms=_elapsed(started),
                reason_code="tool_timeout",
                error_type="TimeoutError",
            )
            raise TimeoutError(f"extended tool timed out: {tool_name}") from None
        except Exception as exc:
            reason_code = "tool_execution_failed"
            if tool.provider_kind == "mcp":
                reason_code = (
                    "mcp_schema_rejected"
                    if isinstance(exc, ValueError)
                    else "mcp_server_unavailable"
                )
            self._event(
                resolved_run_id,
                "tool_failed",
                "failed",
                tool_name,
                duration_ms=_elapsed(started),
                reason_code=reason_code,
                error_type=type(exc).__name__,
            )
            if _provider_unavailable(exc) and effective_spec.risk in {
                "local_read",
                "network_read",
            }:
                return ToolExecutionResult(
                    tool_name=tool.public_name,
                    status="insufficient",
                    trust=effective_spec.trust,
                    summary={"reason": "provider_unavailable"},
                )
            raise
        if result.status == "approval_required":
            self._event(
                resolved_run_id,
                "runtime_intercepted",
                "intercepted",
                tool_name,
                duration_ms=_elapsed(started),
                reason_code="approval_required",
                error_type="PermissionError",
            )
        else:
            self._event(
                resolved_run_id,
                "tool_completed",
                "succeeded",
                tool_name,
                duration_ms=_elapsed(started),
                returned_count=len(result.items),
            )
        return result

    async def _dispatch(self, name: str, request: dict[str, Any]) -> ToolExecutionResult:
        if name == "get_adjacent_chunks":
            return self.local.get_adjacent_chunks(AdjacentChunksInput.model_validate(request))
        if name == "get_paper_metadata":
            return self.local.get_paper_metadata(PaperMetadataInput.model_validate(request))
        if name == "trace_evidence_source":
            return self.local.trace_evidence_source(ChunkIdsInput.model_validate(request))
        if name == "get_paper_outline":
            return self.local.get_paper_outline(CorpusInput.model_validate(request))
        if name == "search_scholarly_sources":
            return await self.scholarly.search_scholarly_sources(
                ScholarlySearchInput.model_validate(request)
            )
        if name == "resolve_paper_identifier":
            return await self.scholarly.resolve_paper_identifier(
                IdentifierInput.model_validate(request)
            )
        if name == "get_citation_graph":
            return await self.scholarly.get_citation_graph(
                CitationGraphInput.model_validate(request)
            )
        if name == "check_paper_status":
            return await self.scholarly.check_paper_status(IdentifierInput.model_validate(request))
        if name == "extract_table":
            return self.content.extract_table(ElementLookupInput.model_validate(request))
        if name == "inspect_figure":
            return self.content.inspect_figure(ElementLookupInput.model_validate(request))
        if name == "extract_equation":
            return self.content.extract_equation(ElementLookupInput.model_validate(request))
        if name == "calculate":
            return self.analysis.calculate(CalculateInput.model_validate(request))
        if name == "analyze_experiment_data":
            return self.analysis.analyze_experiment_data(
                AnalyzeExperimentDataInput.model_validate(request)
            )
        if name == "verify_claim":
            return await self._verify_claim(VerifyClaimInput.model_validate(request))
        if name == "check_reproducibility":
            return self.analysis.check_reproducibility(CorpusInput.model_validate(request))
        if name == "save_research_note":
            return self.workspace.save_research_note(SaveResearchNoteInput.model_validate(request))
        if name == "export_research_report":
            return self.workspace.export_research_report(
                ExportResearchReportInput.model_validate(request)
            )
        if name == "manage_long_term_memory":
            return self.workspace.manage_long_term_memory(
                ManageLongTermMemoryInput.model_validate(request)
            )
        raise PermissionError(f"unimplemented extended research tool: {name}")

    async def _verify_claim(self, request: VerifyClaimInput) -> ToolExecutionResult:
        search = await self.rag.search_corpus(
            SearchCorpusInput(query=request.claim, top_k=request.top_k)
        )
        chunk_ids = tuple(hit.chunk_id for hit in search.hits)
        evidence = await self.rag.get_evidence(GetEvidenceInput(chunk_ids=chunk_ids))
        items = tuple(
            {
                "chunk_id": record.chunk_id,
                "corpus_id": record.corpus_id,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "text": record.text,
                "text_sha256": record.text_sha256,
                "assessment": "candidate_evidence",
            }
            for record in evidence.records
        )
        return ToolExecutionResult(
            tool_name="verify_claim",
            status="ok" if items else "insufficient",
            items=items,
            summary={
                "candidate_count": len(items),
                "verdict": "evidence_found_needs_synthesis" if items else "insufficient",
            },
        )

    def _event(
        self,
        run_id: str,
        event_type: Any,
        status: Any,
        name: str,
        *,
        duration_ms: float | None = None,
        reason_code: str | None = None,
        error_type: str | None = None,
        returned_count: int | None = None,
    ) -> None:
        emit_agent_event(
            self.event_sink,
            AgentEvent(
                run_id=run_id,
                occurred_at=datetime.now(UTC),
                event_type=event_type,
                status=status,
                component="tool",
                name=name,
                duration_ms=duration_ms,
                reason_code=reason_code,
                error_type=error_type,
                returned_count=returned_count,
            ),
        )


def _elapsed(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000)


def _provider_unavailable(error: Exception) -> bool:
    if isinstance(error, httpx.RequestError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code in {408, 429} or status_code >= 500
    return False
