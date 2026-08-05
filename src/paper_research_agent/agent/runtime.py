"""Runtime boundary from a completed LangGraph state to trusted local evidence."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.agent.models import (
    TERMINATION_REASONS,
    EvidenceAssessment,
    EvidenceRecord,
    ResearchActionRecord,
    ResearchObservation,
    ResearchPlan,
    StorageClass,
    TerminationReason,
)
from paper_research_agent.agent.observability import (
    AgentEvent,
    AgentEventSink,
    emit_agent_event,
    safe_fingerprint,
)
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.models import ContextEvidence

if TYPE_CHECKING:
    from paper_research_agent.agent.dynamic.models import DynamicResearchResult


class AsyncResearchGraph(Protocol):
    async def ainvoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any: ...


class ExtendedToolExecutor(Protocol):
    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> Any: ...

    def approve(self, request_id: str) -> str: ...


class DynamicToolExecutor(Protocol):
    async def run(self, question: str, *, thread_id: str) -> DynamicResearchResult: ...

    async def resume(
        self,
        *,
        thread_id: str,
        approved: bool,
    ) -> DynamicResearchResult: ...


class ResearchRuntimeResult(BaseModel):
    """Validated local result; evidence bodies never belong in ``task_state``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    question: str = Field(min_length=1)
    plan: ResearchPlan
    observations: tuple[ResearchObservation, ...]
    assessments: tuple[EvidenceAssessment, ...]
    action_history: tuple[ResearchActionRecord, ...]
    evidence: tuple[ContextEvidence, ...]
    tool_call_count: int = Field(ge=0)
    replan_count: int = Field(ge=0)
    evidence_sufficient: bool
    termination_reason: TerminationReason
    task_state: str = Field(min_length=1)


class ResearchAgentRuntime:
    """Execute one bounded graph and rejoin its output to immutable local chunks."""

    def __init__(
        self,
        *,
        graph: AsyncResearchGraph,
        chunks: Sequence[EvidenceChunk],
        storage_classes: Mapping[str, StorageClass],
        policy: ResearchRuntimePolicy | None = None,
        close: Callable[[], Awaitable[None]] | None = None,
        clear: Callable[[str], Awaitable[None]] | None = None,
        event_sink: AgentEventSink | None = None,
        extended_tools: ExtendedToolExecutor | None = None,
        dynamic_tools: DynamicToolExecutor | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("research runtime requires at least one evidence chunk")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("research runtime chunks contain duplicate IDs")
        corpus_ids = {chunk.corpus_id for chunk in chunks}
        if not corpus_ids.issubset(storage_classes):
            raise ValueError("research runtime storage rights do not cover every chunk")
        invalid_rights = {
            value
            for corpus_id, value in storage_classes.items()
            if corpus_id in corpus_ids
            and value not in {"redistributable", "internal_research_only"}
        }
        if invalid_rights:
            raise ValueError("research runtime contains an invalid storage class")

        self._graph = graph
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._storage_classes = {corpus_id: storage_classes[corpus_id] for corpus_id in corpus_ids}
        self._policy = policy or ResearchRuntimePolicy()
        self._close = close
        self._clear = clear
        self._event_sink = event_sink
        self._extended_tools = extended_tools
        self._dynamic_tools = dynamic_tools
        self._closed = False

    @property
    def policy(self) -> ResearchRuntimePolicy:
        return self._policy

    @property
    def extended_tools_enabled(self) -> bool:
        return self._extended_tools is not None

    @property
    def dynamic_tools_enabled(self) -> bool:
        return self._dynamic_tools is not None

    async def run_dynamic_tools(
        self,
        question: str,
        *,
        thread_id: str,
    ) -> DynamicResearchResult:
        if self._closed:
            raise RuntimeError("research runtime is closed")
        if self._dynamic_tools is None:
            raise RuntimeError("dynamic research tools are unavailable")
        return await self._dynamic_tools.run(question, thread_id=thread_id)

    async def resume_dynamic_tools(
        self,
        *,
        thread_id: str,
        approved: bool,
    ) -> DynamicResearchResult:
        if self._closed:
            raise RuntimeError("research runtime is closed")
        if self._dynamic_tools is None:
            raise RuntimeError("dynamic research tools are unavailable")
        return await self._dynamic_tools.resume(thread_id=thread_id, approved=approved)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("research runtime is closed")
        if self._extended_tools is None:
            raise RuntimeError("extended research tools are unavailable")
        return await self._extended_tools.execute(tool_name, arguments, run_id=run_id)

    def approve_tool_request(self, request_id: str) -> str:
        if self._closed:
            raise RuntimeError("research runtime is closed")
        if self._extended_tools is None:
            raise RuntimeError("extended research tools are unavailable")
        return self._extended_tools.approve(request_id)

    async def list_long_term_memories(
        self,
        *,
        scope_id: str = "global",
        limit: int = 20,
    ) -> ToolExecutionResult:
        result = await self.execute_tool(
            "manage_long_term_memory",
            {"action": "list", "scope_id": scope_id, "limit": limit},
        )
        return ToolExecutionResult.model_validate(result)

    async def run(self, question: str, *, thread_id: str) -> ResearchRuntimeResult:
        normalized_question = question.strip()
        normalized_thread = thread_id.strip()
        if not normalized_question:
            raise ValueError("research question cannot be blank")
        if not normalized_thread or len(normalized_thread) > 256:
            raise ValueError("thread_id must contain between 1 and 256 characters")
        if self._closed:
            raise RuntimeError("research runtime is closed")

        run_id = uuid.uuid4().hex
        question_sha256 = safe_fingerprint(normalized_question)
        thread_sha256 = safe_fingerprint(normalized_thread)
        started = time.perf_counter()
        common = {
            "run_id": run_id,
            "question_sha256": question_sha256,
            "thread_sha256": thread_sha256,
            "component": "runtime",
            "name": "research_agent",
        }
        self._emit(
            AgentEvent(
                **common,
                occurred_at=datetime.now(UTC),
                event_type="run_started",
                status="started",
                max_steps=self._policy.max_steps,
                max_tool_calls=self._policy.max_tool_calls,
                timeout_seconds=self._policy.timeout_seconds,
            )
        )
        config: dict[str, object] = {
            "configurable": {"thread_id": normalized_thread},
            "recursion_limit": max(20, self._policy.max_steps * 4 + 8),
        }
        try:
            async with asyncio.timeout(self._policy.timeout_seconds):
                raw_state = await self._graph.ainvoke(
                    {"question": normalized_question, "run_id": run_id},
                    config=config,
                )
        except TimeoutError:
            self._emit(
                AgentEvent(
                    **common,
                    occurred_at=datetime.now(UTC),
                    event_type="runtime_intercepted",
                    status="intercepted",
                    duration_ms=_elapsed_ms(started),
                    error_type="TimeoutError",
                    reason_code="total_timeout",
                    max_steps=self._policy.max_steps,
                    max_tool_calls=self._policy.max_tool_calls,
                    timeout_seconds=self._policy.timeout_seconds,
                )
            )
            raise TimeoutError("research agent exceeded its total deadline") from None
        except Exception as exc:
            self._emit(
                AgentEvent(
                    **common,
                    occurred_at=datetime.now(UTC),
                    event_type="run_failed",
                    status="failed",
                    duration_ms=_elapsed_ms(started),
                    error_type=type(exc).__name__,
                    reason_code="graph_execution_failed",
                )
            )
            raise
        try:
            if not isinstance(raw_state, Mapping):
                raise TypeError("research graph returned a non-mapping state")
            result = self._validate_result(run_id, normalized_question, raw_state)
        except (TypeError, ValueError) as exc:
            self._emit(
                AgentEvent(
                    **common,
                    occurred_at=datetime.now(UTC),
                    event_type="output_rejected",
                    status="failed",
                    duration_ms=_elapsed_ms(started),
                    error_type=type(exc).__name__,
                    reason_code="output_validation_failed",
                )
            )
            raise
        self._emit(
            AgentEvent(
                **common,
                occurred_at=datetime.now(UTC),
                event_type="run_completed",
                status="succeeded",
                duration_ms=_elapsed_ms(started),
                termination_reason=result.termination_reason,
                evidence_sufficient=result.evidence_sufficient,
                evidence_count=len(result.evidence),
                tool_call_count=result.tool_call_count,
                replan_count=result.replan_count,
            )
        )
        return result

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()

    async def clear(self, thread_id: str) -> None:
        normalized_thread = thread_id.strip()
        if not normalized_thread or len(normalized_thread) > 256:
            raise ValueError("thread_id must contain between 1 and 256 characters")
        if self._closed:
            raise RuntimeError("research runtime is closed")
        if self._clear is not None:
            await self._clear(normalized_thread)

    def _validate_result(
        self,
        run_id: str,
        question: str,
        state: Mapping[str, object],
    ) -> ResearchRuntimeResult:
        state_question = state.get("question")
        if not isinstance(state_question, str) or state_question.strip() != question:
            raise ValueError("research graph returned a mismatched question")

        plan = ResearchPlan.model_validate(state.get("plan"))
        current_step = _strict_non_negative_int(state.get("current_step"), "current_step")
        if current_step > len(plan.steps):
            raise ValueError("research graph completed an invalid number of steps")

        raw_observations = state.get("observations")
        if not isinstance(raw_observations, list):
            raise TypeError("research graph observations are missing")
        observations = tuple(
            ResearchObservation.model_validate(value) for value in raw_observations
        )
        if not observations:
            raise ValueError("research graph completed without an observation")
        if current_step != len(observations):
            raise ValueError("research observation count does not match completed steps")
        if tuple(item.step_id for item in observations) != tuple(
            step.step_id for step in plan.steps[:current_step]
        ):
            raise ValueError("research observations are not the executed plan prefix")

        raw_assessments = state.get("assessments")
        if not isinstance(raw_assessments, list):
            raise TypeError("research graph assessments are missing")
        assessments = tuple(EvidenceAssessment.model_validate(value) for value in raw_assessments)
        if len(assessments) != len(observations):
            raise ValueError("research assessments do not match observations")

        raw_actions = state.get("action_history")
        if not isinstance(raw_actions, list):
            raise TypeError("research graph action history is missing")
        action_history = tuple(ResearchActionRecord.model_validate(value) for value in raw_actions)
        if [item.sequence for item in action_history] != list(range(1, len(action_history) + 1)):
            raise ValueError("research action sequence is not contiguous")
        if not action_history or action_history[-1].action != "finish":
            raise ValueError("research action history does not end with finish")

        tool_call_count = _strict_non_negative_int(
            state.get("tool_call_count"),
            "tool_call_count",
        )
        if tool_call_count > self._policy.max_tool_calls:
            raise ValueError("research graph exceeded the runtime tool call budget")
        tool_actions = tuple(
            item for item in action_history if item.action in {"search_corpus", "get_evidence"}
        )
        if tool_call_count != len(tool_actions):
            raise ValueError("research tool action count does not match the state counter")
        search_actions = tuple(item for item in action_history if item.action == "search_corpus")
        if tuple((item.step_id, item.query) for item in search_actions) != tuple(
            (item.step_id, item.search.query) for item in observations
        ):
            raise ValueError("research search actions do not match observations")
        expected_get_actions = tuple(
            (
                item.step_id,
                tuple(hit.chunk_id for hit in item.search.hits[: self._policy.evidence_per_step]),
            )
            for item in observations
            if item.search.hits
        )
        get_actions = tuple(item for item in action_history if item.action == "get_evidence")
        if tuple((item.step_id, item.chunk_ids) for item in get_actions) != (expected_get_actions):
            raise ValueError("research evidence actions do not match observations")
        assessment_actions = tuple(
            item for item in action_history if item.action == "assess_evidence"
        )
        if tuple((item.step_id, item.outcome) for item in assessment_actions) != tuple(
            (observation.step_id, assessment.status)
            for observation, assessment in zip(observations, assessments, strict=True)
        ):
            raise ValueError("research assessment actions do not match assessments")

        replan_count = _strict_non_negative_int(state.get("replan_count"), "replan_count")
        replan_actions = tuple(item for item in action_history if item.action == "replan")
        if replan_count != len(replan_actions):
            raise ValueError("research replan count does not match action history")
        plan_by_id = {step.step_id: step for step in plan.steps}
        if any(
            item.step_id not in plan_by_id or plan_by_id[item.step_id].query != item.query
            for item in replan_actions
        ):
            raise ValueError("research replan actions do not match the final plan")

        consecutive_no_new_evidence = _strict_non_negative_int(
            state.get("consecutive_no_new_evidence"),
            "consecutive_no_new_evidence",
        )
        if consecutive_no_new_evidence > current_step:
            raise ValueError("research graph stagnation count exceeds completed steps")

        raw_sufficient = state.get("evidence_sufficient")
        if not isinstance(raw_sufficient, bool):
            raise TypeError("research graph evidence sufficiency is invalid")
        raw_termination = state.get("termination_reason")
        if not isinstance(raw_termination, str) or raw_termination not in TERMINATION_REASONS:
            raise ValueError("research graph termination reason is invalid")
        termination_reason = cast(TerminationReason, raw_termination)
        if state.get("next_action") != "finish" or state.get("active_step") is not None:
            raise ValueError("research graph did not reach a finished terminal state")
        if action_history[-1].outcome != termination_reason:
            raise ValueError("research finish action does not match termination reason")
        if termination_reason == "evidence_sufficient":
            if not raw_sufficient or not assessments[-1].evidence_sufficient:
                raise ValueError("research termination is inconsistent with sufficiency")
        elif raw_sufficient or assessments[-1].evidence_sufficient:
            raise ValueError("research termination is inconsistent with evidence sufficiency")
        if (
            termination_reason == "tool_budget"
            and self._policy.max_tool_calls - tool_call_count >= 2
        ):
            raise ValueError("tool-budget termination does not match the runtime policy")
        if termination_reason == "no_new_evidence" and consecutive_no_new_evidence < 2:
            raise ValueError("no-new-evidence termination is inconsistent with state")
        if termination_reason == "plan_exhausted" and (
            current_step != len(plan.steps) or assessments[-1].next_query is not None
        ):
            raise ValueError("plan-exhausted termination is inconsistent with state")

        raw_records = state.get("evidence_records")
        if not isinstance(raw_records, list):
            raise TypeError("research graph evidence records are missing")
        records = tuple(EvidenceRecord.model_validate(value) for value in raw_records)
        record_ids = [record.chunk_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("research graph returned duplicate evidence records")
        observed_ids: list[str] = []
        for observation in observations:
            for record in observation.evidence.records:
                if record.chunk_id not in observed_ids:
                    observed_ids.append(record.chunk_id)
        if record_ids != observed_ids:
            raise ValueError("research graph evidence merge does not match observations")

        evidence = tuple(
            self._to_context_evidence(record, rank) for rank, record in enumerate(records, start=1)
        )
        task_state = _task_state(
            plan,
            observations,
            assessments,
            action_history,
            tool_call_count,
            replan_count,
            consecutive_no_new_evidence,
            raw_sufficient,
            termination_reason,
            evidence_count=len(evidence),
        )
        return ResearchRuntimeResult(
            run_id=run_id,
            question=question,
            plan=plan,
            observations=observations,
            assessments=assessments,
            action_history=action_history,
            evidence=evidence,
            tool_call_count=tool_call_count,
            replan_count=replan_count,
            evidence_sufficient=raw_sufficient,
            termination_reason=termination_reason,
            task_state=task_state,
        )

    def _emit(self, event: AgentEvent) -> bool:
        return emit_agent_event(self._event_sink, event)

    def _to_context_evidence(
        self,
        record: EvidenceRecord,
        rank: int,
    ) -> ContextEvidence:
        chunk = self._chunks.get(record.chunk_id)
        if chunk is None:
            raise ValueError("research evidence does not exist in the immutable chunk catalog")
        storage_class = self._storage_classes[chunk.corpus_id]
        actual = (
            record.corpus_id,
            record.section_id,
            record.page_start,
            record.page_end,
            record.text,
            record.text_sha256,
            record.evidence_type,
            record.storage_class,
        )
        expected = (
            chunk.corpus_id,
            chunk.section_id,
            chunk.page_start,
            chunk.page_end,
            chunk.text,
            chunk.text_sha256,
            chunk.evidence_type,
            storage_class,
        )
        if actual != expected:
            raise ValueError("research evidence does not match its immutable chunk")
        return ContextEvidence(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            asset_id=chunk.asset_id,
            section_id=chunk.section_id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            text_sha256=chunk.text_sha256,
            evidence_type=chunk.evidence_type,
            figure=chunk.figure,
            storage_class=storage_class,
            final_score=1.0 / rank,
            final_rank=rank,
        )


def _strict_non_negative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"research graph {field} is invalid")
    return value


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000)


def _task_state(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessments: tuple[EvidenceAssessment, ...],
    action_history: tuple[ResearchActionRecord, ...],
    tool_call_count: int,
    replan_count: int,
    consecutive_no_new_evidence: int,
    evidence_sufficient: bool,
    termination_reason: TerminationReason,
    *,
    evidence_count: int,
) -> str:
    payload: dict[str, Any] = {
        "kind": "untrusted_research_task_state",
        "plan": [step.model_dump(mode="json") for step in plan.steps],
        "observations": [
            {
                "degraded": item.search.degraded,
                "degraded_reason": item.search.degraded_reason,
                "evidence_chunk_ids": [record.chunk_id for record in item.evidence.records],
                "index_id": item.search.index_id,
                "missing_chunk_ids": list(item.evidence.missing_chunk_ids),
                "objective": item.objective,
                "query": item.search.query,
                "step_id": item.step_id,
            }
            for item in observations
        ],
        "assessments": [
            {
                "evidence_sufficient": item.evidence_sufficient,
                "next_objective": item.next_objective,
                "next_query": item.next_query,
                "status": item.status,
            }
            for item in assessments
        ],
        "action_history": [item.model_dump(mode="json") for item in action_history],
        "tool_call_count": tool_call_count,
        "replan_count": replan_count,
        "consecutive_no_new_evidence": consecutive_no_new_evidence,
        "evidence_sufficient": evidence_sufficient,
        "partial_answer_allowed": bool(evidence_count) and not evidence_sufficient,
        "termination_reason": termination_reason,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
