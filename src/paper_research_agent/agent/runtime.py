"""Runtime boundary from a completed LangGraph state to trusted local evidence."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.agent.models import (
    EvidenceRecord,
    ResearchObservation,
    ResearchPlan,
    StorageClass,
)
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.models import ContextEvidence


class AsyncResearchGraph(Protocol):
    async def ainvoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any: ...


class ResearchRuntimeResult(BaseModel):
    """Validated local result; evidence bodies never belong in ``task_state``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1)
    plan: ResearchPlan
    observations: tuple[ResearchObservation, ...]
    evidence: tuple[ContextEvidence, ...]
    tool_call_count: int = Field(ge=0)
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
        self._storage_classes = {
            corpus_id: storage_classes[corpus_id] for corpus_id in corpus_ids
        }
        self._policy = policy or ResearchRuntimePolicy()
        self._close = close
        self._clear = clear
        self._closed = False

    @property
    def policy(self) -> ResearchRuntimePolicy:
        return self._policy

    async def run(self, question: str, *, thread_id: str) -> ResearchRuntimeResult:
        normalized_question = question.strip()
        normalized_thread = thread_id.strip()
        if not normalized_question:
            raise ValueError("research question cannot be blank")
        if not normalized_thread or len(normalized_thread) > 256:
            raise ValueError("thread_id must contain between 1 and 256 characters")
        if self._closed:
            raise RuntimeError("research runtime is closed")

        config: dict[str, object] = {
            "configurable": {"thread_id": normalized_thread},
            "recursion_limit": max(10, self._policy.max_steps * 2 + 4),
        }
        try:
            async with asyncio.timeout(self._policy.timeout_seconds):
                raw_state = await self._graph.ainvoke(
                    {"question": normalized_question},
                    config=config,
                )
        except TimeoutError:
            raise TimeoutError("research agent exceeded its total deadline") from None
        if not isinstance(raw_state, Mapping):
            raise TypeError("research graph returned a non-mapping state")
        state = raw_state
        return self._validate_result(normalized_question, state)

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
        question: str,
        state: Mapping[str, object],
    ) -> ResearchRuntimeResult:
        state_question = state.get("question")
        if not isinstance(state_question, str) or state_question.strip() != question:
            raise ValueError("research graph returned a mismatched question")

        plan = ResearchPlan.model_validate(state.get("plan"))
        current_step = _strict_non_negative_int(state.get("current_step"), "current_step")
        if current_step != len(plan.steps):
            raise ValueError("research graph did not complete every planned step")

        raw_observations = state.get("observations")
        if not isinstance(raw_observations, list):
            raise TypeError("research graph observations are missing")
        observations = tuple(
            ResearchObservation.model_validate(value) for value in raw_observations
        )
        if tuple(item.step_id for item in observations) != tuple(
            step.step_id for step in plan.steps
        ):
            raise ValueError("research observations do not match the plan")

        tool_call_count = _strict_non_negative_int(
            state.get("tool_call_count"),
            "tool_call_count",
        )
        if tool_call_count > self._policy.max_tool_calls:
            raise ValueError("research graph exceeded the runtime tool call budget")

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
            self._to_context_evidence(record, rank)
            for rank, record in enumerate(records, start=1)
        )
        task_state = _task_state(plan, observations, tool_call_count)
        return ResearchRuntimeResult(
            question=question,
            plan=plan,
            observations=observations,
            evidence=evidence,
            tool_call_count=tool_call_count,
            task_state=task_state,
        )

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


def _task_state(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    tool_call_count: int,
) -> str:
    payload: dict[str, Any] = {
        "kind": "untrusted_research_task_state",
        "plan": [step.model_dump(mode="json") for step in plan.steps],
        "observations": [
            {
                "degraded": item.search.degraded,
                "degraded_reason": item.search.degraded_reason,
                "evidence_chunk_ids": [
                    record.chunk_id for record in item.evidence.records
                ],
                "index_id": item.search.index_id,
                "missing_chunk_ids": list(item.evidence.missing_chunk_ids),
                "objective": item.objective,
                "query": item.search.query,
                "step_id": item.step_id,
            }
            for item in observations
        ],
        "tool_call_count": tool_call_count,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
