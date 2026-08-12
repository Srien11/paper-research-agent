"""Source-aware final answer synthesis for main-Agent child results."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from paper_research_agent.agent.orchestrator.artifacts import LocalRAGArtifact
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ChildTaskResult,
)
from paper_research_agent.agent.orchestrator.prompts import (
    ANSWER_SYNTHESIZER_PROMPT_VERSION,
    ANSWER_SYNTHESIZER_SYSTEM,
)

_MAX_MODEL_INPUT_CHARS = 24_000
_MAX_ARTIFACT_CHARS = 6_000


class _FrozenSynthesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SynthesizedSection(_FrozenSynthesisModel):
    task_id: str = Field(min_length=1, max_length=64)
    source_kind: Literal["none", "local_paper", "external"]
    text: str = Field(min_length=1, max_length=20_000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("synthesized section text must not be blank")
        return normalized

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("synthesized section source IDs must be unique")
        return values


class SynthesizedAnswer(_FrozenSynthesisModel):
    text: str = Field(min_length=1, max_length=120_000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=1_200)
    sections: tuple[SynthesizedSection, ...] = Field(min_length=1, max_length=12)


class _DraftSection(_FrozenSynthesisModel):
    task_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=8_000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)


class _SynthesisDraft(_FrozenSynthesisModel):
    sections: tuple[_DraftSection, ...] = Field(min_length=1, max_length=12)


class _UnknownSourceError(ValueError):
    pass


class AnswerSynthesizer:
    """Preserve child provenance while producing one bounded final answer."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        *,
        version: str = ANSWER_SYNTHESIZER_PROMPT_VERSION,
    ) -> None:
        self._model = (
            model.with_structured_output(_SynthesisDraft, method="function_calling")
            if model is not None
            else None
        )
        self.version = version

    async def synthesize(
        self,
        context: AgentContextEnvelope,
        child_results: tuple[ChildTaskResult, ...],
    ) -> SynthesizedAnswer:
        results = _latest_results(child_results)
        if not results:
            raise ValueError("cannot synthesize an answer with no child results")
        if len(results) == 1 and results[0].status == "completed":
            return _single_answer(results[0])
        if self._model is None:
            return _deterministic_answer(results)
        system = SystemMessage(
            content=f"{ANSWER_SYNTHESIZER_SYSTEM}\nPROMPT_VERSION={self.version}"
        )
        user = HumanMessage(content=_model_input(context, results))
        try:
            raw = await self._model.ainvoke([system, user])
        except Exception:  # noqa: BLE001 - deterministic synthesis is the fallback
            return _deterministic_answer(results)
        try:
            draft = raw if isinstance(raw, _SynthesisDraft) else _SynthesisDraft.model_validate(raw)
        except (ValidationError, TypeError):
            return _deterministic_answer(results)
        try:
            return _answer_from_draft(draft, results)
        except _UnknownSourceError:
            raise
        except ValueError:
            return _deterministic_answer(results)


def _latest_results(
    child_results: tuple[ChildTaskResult, ...],
) -> tuple[ChildTaskResult, ...]:
    ordered_task_ids: list[str] = []
    latest: dict[str, ChildTaskResult] = {}
    for result in child_results:
        if result.task_id not in latest:
            ordered_task_ids.append(result.task_id)
        latest[result.task_id] = result
    return tuple(latest[task_id] for task_id in ordered_task_ids)


def _single_answer(result: ChildTaskResult) -> SynthesizedAnswer:
    section = _section_from_result(result)
    text = section.text if result.artifact is not None else _render_sections((section,))
    return SynthesizedAnswer(
        text=text,
        source_ids=section.source_ids,
        sections=(section,),
    )


def _deterministic_answer(
    results: tuple[ChildTaskResult, ...],
) -> SynthesizedAnswer:
    if not results:
        raise ValueError("cannot synthesize an answer with no child results")
    sections = tuple(_section_from_result(result) for result in results)
    return SynthesizedAnswer(
        text=_render_sections(sections),
        source_ids=_ordered_source_ids(sections),
        sections=sections,
    )


def _answer_from_draft(
    draft: _SynthesisDraft,
    results: tuple[ChildTaskResult, ...],
) -> SynthesizedAnswer:
    by_task = {result.task_id: result for result in results}
    draft_task_ids = tuple(section.task_id for section in draft.sections)
    if len(draft_task_ids) != len(set(draft_task_ids)):
        raise ValueError("synthesis returned duplicate task sections")
    if set(draft_task_ids) != set(by_task):
        raise ValueError("synthesis did not return exactly one section per task")
    sections: list[SynthesizedSection] = []
    for draft_section in draft.sections:
        result = by_task[draft_section.task_id]
        result_source_ids = _validated_result_source_ids(result)
        allowed = set(result_source_ids)
        unknown = tuple(
            source_id
            for source_id in draft_section.source_ids
            if source_id not in allowed
        )
        if unknown:
            raise _UnknownSourceError(
                f"synthesis returned unknown source IDs for {result.task_id}: {unknown}"
            )
        if isinstance(result.artifact, LocalRAGArtifact):
            text = result.artifact.answer.answer_markdown
            source_ids = result_source_ids
        else:
            text = draft_section.text
            source_ids = draft_section.source_ids
        sections.append(
            SynthesizedSection(
                task_id=result.task_id,
                source_kind=result.citation_kind,
                text=text,
                source_ids=source_ids,
            )
        )
    section_tuple = tuple(sections)
    return SynthesizedAnswer(
        text=_render_sections(section_tuple),
        source_ids=_ordered_source_ids(section_tuple),
        sections=section_tuple,
    )


def _section_from_result(result: ChildTaskResult) -> SynthesizedSection:
    artifact = result.artifact
    source_ids = _validated_result_source_ids(result)
    if isinstance(artifact, LocalRAGArtifact):
        text = artifact.answer.answer_markdown
    elif artifact is not None and artifact.text:
        text = artifact.text
    else:
        text = result.summary or _status_text(result)
    return SynthesizedSection(
        task_id=result.task_id,
        source_kind=result.citation_kind,
        text=text,
        source_ids=source_ids,
    )


def _validated_result_source_ids(result: ChildTaskResult) -> tuple[str, ...]:
    artifact = result.artifact
    if artifact is not None and artifact.source_ids != result.source_ids:
        raise ValueError(
            f"child result source IDs do not match artifact for {result.task_id}"
        )
    return result.source_ids


def _status_text(result: ChildTaskResult) -> str:
    if result.status == "failed":
        return "任务执行失败。"
    if result.status == "insufficient_evidence":
        return "当前证据不足。"
    if result.status == "waiting_approval":
        return "等待敏感工具审批。"
    return "任务已完成。"


def _render_sections(sections: tuple[SynthesizedSection, ...]) -> str:
    if len(sections) == 1:
        section = sections[0]
        if section.task_id == "main-agent" or section.source_kind == "none":
            return section.text
        return f"[{section.source_kind}] {section.text}"
    labels = {
        "none": "任务结果",
        "local_paper": "本地论文证据",
        "external": "外部信息",
    }
    return "\n\n".join(
        f"### {labels[section.source_kind]}\n\n{section.text}" for section in sections
    )


def _ordered_source_ids(sections: tuple[SynthesizedSection, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source_id for section in sections for source_id in section.source_ids
        )
    )


def _model_input(
    context: AgentContextEnvelope,
    results: tuple[ChildTaskResult, ...],
) -> str:
    goal = context.workspace.active_goal
    criteria = (
        tuple(item.description for item in goal.acceptance_criteria)
        if goal is not None
        else ()
    )
    remaining = _MAX_MODEL_INPUT_CHARS
    artifacts: list[dict[str, object]] = []
    for result in results:
        text = _section_from_result(result).text[:_MAX_ARTIFACT_CHARS]
        text = text[:remaining]
        remaining -= len(text)
        artifacts.append(
            {
                "task_id": result.task_id,
                "source_kind": result.citation_kind,
                "allowed_source_ids": result.source_ids,
                "artifact_text": text,
            }
        )
    payload = {
        "kind": "untrusted_child_artifacts_for_synthesis",
        "current_request": context.current_message,
        "goal": goal.objective if goal is not None else context.current_message,
        "acceptance_criteria": criteria,
        "artifacts": artifacts,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
