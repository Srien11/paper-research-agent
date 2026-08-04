"""Minimal LangGraph orchestration for bounded, read-only research plans."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.agent.models import (
    GetEvidenceInput,
    GetEvidenceResult,
    ResearchObservation,
    ResearchPlan,
    SearchCorpusInput,
)
from paper_research_agent.agent.service import ResearchToolService


class ResearchPlanner(Protocol):
    async def plan(self, question: str, *, max_steps: int) -> ResearchPlan: ...


class ResearchGraphInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research question must not be blank")
        return normalized


class ResearchGraphState(TypedDict, total=False):
    question: str
    plan: dict[str, Any]
    current_step: int
    observations: list[dict[str, Any]]
    evidence_records: list[dict[str, Any]]


def build_research_graph(
    *,
    planner: ResearchPlanner,
    service: ResearchToolService,
    max_steps: int = 4,
    evidence_per_step: int = 4,
    checkpointer: Any | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile a fixed planner-executor loop over the two read-only domain tools."""
    if max_steps <= 0 or max_steps > 6:
        raise ValueError("max_steps must be between 1 and 6")
    if evidence_per_step <= 0 or evidence_per_step > 20:
        raise ValueError("evidence_per_step must be between 1 and 20")

    async def create_plan(state: ResearchGraphState) -> ResearchGraphState:
        graph_input = ResearchGraphInput(question=state["question"])
        plan = await planner.plan(graph_input.question, max_steps=max_steps)
        if len(plan.steps) > max_steps:
            raise ValueError("research plan exceeds the graph step budget")
        return {
            "question": graph_input.question,
            "plan": plan.model_dump(mode="json"),
            "current_step": 0,
            "observations": [],
            "evidence_records": [],
        }

    async def execute_step(state: ResearchGraphState) -> ResearchGraphState:
        step_index = state["current_step"]
        plan = ResearchPlan.model_validate(state["plan"])
        step = plan.steps[step_index]
        search = await service.search_corpus(
            SearchCorpusInput(query=step.query, top_k=step.top_k)
        )
        selected_ids = tuple(hit.chunk_id for hit in search.hits[:evidence_per_step])
        if selected_ids:
            evidence = await service.get_evidence(GetEvidenceInput(chunk_ids=selected_ids))
        else:
            evidence = GetEvidenceResult(records=())
        observation = ResearchObservation(
            step_id=step.step_id,
            objective=step.objective,
            search=search,
            evidence=evidence,
        )

        merged = list(state["evidence_records"])
        seen = {str(record["chunk_id"]) for record in merged}
        for record in evidence.records:
            if record.chunk_id not in seen:
                merged.append(record.model_dump(mode="json"))
                seen.add(record.chunk_id)
        return {
            "current_step": step_index + 1,
            "observations": [*state["observations"], observation.model_dump(mode="json")],
            "evidence_records": merged,
        }

    def route_after_step(state: ResearchGraphState) -> str:
        plan = ResearchPlan.model_validate(state["plan"])
        return "continue" if state["current_step"] < len(plan.steps) else "done"

    builder = StateGraph(ResearchGraphState)
    builder.add_node("plan", create_plan)
    builder.add_node("execute_step", execute_step)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute_step")
    builder.add_conditional_edges(
        "execute_step",
        route_after_step,
        {"continue": "execute_step", "done": END},
    )
    return builder.compile(checkpointer=checkpointer, name="paper_research_workflow_v1")
