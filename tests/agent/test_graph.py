from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from paper_research_agent.agent.graph import (
    _remaining_runtime_seconds,
    _supplemental_completion_reserve_seconds,
    build_research_graph,
)
from paper_research_agent.agent.models import (
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceFactRequirement,
    EvidenceFollowup,
    EvidenceLedgerCell,
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceInput,
    GetEvidenceResult,
    ResearchDimension,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusHit,
    SearchCorpusInput,
    SearchCorpusResult,
)
from paper_research_agent.agent.observability import AgentEvent
from paper_research_agent.agent.policy import ResearchRuntimePolicy


def _hit(chunk_id: str, corpus_id: str, rank: int) -> SearchCorpusHit:
    return SearchCorpusHit(
        chunk_id=chunk_id,
        corpus_id=corpus_id,
        section_id="results",
        page_start=rank,
        page_end=rank,
        text_sha256=("a" if chunk_id == "chunk-1" else "b") * 64,
        storage_class=(
            "internal_research_only" if corpus_id.startswith("C") else "redistributable"
        ),
        final_rank=rank,
    )


def _record(chunk_id: str, corpus_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        chunk_id=chunk_id,
        corpus_id=corpus_id,
        section_id="results",
        page_start=1,
        page_end=1,
        text=f"Evidence for {chunk_id}.",
        text_sha256=("a" if chunk_id == "chunk-1" else "b") * 64,
        storage_class=(
            "internal_research_only" if corpus_id.startswith("C") else "redistributable"
        ),
    )


def _assessment(
    sufficient: bool,
    *,
    next_query: str | None = None,
    next_objective: str | None = None,
    status: str | None = None,
) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_sufficient=sufficient,
        status=status or ("sufficient" if sufficient else "missing_coverage"),
        next_query=next_query,
        next_objective=next_objective,
    )


class FakePlanner:
    def __init__(self, plan: ResearchPlan):
        self.plan_value = plan
        self.calls: list[tuple[str, int, bool]] = []

    async def plan(
        self,
        question: str,
        *,
        max_steps: int,
        planning_required: bool = False,
    ) -> ResearchPlan:
        self.calls.append((question, max_steps, planning_required))
        return self.plan_value


class FakeReasoner:
    def __init__(self, *assessments: EvidenceAssessment):
        self.assessments = list(assessments)
        self.calls: list[tuple[str, ResearchPlan, tuple[ResearchObservation, ...], int]] = []

    async def assess(
        self,
        question: str,
        *,
        plan: ResearchPlan,
        observations: tuple[ResearchObservation, ...],
        remaining_steps: int,
    ) -> EvidenceAssessment:
        self.calls.append((question, plan, observations, remaining_steps))
        return self.assessments.pop(0)


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def write(self, event: AgentEvent) -> bool:
        self.events.append(event)
        return True


class ResearchGraphTests(unittest.IsolatedAsyncioTestCase):
    def test_remaining_runtime_uses_persisted_wall_clock_start(self) -> None:
        with patch(
            "paper_research_agent.agent.graph.time.time", return_value=150.0
        ):
            remaining = _remaining_runtime_seconds(
                {"started_at_epoch_seconds": 100.0},
                ResearchRuntimePolicy(timeout_seconds=180),
            )

        self.assertEqual(remaining, 130.0)

    def test_supplemental_reserve_uses_latest_compilation_duration(self) -> None:
        reserve = _supplemental_completion_reserve_seconds(
            {
                "assessment_durations_ms": [60_000.0],
                "pending_followups": [{}, {}],
                "current_step": 6,
                "step_budget": 8,
                "tool_call_count": 12,
                "tool_call_budget": 16,
            },
            ResearchRuntimePolicy(
                max_steps=8,
                max_tool_calls=16,
                timeout_seconds=180,
            ),
        )

        self.assertEqual(reserve, 97.0)

    def test_supplemental_reserve_has_safety_floor(self) -> None:
        reserve = _supplemental_completion_reserve_seconds(
            {
                "assessment_durations_ms": [10_000.0],
                "pending_followups": [{}],
                "current_step": 1,
                "step_budget": 2,
                "tool_call_count": 2,
                "tool_call_budget": 4,
            },
            ResearchRuntimePolicy(
                max_steps=2,
                max_tool_calls=4,
                timeout_seconds=180,
            ),
        )

        self.assertEqual(reserve, 45.0)

    async def test_structured_fact_queries_all_run_before_first_compiler_call(
        self,
    ) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimension = ResearchDimension(dimension_id="limits", label="Limits")
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-limits",
                target_id=target.target_id,
                dimension_id="limits",
                description=f"{target.label} limits",
                fact_requirements=(
                    EvidenceFactRequirement(
                        fact_requirement_id=f"{target.target_id}-reasoning-failure",
                        description="Long-context reasoning failure",
                        protected_anchors=("长上下文", "推理", "失败"),
                        retrieval_expansions=("long-context", "reasoning", "failure"),
                    ),
                    EvidenceFactRequirement(
                        fact_requirement_id=f"{target.target_id}-mitigation",
                        description="Mitigation",
                        protected_anchors=("缓解方法",),
                        retrieval_expansions=("mitigation",),
                    ),
                ),
            )
            for target in targets
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=(dimension,),
                requirements=requirements,
                steps=tuple(
                    ResearchStep(
                        step_id=f"{target.target_id}-limits",
                        objective=f"Discover {target.label} limits",
                        query=f"broad {target.label} query",
                        corpus_id=target.corpus_id,
                        target_ids=(target.target_id,),
                        dimension_ids=("limits",),
                    )
                    for target in targets
                ),
            )
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=False,
                status="compiler_failed",
                coverage=tuple(
                    EvidenceCoverage(
                        requirement_id=requirement.requirement_id,
                        covered=False,
                    )
                    for requirement in requirements
                ),
            )
        )
        query_scope = {
            "Paper A 长上下文 推理 失败 long-context reasoning failure": "C001",
            "Paper A 缓解方法 mitigation": "C001",
            "Paper B 长上下文 推理 失败 long-context reasoning failure": "T001",
            "Paper B 缓解方法 mitigation": "T001",
        }
        service = AsyncMock()

        async def search(request: SearchCorpusInput) -> SearchCorpusResult:
            corpus_id = query_scope[request.query]
            self.assertEqual(request.corpus_id, corpus_id)
            prefix = f"{corpus_id}-{len(service.search_corpus.await_args_list)}"
            return SearchCorpusResult(
                query=request.query,
                corpus_id=corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=tuple(
                    _hit(f"{prefix}-{rank}", corpus_id, rank) for rank in range(1, 6)
                ),
            )

        async def hydrate(request: GetEvidenceInput) -> GetEvidenceResult:
            self.assertEqual(len(request.chunk_ids), 4)
            records = tuple(
                _record(
                    chunk_id,
                    "C001" if chunk_id.startswith("C001") else "T001",
                )
                for chunk_id in request.chunk_ids
            )
            return GetEvidenceResult(records=records)

        service.search_corpus.side_effect = search
        service.get_evidence.side_effect = hydrate
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=8, max_tool_calls=16),
        )

        state = await graph.ainvoke({"question": "比较长上下文推理失败及缓解方法"})

        self.assertEqual(service.search_corpus.await_count, 4)
        self.assertEqual(service.get_evidence.await_count, 4)
        self.assertEqual(len(reasoner.calls), 1)
        self.assertEqual(len(reasoner.calls[0][2]), 4)
        executed_plan = reasoner.calls[0][1]
        self.assertEqual(
            [step.fact_requirement_id for step in executed_plan.steps],
            [
                "a-reasoning-failure",
                "a-mitigation",
                "b-reasoning-failure",
                "b-mitigation",
            ],
        )
        self.assertTrue(
            all("broad" not in observation.search.query for observation in reasoner.calls[0][2])
        )
        self.assertEqual(state["assessment_observation_counts"], [4])
        self.assertEqual(state["termination_reason"], "compiler_failed")

    async def test_comparison_executes_each_planned_target_before_dynamic_replan(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=(
                    ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
                    ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
                ),
                dimensions=(ResearchDimension(dimension_id="method", label="Method"),),
                requirements=(
                    EvidenceRequirement(
                        requirement_id="a-method",
                        target_id="a",
                        dimension_id="method",
                        description="Paper A method",
                    ),
                    EvidenceRequirement(
                        requirement_id="b-method",
                        target_id="b",
                        dimension_id="method",
                        description="Paper B method",
                    ),
                ),
                steps=(
                    ResearchStep(
                        step_id="a",
                        objective="Find A",
                        query="shared method query",
                        top_k=1,
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method",),
                    ),
                    ResearchStep(
                        step_id="b",
                        objective="Find B",
                        query="shared method query",
                        top_k=1,
                        corpus_id="T001",
                        target_ids=("b",),
                        dimension_ids=("method",),
                    ),
                ),
            )
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                coverage=(
                    EvidenceCoverage(
                        requirement_id="a-method", covered=True, chunk_ids=("chunk-1",)
                    ),
                    EvidenceCoverage(
                        requirement_id="b-method", covered=True, chunk_ids=("chunk-2",)
                    ),
                ),
            ),
        )
        service = AsyncMock()
        service.search_corpus.side_effect = (
            SearchCorpusResult(
                query="shared method query",
                corpus_id="C001",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-1", "C001", 1),),
            ),
            SearchCorpusResult(
                query="shared method query",
                corpus_id="T001",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-2", "T001", 1),),
            ),
        )
        service.get_evidence.side_effect = (
            GetEvidenceResult(records=(_record("chunk-1", "C001"),)),
            GetEvidenceResult(records=(_record("chunk-2", "T001"),)),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
        )

        state = await graph.ainvoke({"question": "Compare Paper A and Paper B"})

        self.assertEqual(
            [item["search"]["query"] for item in state["observations"]],
            ["shared method query", "shared method query"],
        )
        self.assertEqual(state["replan_count"], 0)
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(
            [call.args[0].corpus_id for call in service.search_corpus.await_args_list],
            ["C001", "T001"],
        )
        self.assertEqual(
            [call.args[0].top_k for call in service.search_corpus.await_args_list],
            [20, 20],
        )
        self.assertEqual(len(reasoner.calls), 1)
        self.assertEqual(len(reasoner.calls[0][2]), 2)

    async def test_comparison_executes_six_cell_grid_before_one_assessment(self) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimensions = tuple(
            ResearchDimension(dimension_id=value, label=value.title())
            for value in ("method", "metric", "dataset")
        )
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-{dimension.dimension_id}",
                target_id=target.target_id,
                dimension_id=dimension.dimension_id,
                description=f"{target.label} {dimension.label}",
            )
            for target in targets
            for dimension in dimensions
        )
        steps = tuple(
            ResearchStep(
                step_id=requirement.requirement_id,
                objective=requirement.description,
                query=f"query {requirement.requirement_id}",
                corpus_id=("C001" if requirement.target_id == "a" else "T001"),
                target_ids=(requirement.target_id,),
                dimension_ids=(requirement.dimension_id,),
            )
            for requirement in requirements
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=dimensions,
                requirements=requirements,
                steps=steps,
            )
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                coverage=tuple(
                    EvidenceCoverage(
                        requirement_id=requirement.requirement_id,
                        covered=True,
                        chunk_ids=(f"chunk-{requirement.requirement_id}",),
                    )
                    for requirement in requirements
                ),
            )
        )
        service = AsyncMock()
        active_searches = 0
        max_active_searches = 0
        search_timeline: list[tuple[str, str]] = []
        step_by_query = {step.query: step for step in steps}

        async def tracked_search(request: SearchCorpusInput) -> SearchCorpusResult:
            nonlocal active_searches, max_active_searches
            step = step_by_query[request.query]
            active_searches += 1
            max_active_searches = max(max_active_searches, active_searches)
            search_timeline.append(("start", step.step_id))
            await asyncio.sleep(0.01)
            search_timeline.append(("finish", step.step_id))
            active_searches -= 1
            return SearchCorpusResult(
                query=step.query,
                corpus_id=step.corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=(_hit(f"chunk-{step.step_id}", str(step.corpus_id), 1),),
            )

        service.search_corpus.side_effect = tracked_search
        service.get_evidence.side_effect = tuple(
            GetEvidenceResult(
                records=(_record(f"chunk-{step.step_id}", str(step.corpus_id)),)
            )
            for step in steps
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(),
        )

        state = await graph.ainvoke({"question": "Compare three dimensions"})

        self.assertEqual(planner.calls, [("Compare three dimensions", 20, False)])
        self.assertEqual(state["current_step"], 6)
        self.assertEqual(state["tool_call_count"], 12)
        self.assertEqual(service.search_corpus.await_count, 6)
        self.assertEqual(service.get_evidence.await_count, 6)
        self.assertEqual(max_active_searches, 2)
        self.assertEqual(
            [item["step_id"] for item in state["observations"]],
            [step.step_id for step in steps],
        )
        self.assertEqual(
            [
                item["step_id"]
                for item in state["action_history"]
                if item["action"] == "search_corpus"
            ],
            [step.step_id for step in steps],
        )
        for offset in range(0, len(search_timeline), 4):
            self.assertEqual(
                [event for event, _step_id in search_timeline[offset : offset + 4]],
                ["start", "start", "finish", "finish"],
            )
        self.assertEqual(len(reasoner.calls), 1)
        self.assertEqual(len(reasoner.calls[0][2]), 6)
        self.assertTrue(state["evidence_sufficient"])

    async def test_comparison_concurrency_one_keeps_searches_serial(self) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimension = ResearchDimension(dimension_id="method", label="Method")
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-method",
                target_id=target.target_id,
                dimension_id="method",
                description=f"{target.label} method",
            )
            for target in targets
        )
        steps = tuple(
            ResearchStep(
                step_id=requirement.requirement_id,
                objective=requirement.description,
                query=f"query {requirement.target_id}",
                corpus_id=("C001" if requirement.target_id == "a" else "T001"),
                target_ids=(requirement.target_id,),
                dimension_ids=("method",),
            )
            for requirement in requirements
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=(dimension,),
                requirements=requirements,
                steps=steps,
            )
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                coverage=tuple(
                    EvidenceCoverage(
                        requirement_id=requirement.requirement_id,
                        covered=True,
                        chunk_ids=(f"chunk-{requirement.target_id}",),
                    )
                    for requirement in requirements
                ),
            )
        )
        service = AsyncMock()
        active_searches = 0
        max_active_searches = 0
        step_by_query = {step.query: step for step in steps}

        async def tracked_search(request: SearchCorpusInput) -> SearchCorpusResult:
            nonlocal active_searches, max_active_searches
            step = step_by_query[request.query]
            active_searches += 1
            max_active_searches = max(max_active_searches, active_searches)
            await asyncio.sleep(0)
            active_searches -= 1
            return SearchCorpusResult(
                query=step.query,
                corpus_id=step.corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=(_hit(f"chunk-{step.target_ids[0]}", str(step.corpus_id), 1),),
            )

        service.search_corpus.side_effect = tracked_search
        service.get_evidence.side_effect = tuple(
            GetEvidenceResult(
                records=(_record(f"chunk-{target.target_id}", str(target.corpus_id)),)
            )
            for target in targets
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(
                max_steps=2,
                max_tool_calls=4,
                comparison_search_concurrency=1,
            ),
        )

        state = await graph.ainvoke({"question": "Compare serially"})

        self.assertEqual(max_active_searches, 1)
        self.assertEqual(service.search_corpus.await_count, 2)
        self.assertEqual(
            [item["step_id"] for item in state["observations"]],
            [step.step_id for step in steps],
        )

    async def test_comparison_freezes_budget_from_ten_cell_grid(self) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimensions = tuple(
            ResearchDimension(dimension_id=f"dim{index}", label=f"Dimension {index}")
            for index in range(1, 6)
        )
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-{dimension.dimension_id}",
                target_id=target.target_id,
                dimension_id=dimension.dimension_id,
                description=f"{target.label} {dimension.label}",
            )
            for target in targets
            for dimension in dimensions
        )
        steps = tuple(
            ResearchStep(
                step_id=item.requirement_id,
                objective=item.description,
                query=f"query {item.requirement_id}",
                corpus_id=("C001" if item.target_id == "a" else "T001"),
                target_ids=(item.target_id,),
                dimension_ids=(item.dimension_id,),
            )
            for item in requirements
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=dimensions,
                requirements=requirements,
                steps=steps,
            )
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                coverage=tuple(
                    EvidenceCoverage(
                        requirement_id=item.requirement_id,
                        covered=True,
                        chunk_ids=(f"chunk-{item.requirement_id}",),
                    )
                    for item in requirements
                ),
            )
        )
        service = AsyncMock()
        service.search_corpus.side_effect = tuple(
            SearchCorpusResult(
                query=step.query,
                corpus_id=step.corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=(_hit(f"chunk-{step.step_id}", str(step.corpus_id), 1),),
            )
            for step in steps
        )
        service.get_evidence.side_effect = tuple(
            GetEvidenceResult(
                records=(_record(f"chunk-{step.step_id}", str(step.corpus_id)),)
            )
            for step in steps
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(),
        )

        state = await graph.ainvoke({"question": "Compare five dimensions"})

        self.assertEqual(planner.calls, [("Compare five dimensions", 20, False)])
        self.assertEqual(state["step_budget"], 13)
        self.assertEqual(state["tool_call_budget"], 26)
        self.assertEqual(state["current_step"], 10)
        self.assertEqual(state["tool_call_count"], 20)

    async def test_comparison_followups_stop_at_frozen_dynamic_budget(self) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimension = ResearchDimension(dimension_id="method", label="Method")
        requirements = (
            EvidenceRequirement(
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                description="Paper A method",
            ),
            EvidenceRequirement(
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                description="Paper B method",
            ),
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=(dimension,),
                requirements=requirements,
                steps=(
                    ResearchStep(
                        step_id="a-method",
                        objective="Find A method",
                        query="initial A",
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method",),
                    ),
                    ResearchStep(
                        step_id="b-method",
                        objective="Find B method",
                        query="initial B",
                        corpus_id="T001",
                        target_ids=("b",),
                        dimension_ids=("method",),
                    ),
                ),
            )
        )

        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=(
                    EvidenceCoverage(requirement_id="a-method", covered=False),
                    EvidenceCoverage(requirement_id="b-method", covered=False),
                ),
                followups=(
                    EvidenceFollowup(
                        requirement_id="a-method",
                        query="followup A",
                        objective="Retry A method",
                    ),
                    EvidenceFollowup(
                        requirement_id="b-method",
                        query="followup B",
                        objective="Retry B method",
                    ),
                ),
            ),
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=(
                    EvidenceCoverage(requirement_id="a-method", covered=False),
                    EvidenceCoverage(requirement_id="b-method", covered=False),
                ),
                next_query="followup B three",
                next_objective="Retry B method again",
                next_requirement_ids=("b-method",),
            ),
        )
        queries = ("initial A", "initial B", "followup A")
        corpora = ("C001", "T001", "C001")
        chunk_ids = (
            "chunk-a-initial",
            "chunk-b-initial",
            "chunk-a-followup",
        )
        service = AsyncMock()
        service.search_corpus.side_effect = tuple(
            SearchCorpusResult(
                query=query,
                corpus_id=corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=(_hit(chunk_id, corpus_id, 1),),
            )
            for query, corpus_id, chunk_id in zip(
                queries, corpora, chunk_ids, strict=True
            )
        )
        service.get_evidence.side_effect = tuple(
            GetEvidenceResult(records=(_record(chunk_id, corpus_id),))
            for chunk_id, corpus_id in zip(chunk_ids, corpora, strict=True)
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=3, max_tool_calls=6),
        )

        state = await graph.ainvoke({"question": "Compare methods"})

        self.assertEqual(state["step_budget"], 3)
        self.assertEqual(state["tool_call_budget"], 6)
        self.assertEqual(state["current_step"], 3)
        self.assertEqual(state["tool_call_count"], 6)
        self.assertEqual(state["termination_reason"], "step_budget")
        self.assertEqual(service.search_corpus.await_count, 3)
        self.assertEqual(len(reasoner.calls), 2)
        self.assertEqual([len(call[2]) for call in reasoner.calls], [2, 3])
        self.assertEqual(state["assessment_observation_counts"], [2, 3])

    async def test_comparison_executes_fact_bound_atomic_followups_at_fixed_top_four(
        self,
    ) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimension = ResearchDimension(dimension_id="method", label="Method")
        requirements = (
            EvidenceRequirement(
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                description="Paper A method",
                fact_requirements=(
                    EvidenceFactRequirement(
                        fact_requirement_id="a-method-mechanism",
                        description="Explain the mechanism",
                        search_query="Paper A mechanism retrieval terms",
                    ),
                    EvidenceFactRequirement(
                        fact_requirement_id="a-method-input",
                        description="Identify the input dependency",
                        search_query="Paper A input retrieval terms",
                    ),
                ),
            ),
            EvidenceRequirement(
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                description="Paper B method",
            ),
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=(dimension,),
                requirements=requirements,
                steps=(
                    ResearchStep(
                        step_id="a-method",
                        objective="Discover Paper A method",
                        query="initial A discovery",
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method",),
                    ),
                    ResearchStep(
                        step_id="b-method",
                        objective="Discover Paper B method",
                        query="initial B discovery",
                        corpus_id="T001",
                        target_ids=("b",),
                        dimension_ids=("method",),
                    ),
                ),
            )
        )
        missing_coverage = (
            EvidenceCoverage(requirement_id="a-method", covered=False),
            EvidenceCoverage(requirement_id="b-method", covered=False),
        )
        missing_ledger = (
            EvidenceLedgerCell(
                requirement_id="a-method",
                status="missing",
                missing_fact_requirement_ids=(
                    "a-method-mechanism",
                    "a-method-input",
                ),
            ),
            EvidenceLedgerCell(
                requirement_id="b-method",
                status="missing",
                missing_fact_requirement_ids=("b-method-primary",),
            ),
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=missing_coverage,
                ledger=missing_ledger,
                followups=(
                    EvidenceFollowup(
                        requirement_id="a-method",
                        fact_requirement_id="a-method-mechanism",
                        query="Paper A Method: Paper A mechanism retrieval terms",
                        objective="Find the Paper A mechanism",
                    ),
                    EvidenceFollowup(
                        requirement_id="a-method",
                        fact_requirement_id="a-method-input",
                        query="Paper A Method: Paper A input retrieval terms",
                        objective="Find the Paper A input dependency",
                    ),
                ),
            ),
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=missing_coverage,
                ledger=missing_ledger,
            ),
        )
        query_scope = {
            "initial A discovery": ("C001", "a-initial"),
            "initial B discovery": ("T001", "b-initial"),
            "Paper A Method: Paper A mechanism retrieval terms": (
                "C001",
                "a-mechanism",
            ),
            "Paper A Method: Paper A input retrieval terms": ("C001", "a-input"),
        }
        service = AsyncMock()

        async def search(request: SearchCorpusInput) -> SearchCorpusResult:
            corpus_id, prefix = query_scope[request.query]
            self.assertEqual(request.corpus_id, corpus_id)
            return SearchCorpusResult(
                query=request.query,
                corpus_id=corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=tuple(
                    _hit(f"{prefix}-{rank}", corpus_id, rank)
                    for rank in range(1, 7)
                ),
            )

        async def hydrate(request: GetEvidenceInput) -> GetEvidenceResult:
            records = tuple(
                _record(
                    chunk_id,
                    "T001" if chunk_id.startswith("b-") else "C001",
                )
                for chunk_id in request.chunk_ids
            )
            return GetEvidenceResult(records=records)

        service.search_corpus.side_effect = search
        service.get_evidence.side_effect = hydrate
        policy = ResearchRuntimePolicy()
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=policy,
        )

        state = await graph.ainvoke({"question": "Compare methods"})

        self.assertEqual(
            [call.args[0].query for call in service.search_corpus.await_args_list],
            list(query_scope),
        )
        requested_chunks = [
            tuple(call.args[0].chunk_ids)
            for call in service.get_evidence.await_args_list
        ]
        self.assertEqual([len(chunk_ids) for chunk_ids in requested_chunks], [4] * 4)
        self.assertTrue(
            all(chunk_id.startswith("a-") for chunk_id in requested_chunks[2])
        )
        self.assertTrue(
            all(chunk_id.startswith("a-") for chunk_id in requested_chunks[3])
        )
        first_followups = state["assessments"][0]["followups"]
        self.assertEqual(
            [item["fact_requirement_id"] for item in first_followups],
            ["a-method-mechanism", "a-method-input"],
        )
        self.assertEqual(planner.calls, [("Compare methods", 20, False)])
        self.assertEqual([len(call[2]) for call in reasoner.calls], [2, 4])
        self.assertFalse(policy.adaptive_evidence_hydration_enabled)
        self.assertEqual((policy.max_steps, policy.max_tool_calls, policy.timeout_seconds), (24, 48, 180))
        self.assertEqual((state["step_budget"], state["tool_call_budget"]), (4, 8))
        self.assertEqual((state["current_step"], state["tool_call_count"]), (4, 8))

    async def test_comparison_hydration_expands_from_four_to_six_to_ten(self) -> None:
        targets = (
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        )
        dimension = ResearchDimension(dimension_id="method", label="Method")
        requirements = (
            EvidenceRequirement(
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                description="Paper A method",
            ),
            EvidenceRequirement(
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                description="Paper B method",
            ),
        )
        planner = FakePlanner(
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=(dimension,),
                requirements=requirements,
                steps=(
                    ResearchStep(
                        step_id="a-method",
                        objective="Find A method",
                        query="initial A",
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method",),
                    ),
                    ResearchStep(
                        step_id="b-method",
                        objective="Find B method",
                        query="initial B",
                        corpus_id="T001",
                        target_ids=("b",),
                        dimension_ids=("method",),
                    ),
                ),
            )
        )
        missing_coverage = (
            EvidenceCoverage(requirement_id="a-method", covered=False),
            EvidenceCoverage(requirement_id="b-method", covered=False),
        )
        reasoner = FakeReasoner(
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=missing_coverage,
                followups=(
                    EvidenceFollowup(
                        requirement_id="a-method",
                        query="followup A one",
                        objective="Retry A method",
                    ),
                ),
            ),
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=missing_coverage,
                followups=(
                    EvidenceFollowup(
                        requirement_id="a-method",
                        query="followup A two",
                        objective="Retry A method again",
                    ),
                ),
            ),
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=missing_coverage,
                followups=(
                    EvidenceFollowup(
                        requirement_id="a-method",
                        query="followup A three",
                        objective="Retry A method a third time",
                    ),
                ),
            ),
        )
        searches = (
            ("initial A", "C001", "a-initial"),
            ("initial B", "T001", "b-initial"),
            ("followup A one", "C001", "a-followup-one"),
            ("followup A two", "C001", "a-followup-two"),
        )
        service = AsyncMock()
        service.search_corpus.side_effect = tuple(
            SearchCorpusResult(
                query=query,
                corpus_id=corpus_id,
                index_id="idx-test",
                degraded=False,
                hits=tuple(
                    _hit(f"{prefix}-{rank}", corpus_id, rank)
                    for rank in range(1, 13)
                ),
            )
            for query, corpus_id, prefix in searches
        )

        async def hydrate(request: GetEvidenceInput) -> GetEvidenceResult:
            chunk_ids = tuple(request.chunk_ids)
            records = tuple(
                _record(
                    chunk_id,
                    "T001" if chunk_id.startswith("b-") else "C001",
                )
                for chunk_id in chunk_ids
            )
            return GetEvidenceResult(records=records)

        service.get_evidence.side_effect = hydrate
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(
                adaptive_evidence_hydration_enabled=True
            ),
        )

        state = await graph.ainvoke({"question": "Compare methods"})

        requested = [
            tuple(call.args[0].chunk_ids) for call in service.get_evidence.await_args_list
        ]
        self.assertEqual([len(chunk_ids) for chunk_ids in requested], [4, 4, 6, 10])
        self.assertTrue(all(item.startswith("a-") for item in requested[2]))
        self.assertTrue(all(item.startswith("a-") for item in requested[3]))
        self.assertEqual(state["step_budget"], 4)
        self.assertEqual(state["tool_call_budget"], 8)
        self.assertEqual(state["termination_reason"], "step_budget")

    async def test_stops_early_when_first_observation_is_sufficient(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="methods",
                        objective="Find evaluation methods",
                        query="RAG evaluation methods",
                        top_k=2,
                    ),
                    ResearchStep(
                        step_id="labels",
                        objective="Compare annotation needs",
                        query="RAG evaluation manual annotation",
                        top_k=2,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(_assessment(True))
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="RAG evaluation methods",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1), _hit("chunk-2", "T001", 2)),
        )
        service.get_evidence.return_value = GetEvidenceResult(
            records=(_record("chunk-1", "C001"), _record("chunk-2", "T001"))
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            max_steps=4,
            evidence_per_step=2,
        )

        state = await graph.ainvoke({"question": "Compare RAG evaluation methods"})

        self.assertEqual(planner.calls, [("Compare RAG evaluation methods", 4, False)])
        self.assertEqual(state["current_step"], 1)
        self.assertEqual(state["tool_call_count"], 2)
        self.assertTrue(state["evidence_sufficient"])
        self.assertEqual(state["termination_reason"], "evidence_sufficient")
        self.assertEqual(state["next_action"], "finish")
        self.assertEqual(
            [item["action"] for item in state["action_history"]],
            ["search_corpus", "get_evidence", "assess_evidence", "finish"],
        )
        self.assertEqual(service.search_corpus.await_count, 1)
        self.assertEqual(service.get_evidence.await_count, 1)

    async def test_compiler_failure_terminates_without_becoming_insufficient(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="methods",
                        objective="Find methods",
                        query="RAG evaluation methods",
                        top_k=1,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(_assessment(False, status="compiler_failed"))
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="RAG evaluation methods",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1),),
        )
        service.get_evidence.return_value = GetEvidenceResult(
            records=(_record("chunk-1", "C001"),)
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            max_steps=1,
        )

        state = await graph.ainvoke({"question": "Compare RAG evaluation methods"})

        self.assertFalse(state["evidence_sufficient"])
        self.assertEqual(state["assessments"][-1]["status"], "compiler_failed")
        self.assertEqual(state["termination_reason"], "compiler_failed")
        self.assertEqual(state["next_action"], "finish")

    async def test_replans_next_step_from_insufficient_evidence(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="methods",
                        objective="Find methods",
                        query="RAG evaluation methods",
                        top_k=1,
                    ),
                    ResearchStep(
                        step_id="generic-labels",
                        objective="Find labels",
                        query="RAG labels",
                        top_k=1,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(
            _assessment(
                False,
                next_query="RAG evaluation human annotation requirements",
                next_objective="Find human annotation requirements",
            ),
            _assessment(True),
        )
        service = AsyncMock()
        service.search_corpus.side_effect = (
            SearchCorpusResult(
                query="RAG evaluation methods",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-1", "C001", 1),),
            ),
            SearchCorpusResult(
                query="RAG evaluation human annotation requirements",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-2", "T001", 1),),
            ),
        )
        service.get_evidence.side_effect = (
            GetEvidenceResult(records=(_record("chunk-1", "C001"),)),
            GetEvidenceResult(records=(_record("chunk-2", "T001"),)),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
        )

        state = await graph.ainvoke({"question": "Compare methods and labels"})

        plan = ResearchPlan.model_validate(state["plan"])
        self.assertEqual([step.step_id for step in plan.steps], ["methods", "replan-1"])
        self.assertEqual(
            plan.steps[1].query,
            "RAG evaluation human annotation requirements",
        )
        self.assertEqual(state["current_step"], 2)
        self.assertEqual(state["replan_count"], 1)
        self.assertEqual(state["tool_call_count"], 4)
        self.assertEqual(state["termination_reason"], "evidence_sufficient")
        self.assertIn("replan", [item["action"] for item in state["action_history"]])
        search_inputs = [call.args[0] for call in service.search_corpus.await_args_list]
        self.assertEqual(
            [item.query for item in search_inputs],
            [
                "RAG evaluation methods",
                "RAG evaluation human annotation requirements",
            ],
        )

    async def test_repeated_replan_query_terminates_without_looping(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="search",
                        objective="Find evidence",
                        query="same query",
                    ),
                )
            )
        )
        reasoner = FakeReasoner(
            _assessment(
                False,
                status="no_hits",
                next_query=" same query ",
                next_objective="Try the same query again",
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="same query",
            index_id="idx-test",
            degraded=False,
            hits=(),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
        )

        state = await graph.ainvoke({"question": "A hard question"})

        self.assertEqual(state["termination_reason"], "repeated_query")
        self.assertEqual(state["current_step"], 1)
        self.assertEqual(state["replan_count"], 0)
        self.assertEqual(service.search_corpus.await_count, 1)

    async def test_continues_past_initial_plan_budget_for_new_evidence_query(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="initial",
                        objective="Find initial evidence",
                        query="initial query",
                        top_k=1,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(
            _assessment(
                False,
                next_query="focused follow-up query",
                next_objective="Fill the remaining evidence gap",
            ),
            _assessment(True),
        )
        service = AsyncMock()
        service.search_corpus.side_effect = (
            SearchCorpusResult(
                query="initial query",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-1", "C001", 1),),
            ),
            SearchCorpusResult(
                query="focused follow-up query",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-2", "T001", 1),),
            ),
        )
        service.get_evidence.side_effect = (
            GetEvidenceResult(records=(_record("chunk-1", "C001"),)),
            GetEvidenceResult(records=(_record("chunk-2", "T001"),)),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
        )

        state = await graph.ainvoke({"question": "Research beyond the initial plan"})

        self.assertEqual(state["current_step"], 2)
        self.assertEqual(state["tool_call_count"], 4)
        self.assertEqual(state["termination_reason"], "evidence_sufficient")
        self.assertEqual(
            [item["search"]["query"] for item in state["observations"]],
            ["initial query", "focused follow-up query"],
        )

    async def test_stops_after_two_rounds_without_new_evidence(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="first",
                        objective="Try the first query",
                        query="missing topic one",
                    ),
                    ResearchStep(
                        step_id="second",
                        objective="Try the second query",
                        query="missing topic two",
                    ),
                )
            )
        )
        reasoner = FakeReasoner(
            _assessment(False, status="no_hits"),
            _assessment(False, status="no_hits"),
        )
        service = AsyncMock()
        service.search_corpus.side_effect = (
            SearchCorpusResult(
                query="missing topic one",
                index_id="idx-test",
                degraded=False,
                hits=(),
            ),
            SearchCorpusResult(
                query="missing topic two",
                index_id="idx-test",
                degraded=False,
                hits=(),
            ),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=8),
        )

        state = await graph.ainvoke({"question": "Find a missing topic"})

        self.assertEqual(state["termination_reason"], "no_new_evidence")
        self.assertEqual(state["consecutive_no_new_evidence"], 2)
        self.assertEqual(service.search_corpus.await_count, 2)

    async def test_tool_budget_finishes_gracefully_before_another_round(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="initial",
                        objective="Find initial evidence",
                        query="budgeted query",
                        top_k=1,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(
            _assessment(
                False,
                next_query="another focused query",
                next_objective="Continue the search",
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="budgeted query",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1),),
        )
        service.get_evidence.return_value = GetEvidenceResult(
            records=(_record("chunk-1", "C001"),)
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=ResearchRuntimePolicy(max_steps=3, max_tool_calls=2),
        )

        state = await graph.ainvoke({"question": "Use the safe tool budget"})

        self.assertEqual(state["termination_reason"], "tool_budget")
        self.assertEqual(state["tool_call_count"], 2)
        self.assertEqual(service.search_corpus.await_count, 1)

    async def test_rejects_a_plan_above_the_graph_step_budget(self) -> None:
        plan = ResearchPlan(
            steps=tuple(
                ResearchStep(
                    step_id=f"step-{index}",
                    objective=f"Objective {index}",
                    query=f"query {index}",
                )
                for index in range(3)
            )
        )
        graph = build_research_graph(
            planner=FakePlanner(plan),
            reasoner=FakeReasoner(),
            service=AsyncMock(),
            max_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "step budget"):
            await graph.ainvoke({"question": "Complex question"})

    async def test_skips_evidence_hydration_when_search_has_no_hits(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="search",
                        objective="Find evidence",
                        query="missing topic",
                    ),
                )
            )
        )
        reasoner = FakeReasoner(_assessment(False, status="no_hits"))
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="missing topic",
            index_id="idx-test",
            degraded=False,
            hits=(),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
        )

        state = await graph.ainvoke({"question": "Question with no hits"})

        service.get_evidence.assert_not_awaited()
        self.assertEqual(state["observations"][0]["evidence"]["records"], [])
        self.assertEqual(state["evidence_records"], [])
        self.assertEqual(state["tool_call_count"], 1)
        self.assertEqual(state["termination_reason"], "plan_exhausted")

    async def test_rejects_evidence_call_above_runtime_tool_budget(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="search",
                        objective="Find evidence",
                        query="bounded tools",
                    ),
                )
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="bounded tools",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1),),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=FakeReasoner(),
            service=service,
            policy=ResearchRuntimePolicy(max_tool_calls=1),
        )

        with self.assertRaisesRegex(RuntimeError, "tool call budget"):
            await graph.ainvoke({"question": "Budget-limited question"})
        service.get_evidence.assert_not_awaited()

    async def test_checkpoint_can_restore_completed_thread_state(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="search",
                        objective="Find evidence",
                        query="checkpoint",
                    ),
                )
            )
        )
        reasoner = FakeReasoner(_assessment(False, status="no_hits"))
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="checkpoint",
            index_id="idx-test",
            degraded=False,
            hits=(),
        )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "research-thread-1"}}

        await graph.ainvoke({"question": "Verify checkpoint"}, config=config)
        snapshot = await graph.aget_state(config)

        self.assertEqual(snapshot.values["question"], "Verify checkpoint")
        self.assertEqual(snapshot.values["current_step"], 1)
        self.assertEqual(snapshot.values["tool_call_count"], 1)
        self.assertEqual(snapshot.values["termination_reason"], "plan_exhausted")
        self.assertEqual(snapshot.next, ())

    async def test_emits_body_free_node_and_tool_events(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="private-step",
                        objective="Find private evidence",
                        query="private retrieval query",
                        top_k=1,
                    ),
                )
            )
        )
        reasoner = FakeReasoner(_assessment(True))
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="private retrieval query",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1),),
        )
        service.get_evidence.return_value = GetEvidenceResult(records=(_record("chunk-1", "C001"),))
        events = EventRecorder()
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            event_sink=events,
        )

        await graph.ainvoke({"question": "private research question", "run_id": "a" * 32})

        event_types = [event.event_type for event in events.events]
        self.assertIn("node_started", event_types)
        self.assertIn("node_completed", event_types)
        self.assertEqual(event_types.count("tool_started"), 2)
        self.assertEqual(event_types.count("tool_completed"), 2)
        tool_events = [event for event in events.events if event.component == "tool"]
        self.assertEqual(tool_events[-1].returned_count, 1)
        serialized = "".join(event.model_dump_json() for event in events.events)
        self.assertNotIn("private research question", serialized)
        self.assertNotIn("private retrieval query", serialized)
        self.assertNotIn("private-step", serialized)
        self.assertNotIn("chunk-1", serialized)
        self.assertNotIn("Evidence for chunk-1", serialized)

    async def test_emits_runtime_intercept_when_tool_budget_is_exhausted(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="bounded",
                        objective="Find evidence",
                        query="bounded query",
                    ),
                )
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="bounded query",
            index_id="idx-test",
            degraded=False,
            hits=(_hit("chunk-1", "C001", 1),),
        )
        events = EventRecorder()
        graph = build_research_graph(
            planner=planner,
            reasoner=FakeReasoner(),
            service=service,
            policy=ResearchRuntimePolicy(max_tool_calls=1),
            event_sink=events,
        )

        with self.assertRaisesRegex(RuntimeError, "tool call budget"):
            await graph.ainvoke({"question": "question", "run_id": "b" * 32})

        intercepted = [
            event for event in events.events if event.event_type == "runtime_intercepted"
        ]
        self.assertEqual(len(intercepted), 1)
        self.assertEqual(intercepted[0].name, "get_evidence")
        self.assertEqual(intercepted[0].reason_code, "tool_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
