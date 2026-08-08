from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import InMemorySaver

from paper_research_agent.agent.graph import build_research_graph
from paper_research_agent.agent.models import (
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceResult,
    ResearchDimension,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusHit,
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
                evidence_sufficient=False,
                status="missing_coverage",
                coverage=(
                    EvidenceCoverage(
                        requirement_id="a-method", covered=True, chunk_ids=("chunk-1",)
                    ),
                    EvidenceCoverage(requirement_id="b-method", covered=False),
                ),
                next_query="Paper B refined method",
                next_objective="Refine Paper B method",
                next_requirement_ids=("b-method",),
            ),
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
            policy=ResearchRuntimePolicy(max_steps=1, max_tool_calls=4),
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
            policy=ResearchRuntimePolicy(max_steps=1, max_tool_calls=2),
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
