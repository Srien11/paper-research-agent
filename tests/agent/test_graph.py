from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import InMemorySaver

from paper_research_agent.agent.graph import build_research_graph
from paper_research_agent.agent.models import (
    EvidenceRecord,
    GetEvidenceResult,
    ResearchPlan,
    ResearchStep,
    SearchCorpusHit,
    SearchCorpusResult,
)


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


class FakePlanner:
    def __init__(self, plan: ResearchPlan):
        self.plan_value = plan
        self.calls: list[tuple[str, int]] = []

    async def plan(self, question: str, *, max_steps: int) -> ResearchPlan:
        self.calls.append((question, max_steps))
        return self.plan_value


class ResearchGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_executes_planned_steps_and_deduplicates_evidence(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="methods",
                        objective="查找评测方法",
                        query="RAG evaluation methods",
                        top_k=2,
                    ),
                    ResearchStep(
                        step_id="labels",
                        objective="比较人工标注需求",
                        query="RAG evaluation manual annotation",
                        top_k=2,
                    ),
                )
            )
        )
        service = AsyncMock()
        service.search_corpus.side_effect = (
            SearchCorpusResult(
                query="RAG evaluation methods",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-1", "C001", 1), _hit("chunk-2", "T001", 2)),
            ),
            SearchCorpusResult(
                query="RAG evaluation manual annotation",
                index_id="idx-test",
                degraded=False,
                hits=(_hit("chunk-1", "C001", 1),),
            ),
        )
        service.get_evidence.side_effect = (
            GetEvidenceResult(records=(_record("chunk-1", "C001"), _record("chunk-2", "T001"))),
            GetEvidenceResult(records=(_record("chunk-1", "C001"),)),
        )
        graph = build_research_graph(
            planner=planner,
            service=service,
            max_steps=4,
            evidence_per_step=2,
        )

        state = await graph.ainvoke({"question": "比较 RAG 评测方法的标注需求"})

        self.assertEqual(planner.calls, [("比较 RAG 评测方法的标注需求", 4)])
        self.assertEqual(state["current_step"], 2)
        self.assertEqual(len(state["observations"]), 2)
        self.assertEqual(
            [record["chunk_id"] for record in state["evidence_records"]],
            ["chunk-1", "chunk-2"],
        )
        self.assertEqual(service.search_corpus.await_count, 2)
        self.assertEqual(service.get_evidence.await_count, 2)

    async def test_rejects_a_plan_above_the_graph_step_budget(self) -> None:
        plan = ResearchPlan(
            steps=tuple(
                ResearchStep(
                    step_id=f"step-{index}",
                    objective=f"目标 {index}",
                    query=f"query {index}",
                )
                for index in range(3)
            )
        )
        graph = build_research_graph(
            planner=FakePlanner(plan),
            service=AsyncMock(),
            max_steps=2,
        )

        with self.assertRaisesRegex(ValueError, "step budget"):
            await graph.ainvoke({"question": "复杂问题"})

    async def test_skips_evidence_hydration_when_search_has_no_hits(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(step_id="search", objective="查找证据", query="missing topic"),
                )
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="missing topic",
            index_id="idx-test",
            degraded=False,
            hits=(),
        )
        graph = build_research_graph(planner=planner, service=service)

        state = await graph.ainvoke({"question": "没有命中的问题"})

        service.get_evidence.assert_not_awaited()
        self.assertEqual(state["observations"][0]["evidence"]["records"], [])
        self.assertEqual(state["evidence_records"], [])

    async def test_checkpoint_can_restore_completed_thread_state(self) -> None:
        planner = FakePlanner(
            ResearchPlan(
                steps=(
                    ResearchStep(step_id="search", objective="查找证据", query="checkpoint"),
                )
            )
        )
        service = AsyncMock()
        service.search_corpus.return_value = SearchCorpusResult(
            query="checkpoint",
            index_id="idx-test",
            degraded=False,
            hits=(),
        )
        graph = build_research_graph(
            planner=planner,
            service=service,
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "research-thread-1"}}

        await graph.ainvoke({"question": "验证检查点"}, config=config)
        snapshot = await graph.aget_state(config)

        self.assertEqual(snapshot.values["question"], "验证检查点")
        self.assertEqual(snapshot.values["current_step"], 1)
        self.assertEqual(snapshot.next, ())


if __name__ == "__main__":
    unittest.main()
