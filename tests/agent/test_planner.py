from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from paper_research_agent.agent.models import ResearchPlan, ResearchStep
from paper_research_agent.agent.planner import (
    ComparisonTargetResolutionError,
    LangChainComparisonTargetResolver,
    LangChainResearchPlanner,
    parse_explicit_corpus_ids,
)
from paper_research_agent.retrieval.contracts import QueryRewriteTrace
from paper_research_agent.retrieval.papers import PaperCandidateHit, PaperCandidateQuery


class LangChainComparisonTargetResolverTests(unittest.IsolatedAsyncioTestCase):
    def test_explicit_parser_rejects_partial_or_overlong_identifiers(self) -> None:
        self.assertEqual(parse_explicit_corpus_ids("C001/T002 and c003"), ("C001", "T002", "C003"))
        self.assertEqual(parse_explicit_corpus_ids("XC001 C0010 C01 T0001"), ())

    async def test_explicit_ids_bypass_candidate_retrieval_and_selection(self) -> None:
        candidate_retriever = AsyncMock()
        query_resolver = AsyncMock()
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=candidate_retriever,
            query_resolver=query_resolver,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        resolved = await resolver.resolve("Compare C001 with T001")

        self.assertEqual(resolved, {"C001": "Paper A", "T001": "Paper B"})
        candidate_retriever.search.assert_not_awaited()
        query_resolver.resolve_query.assert_not_awaited()

    async def test_implicit_targets_use_only_paper_level_candidates(self) -> None:
        candidate_retriever = AsyncMock()
        query_resolver = AsyncMock()
        query_resolver.resolve_query.return_value = QueryRewriteTrace(
            status="success",
            english_query="Faithful English comparison with all constraints",
            requested_model="qwen-test",
            actual_model="qwen-test",
            prompt_version="query-rewrite-v3",
            latency_ms=1,
        )
        candidate_retriever.search.return_value = (
            PaperCandidateHit(
                corpus_id="C001",
                title="Paper A",
                abstract="Paper A abstract",
                used_fallback=False,
                final_score=0.04,
                ranks={"bm25": 1, "vector": 2},
            ),
            PaperCandidateHit(
                corpus_id="T001",
                title="Paper B",
                abstract="Paper B abstract",
                used_fallback=False,
                final_score=0.03,
                ranks={"bm25": 2, "vector": 1},
            ),
        )
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=candidate_retriever,
            query_resolver=query_resolver,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        resolved = await resolver.resolve("Compare the two described methods")

        query_resolver.resolve_query.assert_awaited_once_with(
            "Compare the two described methods"
        )
        candidate_retriever.search.assert_awaited_once_with(
            PaperCandidateQuery(
                original_query="Compare the two described methods",
                english_query="Faithful English comparison with all constraints",
            ),
            top_k=8,
        )
        self.assertEqual(resolved, {"C001": "Paper A", "T001": "Paper B"})

    async def test_rejects_candidate_outside_local_catalog(self) -> None:
        candidate_retriever = AsyncMock()
        query_resolver = AsyncMock()
        query_resolver.resolve_query.return_value = QueryRewriteTrace(
            status="error",
            requested_model="qwen-test",
            prompt_version="query-rewrite-v3",
            latency_ms=1,
            error_class="QueryRewriteError",
            fallback_reason="error",
        )
        candidate_retriever.search.return_value = (
            PaperCandidateHit(
                corpus_id="C001",
                title="Paper A",
                abstract="A",
                used_fallback=False,
                final_score=0.04,
                ranks={"bm25": 1},
            ),
            PaperCandidateHit(
                corpus_id="T999",
                title="Outside",
                abstract="Outside",
                used_fallback=False,
                final_score=0.03,
                ranks={"vector": 1},
            ),
        )
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=candidate_retriever,
            query_resolver=query_resolver,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await resolver.resolve("Compare the two methods")

        self.assertEqual(raised.exception.reason_code, "candidate_outside_catalog")
        candidate_retriever.search.assert_awaited_once_with(
            PaperCandidateQuery(original_query="Compare the two methods"),
            top_k=8,
        )

    async def test_explicit_id_parser_is_exact_and_case_insensitive(self) -> None:
        cases = (
            ("compare c001, T001.", {"C001": "Paper A", "T001": "Paper B"}),
            ("C001 versus C001 and t001", {"C001": "Paper A", "T001": "Paper B"}),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                candidate_retriever = AsyncMock()
                query_resolver = AsyncMock()
                resolver = LangChainComparisonTargetResolver(
                    candidate_retriever=candidate_retriever,
                    query_resolver=query_resolver,
                    corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
                )
                self.assertEqual(await resolver.resolve(question), expected)
                candidate_retriever.search.assert_not_awaited()
                query_resolver.resolve_query.assert_not_awaited()

    async def test_unknown_explicit_id_fails_before_candidate_retrieval(self) -> None:
        candidate_retriever = AsyncMock()
        query_resolver = AsyncMock()
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=candidate_retriever,
            query_resolver=query_resolver,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await resolver.resolve("Compare C001 and T999")

        self.assertEqual(raised.exception.reason_code, "unknown_explicit_corpus_id")
        candidate_retriever.search.assert_not_awaited()
        query_resolver.resolve_query.assert_not_awaited()


class LangChainResearchPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_bounded_structured_plan_from_chat_model(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "schema_version": "research-plan-v1",
            "steps": [
                {
                    "step_id": "methods",
                    "objective": "查找评测方法",
                    "query": "RAG evaluation methods",
                    "top_k": 6,
                }
            ],
        }
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(model, corpus_catalog={"C001": "Paper A"})

        plan = await planner.plan("比较 RAG 评测方法", max_steps=3)

        model.with_structured_output.assert_called_once_with(
            ResearchPlan,
            method="function_calling",
        )
        self.assertEqual(plan.steps[0].query, "RAG evaluation methods")
        messages = structured.ainvoke.await_args.args[0]
        self.assertIn("最多 3", messages[0].content)
        self.assertIn("比较对象", messages[0].content)
        self.assertIn("逐对象", messages[0].content)
        self.assertIn("完整证据网格", messages[0].content)
        self.assertIn('"C001":"Paper A"', messages[0].content)
        self.assertEqual(messages[1].content, "比较 RAG 评测方法")

    async def test_repairs_model_plan_above_requested_budget_with_safe_fallback(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = ResearchPlan(
            steps=(
                ResearchStep(step_id="one", objective="一", query="one"),
                ResearchStep(step_id="two", objective="二", query="two"),
            )
        )
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(model)

        plan = await planner.plan("问题", max_steps=1)

        self.assertEqual(structured.ainvoke.await_count, 2)
        self.assertEqual(plan.steps[0].step_id, "fallback")

    async def test_retries_when_upstream_requires_comparison_plan(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.side_effect = (
            {
                "schema_version": "research-plan-v1",
                "steps": [
                    {
                        "step_id": "broad",
                        "objective": "Broad search",
                        "query": "Paper A Paper B",
                    }
                ],
            },
            {
                "task_type": "comparison",
                "targets": [
                    {"target_id": "a", "label": "Paper A", "corpus_id": "C001"},
                    {"target_id": "b", "label": "Paper B", "corpus_id": "T001"},
                ],
                "dimensions": [{"dimension_id": "method", "label": "Method"}],
                "requirements": [
                    {
                        "requirement_id": "a-method",
                        "target_id": "a",
                        "dimension_id": "method",
                        "description": "Paper A method",
                    },
                    {
                        "requirement_id": "b-method",
                        "target_id": "b",
                        "dimension_id": "method",
                        "description": "Paper B method",
                    },
                ],
                "steps": [
                    {
                        "step_id": "a",
                        "objective": "Paper A method",
                        "query": "Paper A method",
                        "corpus_id": "C001",
                        "target_ids": ["a"],
                        "dimension_ids": ["method"],
                    },
                    {
                        "step_id": "b",
                        "objective": "Paper B method",
                        "query": "Paper B method",
                        "corpus_id": "T001",
                        "target_ids": ["b"],
                        "dimension_ids": ["method"],
                    },
                ],
            },
        )
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(model)

        plan = await planner.plan(
            "Compare Paper A and Paper B",
            max_steps=2,
            planning_required=True,
        )

        self.assertEqual(plan.task_type, "comparison")
        self.assertEqual(structured.ainvoke.await_count, 2)

    async def test_required_comparison_fails_closed_after_two_invalid_plans(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "schema_version": "research-plan-v1",
            "steps": [{"step_id": "broad", "objective": "Broad", "query": "broad"}],
        }
        model.with_structured_output.return_value = structured

        planner = LangChainResearchPlanner(model)

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await planner.plan("Compare C001 and T001", max_steps=2, planning_required=True)

        self.assertEqual(raised.exception.reason_code, "planner_contract_invalid")

    async def test_required_comparison_rejects_targets_outside_resolved_candidates(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "task_type": "comparison",
            "targets": [
                {"target_id": "a", "label": "Paper A", "corpus_id": "C001"},
                {"target_id": "x", "label": "Paper X", "corpus_id": "T999"},
            ],
            "dimensions": [{"dimension_id": "method", "label": "Method"}],
            "requirements": [
                {
                    "requirement_id": "a-method",
                    "target_id": "a",
                    "dimension_id": "method",
                    "description": "Paper A method",
                },
                {
                    "requirement_id": "x-method",
                    "target_id": "x",
                    "dimension_id": "method",
                    "description": "Paper X method",
                },
            ],
            "steps": [
                {
                    "step_id": "a-method",
                    "objective": "Paper A method",
                    "query": "Paper A method",
                    "corpus_id": "C001",
                    "target_ids": ["a"],
                    "dimension_ids": ["method"],
                },
                {
                    "step_id": "x-method",
                    "objective": "Paper X method",
                    "query": "Paper X method",
                    "corpus_id": "T999",
                    "target_ids": ["x"],
                    "dimension_ids": ["method"],
                },
            ],
        }
        model.with_structured_output.return_value = structured
        resolver = AsyncMock()
        resolver.resolve.return_value = {"C001": "Paper A", "T001": "Paper B"}
        planner = LangChainResearchPlanner(model, target_resolver=resolver)

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await planner.plan("Compare the methods", max_steps=2, planning_required=True)

        self.assertEqual(raised.exception.reason_code, "planner_contract_invalid")
        self.assertEqual(structured.ainvoke.await_count, 2)


if __name__ == "__main__":
    unittest.main()
