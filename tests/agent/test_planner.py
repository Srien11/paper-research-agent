from __future__ import annotations

import copy
import unittest
from unittest.mock import AsyncMock, Mock

from paper_research_agent.agent.models import (
    ComparisonPlanProposal,
    ResearchPlan,
    ResearchStep,
)
from paper_research_agent.agent.planner import (
    ComparisonPlanningError,
    ComparisonTargetResolutionError,
    LangChainComparisonTargetResolver,
    LangChainResearchPlanner,
    _planner_failure_code,
    build_comparison_research_plan,
    comparison_dimension_hints,
    parse_explicit_corpus_ids,
)
from paper_research_agent.retrieval.contracts import QueryRewriteTrace
from paper_research_agent.retrieval.papers import PaperCandidateHit, PaperCandidateQuery


CPG020_LIKE_QUESTION = (
    "在逻辑推理自我修正研究中，一篇认为没有外部反馈时经常无效，"
    "另一篇区分找错与改错，指出给出错误位置后能够纠正。请找出论文。"
)


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

    async def test_too_many_explicit_ids_remain_target_resolution_failure(self) -> None:
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=AsyncMock(),
            query_resolver=AsyncMock(),
            corpus_catalog={
                "C001": "Paper A",
                "C002": "Paper B",
                "C003": "Paper C",
                "C004": "Paper D",
                "C005": "Paper E",
            },
        )

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await resolver.resolve("Compare C001 C002 C003 C004 C005")

        self.assertEqual(raised.exception.reason_code, "too_many_explicit_corpus_ids")

    async def test_insufficient_candidates_remain_target_resolution_failure(self) -> None:
        candidate_retriever = AsyncMock()
        candidate_retriever.search.return_value = ()
        query_resolver = AsyncMock()
        query_resolver.resolve_query.return_value = QueryRewriteTrace(
            status="error",
            requested_model="qwen-test",
            prompt_version="query-rewrite-v3",
            latency_ms=1,
            error_class="QueryRewriteError",
            fallback_reason="error",
        )
        resolver = LangChainComparisonTargetResolver(
            candidate_retriever=candidate_retriever,
            query_resolver=query_resolver,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        with self.assertRaises(ComparisonTargetResolutionError) as raised:
            await resolver.resolve("Compare the described methods")

        self.assertEqual(
            raised.exception.reason_code,
            "insufficient_retrieval_candidates",
        )


class LangChainResearchPlannerTests(unittest.IsolatedAsyncioTestCase):
    def test_derives_explicit_dimension_hints_without_answer_data(self) -> None:
        self.assertEqual(
            comparison_dimension_hints(
                "比较两项研究：一项讨论偏差并以竞技场验证，"
                "另一项展示顺序操纵。请找出论文。"
            ),
            ("一项讨论偏差", "以竞技场验证", "另一项展示顺序操纵"),
        )
        self.assertEqual(
            comparison_dimension_hints(
                "在逻辑推理自我修正研究中，一篇认为没有外部反馈时经常无效，"
                "另一篇区分找错与改错，指出给出错误位置后能够纠正。请找出论文。"
            ),
            (
                "一篇认为没有外部反馈时经常无效",
                "另一篇区分找错与改错",
                "指出给出错误位置后能够纠正",
            ),
        )

    def test_builds_comparison_control_plane_deterministically(self) -> None:
        proposal = ComparisonPlanProposal.model_validate(
            {
                "selected_corpus_ids": ["C001", "T001"],
                "dimension_labels": ["Method", "Result"],
                "facts": [
                    {
                        "dimension_index": dimension_index,
                        "description": f"Fact {dimension_index}",
                        "protected_anchor_ids": [dimension_index],
                    }
                    for dimension_index in (0, 1)
                ],
            }
        )

        plan = build_comparison_research_plan(
            proposal,
            resolved_catalog={"C001": "Paper A", "T001": "Paper B"},
            dimension_labels=("Method", "Result"),
            max_steps=4,
        )

        self.assertEqual(
            [item.target_id for item in plan.targets],
            ["target-01", "target-02"],
        )
        self.assertEqual(
            [item.dimension_id for item in plan.dimensions],
            ["dimension-01", "dimension-02"],
        )
        self.assertEqual(
            [item.requirement_id for item in plan.requirements],
            [
                "requirement-01-01",
                "requirement-01-02",
                "requirement-02-01",
                "requirement-02-02",
            ],
        )
        self.assertEqual(len(plan.steps), 4)
        self.assertTrue(
            all(step.corpus_id == target.corpus_id for target in plan.targets for step in plan.steps if step.target_ids == (target.target_id,))
        )

    def test_local_dimension_skeleton_owns_count_order_and_ids(self) -> None:
        hints = comparison_dimension_hints(CPG020_LIKE_QUESTION)
        proposal = ComparisonPlanProposal.model_validate(
            {
                "selected_corpus_ids": ["C001", "T001"],
                "facts": [
                    {
                        "dimension_index": index,
                        "description": hint,
                        "protected_anchor_ids": [index + 2],
                    }
                    for index, hint in enumerate(hints)
                ],
            }
        )

        plan = build_comparison_research_plan(
            proposal,
            resolved_catalog={"C001": "Paper A", "T001": "Paper B"},
            dimension_labels=hints,
            max_steps=6,
        )

        self.assertEqual(tuple(item.label for item in plan.dimensions), hints)
        self.assertEqual(
            tuple(item.dimension_id for item in plan.dimensions),
            ("dimension-01", "dimension-02", "dimension-03"),
        )
        self.assertEqual(len(plan.requirements), 6)
        self.assertEqual(len(plan.steps), 6)

    def test_comparison_builder_rejects_invalid_local_skeletons(self) -> None:
        proposal = ComparisonPlanProposal.model_validate(
            {
                "selected_corpus_ids": ["C001", "T001"],
                "facts": [
                    {
                        "dimension_index": 0,
                        "description": "Method",
                        "protected_anchor_ids": [0],
                    }
                ],
            }
        )
        for labels in ((), ("Method", "method"), (" ",)):
            with self.subTest(labels=labels), self.assertRaises(ValueError) as raised:
                build_comparison_research_plan(
                    proposal,
                    resolved_catalog={"C001": "Paper A", "T001": "Paper B"},
                    dimension_labels=labels,
                    max_steps=6,
                )
            self.assertEqual(
                _planner_failure_code(raised.exception),
                "local_dimension_skeleton_invalid",
            )

    def test_fact_indices_and_grid_budget_have_stable_failure_codes(self) -> None:
        hints = ("Method", "Result", "Limitation")
        base_facts = [
            {
                "dimension_index": index,
                "description": hint,
                "protected_anchor_ids": [index],
            }
            for index, hint in enumerate(hints)
        ]
        cases = (
            (
                "planner_fact_proposal_invalid",
                base_facts[:2],
                ("C001", "T001"),
                6,
            ),
            (
                "planner_fact_dimension_reference_invalid",
                [*base_facts, {**base_facts[-1], "dimension_index": 3}],
                ("C001", "T001"),
                8,
            ),
            (
                "planner_step_budget_invalid",
                base_facts,
                ("C001", "T001", "C002"),
                6,
            ),
        )
        catalog = {
            "C001": "Paper A",
            "T001": "Paper B",
            "C002": "Paper C",
        }
        for expected, facts, corpus_ids, max_steps in cases:
            with self.subTest(expected=expected), self.assertRaises(ValueError) as raised:
                build_comparison_research_plan(
                    ComparisonPlanProposal.model_validate(
                        {
                            "selected_corpus_ids": corpus_ids,
                            "facts": facts,
                        }
                    ),
                    resolved_catalog=catalog,
                    dimension_labels=hints,
                    max_steps=max_steps,
                )
            self.assertEqual(_planner_failure_code(raised.exception), expected)

    def test_multiple_atomic_facts_do_not_change_fixed_dimension_grid(self) -> None:
        proposal = ComparisonPlanProposal.model_validate(
            {
                "selected_corpus_ids": ["C001", "T001"],
                "facts": [
                    {
                        "dimension_index": 0,
                        "description": "Method architecture",
                        "protected_anchor_ids": [0],
                    },
                    {
                        "dimension_index": 0,
                        "description": "Method input",
                        "protected_anchor_ids": [1],
                    },
                    {
                        "dimension_index": 1,
                        "description": "Result",
                        "protected_anchor_ids": [2],
                    },
                    {
                        "dimension_index": 2,
                        "description": "Limitation",
                        "protected_anchor_ids": [3],
                    },
                ],
            }
        )

        plan = build_comparison_research_plan(
            proposal,
            resolved_catalog={"C001": "Paper A", "T001": "Paper B"},
            dimension_labels=("Method", "Result", "Limitation"),
            max_steps=8,
        )

        self.assertEqual(len(plan.dimensions), 3)
        self.assertEqual(len(plan.requirements), 6)
        self.assertEqual(len(plan.steps), 6)

    async def test_provider_dimension_labels_cannot_change_local_skeleton(self) -> None:
        hints = comparison_dimension_hints(CPG020_LIKE_QUESTION)
        for ignored_labels in (
            ["A", "B"],
            ["A", "B", "C", "D"],
            ["A", "A", "C"],
            ["", "B", "C"],
        ):
            with self.subTest(ignored_labels=ignored_labels):
                model = Mock()
                structured = AsyncMock()
                structured.ainvoke.return_value = {
                    "selected_corpus_ids": ["C001", "T001"],
                    "dimension_labels": ignored_labels,
                    "facts": [
                        {
                            "dimension_index": index,
                            "description": hint,
                            "protected_anchor_ids": [index + 2],
                        }
                        for index, hint in enumerate(hints)
                    ],
                }
                model.with_structured_output.return_value = structured
                planner = LangChainResearchPlanner(
                    model,
                    corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
                )

                plan = await planner.plan(
                    CPG020_LIKE_QUESTION,
                    max_steps=6,
                    planning_required=True,
                )

                self.assertEqual(
                    [item.label for item in plan.dimensions],
                    list(hints),
                )
                self.assertEqual(len(plan.steps), 6)

    def test_planner_failures_have_stable_body_free_codes(self) -> None:
        base = {
            "task_type": "comparison",
            "targets": [
                {"target_id": "a", "label": "Paper A", "corpus_id": "C001"},
                {"target_id": "b", "label": "Paper B", "corpus_id": "T001"},
            ],
            "dimensions": [{"dimension_id": "method", "label": "Method"}],
            "requirements": [
                {
                    "requirement_id": f"{target}-method",
                    "target_id": target,
                    "dimension_id": "method",
                    "description": f"Paper {target} method",
                    "fact_requirements": [
                        {
                            "fact_requirement_id": f"{target}-fact",
                            "description": f"Paper {target} mechanism",
                            "protected_anchor_ids": [0],
                        }
                    ],
                }
                for target in ("a", "b")
            ],
            "steps": [
                {
                    "step_id": target,
                    "objective": f"Paper {target} method",
                    "query": f"Paper {target} method",
                    "corpus_id": "C001" if target == "a" else "T001",
                    "target_ids": [target],
                    "dimension_ids": ["method"],
                }
                for target in ("a", "b")
            ],
        }

        invalid_payloads: list[tuple[str, dict[str, object]]] = []
        one_target = copy.deepcopy(base)
        one_target["targets"] = one_target["targets"][:1]
        one_target["requirements"] = one_target["requirements"][:1]
        one_target["steps"] = one_target["steps"][:1]
        invalid_payloads.append(("planner_target_selection_invalid", one_target))
        duplicate_target = copy.deepcopy(base)
        duplicate_target["targets"][1]["corpus_id"] = "C001"
        invalid_payloads.append(("planner_target_selection_invalid", duplicate_target))
        no_dimensions = copy.deepcopy(base)
        no_dimensions["dimensions"] = []
        invalid_payloads.append(("local_dimension_skeleton_invalid", no_dimensions))
        incomplete_grid = copy.deepcopy(base)
        incomplete_grid["requirements"] = incomplete_grid["requirements"][:1]
        invalid_payloads.append(("planner_fact_proposal_invalid", incomplete_grid))
        bad_requirement_reference = copy.deepcopy(base)
        bad_requirement_reference["requirements"][0]["target_id"] = "outside"
        invalid_payloads.append(
            ("planner_fact_dimension_reference_invalid", bad_requirement_reference)
        )
        bad_step_scope = copy.deepcopy(base)
        bad_step_scope["steps"][0]["target_ids"] = ["a", "b"]
        invalid_payloads.append(("planner_step_scope_invalid", bad_step_scope))
        incomplete_steps = copy.deepcopy(base)
        incomplete_steps["steps"] = incomplete_steps["steps"][:1]
        invalid_payloads.append(("planner_step_grid_incomplete", incomplete_steps))
        duplicate_id = copy.deepcopy(base)
        duplicate_id["steps"][1]["step_id"] = "a"
        invalid_payloads.append(("planner_id_duplicate", duplicate_id))

        errors: list[tuple[str, ValueError]] = [
            ("planner_task_type_invalid", ValueError("planned research requires a comparison plan")),
            ("planner_target_selection_invalid", ValueError("comparison plan left the resolved candidate set")),
            ("planner_step_budget_invalid", ValueError("research plan exceeds the requested step budget")),
            ("planner_fact_budget_invalid", ValueError("comparison fact requirements exceed the requested step budget")),
            ("planner_anchor_selection_invalid", ValueError("protected anchor selection is invalid")),
        ]
        for expected, payload in invalid_payloads:
            with self.subTest(expected=expected):
                try:
                    ResearchPlan.model_validate(payload)
                except ValueError as error:
                    self.assertEqual(_planner_failure_code(error), expected)
                else:
                    self.fail("invalid plan unexpectedly validated")
        for expected, error in errors:
            with self.subTest(expected=expected):
                self.assertEqual(_planner_failure_code(error), expected)
        try:
            ResearchPlan.model_validate({"steps": "model-authored body"})
        except ValueError as error:
            self.assertEqual(_planner_failure_code(error), "planner_schema_invalid")

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
        self.assertIn("protected_anchor_ids", messages[0].content)
        self.assertIn("retrieval_expansions", messages[0].content)
        self.assertIn("select the shortest available catalog spans", messages[0].content)
        self.assertIn("augment", messages[0].content)
        self.assertIn("Do not replace", messages[0].content)
        self.assertIn("VERBATIM_ANCHOR_SOURCE_JSON", messages[0].content)
        self.assertIn("do not answer the question", messages[0].content)
        self.assertIn("one bounded discovery step per target-by-dimension cell", messages[0].content)
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
                "selected_corpus_ids": ["C001"],
                "dimension_labels": ["Method"],
                "facts": [
                    {
                        "dimension_index": 0,
                        "description": "Paper A mechanism",
                        "protected_anchor_ids": [2],
                    }
                ],
            },
            {
                "selected_corpus_ids": ["C001", "T001"],
                "dimension_labels": ["Mechanism", "Input dependency"],
                "facts": [
                    {
                        "dimension_index": 0,
                        "description": "Paper A core mechanism",
                        "protected_anchor_ids": [2],
                        "retrieval_expansions": ["mechanism", "architecture"],
                    },
                    {
                        "dimension_index": 1,
                        "description": "Paper A input dependency",
                        "protected_anchor_ids": [3],
                        "retrieval_expansions": ["required input"],
                    },
                ],
            },
        )
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(
            model,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        plan = await planner.plan(
            "Compare Paper A and Paper B: core mechanism; input dependency.",
            max_steps=4,
            planning_required=True,
        )

        self.assertEqual(plan.task_type, "comparison")
        self.assertEqual(
            [item.outcome for item in plan.planner_attempts],
            ["contract_invalid", "validated"],
        )
        retry_message = structured.ainvoke.await_args_list[1].args[0][-1].content
        self.assertIn("FAILURE_CODE=planner_target_selection_invalid", retry_message)
        self.assertTrue(
            all(
                intent.origin == "planned"
                for requirement in plan.requirements
                for intent in requirement.fact_requirements
            )
        )
        self.assertEqual(len(plan.requirements[0].fact_requirements), 1)
        a_anchors = [
            requirement.fact_requirements[0].protected_anchors
            for requirement in plan.requirements[:2]
        ]
        self.assertEqual(
            a_anchors,
            [("core mechanism",), ("input dependency",)],
        )
        self.assertTrue(
            all(
                item.search_query is None
                for requirement in plan.requirements
                for item in requirement.fact_requirements
            )
        )
        self.assertEqual(structured.ainvoke.await_count, 2)
        self.assertIs(
            model.with_structured_output.call_args_list[-1].args[0],
            ComparisonPlanProposal,
        )

    async def test_required_comparison_falls_back_for_unknown_anchor_catalog_ids(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "selected_corpus_ids": ["C001", "T001"],
            "dimension_labels": ["Method"],
            "facts": [
                {
                    "dimension_index": 0,
                    "description": "Paper A core mechanism",
                    "protected_anchor_ids": [99],
                    "retrieval_expansions": ["mechanism"],
                },
            ],
        }
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(
            model,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        plan = await planner.plan(
            "Compare Paper A and Paper B reasoning failures",
            max_steps=2,
            planning_required=True,
        )

        facts = tuple(
            fact
            for requirement in plan.requirements
            for fact in requirement.fact_requirements
        )
        self.assertTrue(all(fact.protected_anchor_fallback for fact in facts))
        self.assertTrue(
            all(
                fact.protected_anchors
                == ("Compare Paper A and Paper B reasoning failures",)
                for fact in facts
            )
        )
        self.assertEqual(structured.ainvoke.await_count, 1)

    async def test_required_comparison_fails_closed_after_two_invalid_plans(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "selected_corpus_ids": ["C001"],
            "dimension_labels": ["Method"],
            "facts": [
                {
                    "dimension_index": 0,
                    "description": "Method",
                    "protected_anchor_ids": [0],
                }
            ],
        }
        model.with_structured_output.return_value = structured

        planner = LangChainResearchPlanner(
            model,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        with self.assertRaises(ComparisonPlanningError) as raised:
            await planner.plan("Compare C001 and T001", max_steps=2, planning_required=True)

        self.assertEqual(
            raised.exception.reason_code,
            "planner_target_selection_invalid",
        )
        self.assertEqual(
            [item.failure_code for item in raised.exception.attempts],
            [
                "planner_target_selection_invalid",
                "planner_target_selection_invalid",
            ],
        )

    async def test_proposal_failure_is_not_reported_as_target_resolution(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "selected_corpus_ids": ["C001", "T001"],
            "dimension_labels": ["Method", "Result"],
            "facts": [
                {
                    "dimension_index": 0,
                    "description": "model-authored fact",
                    "protected_anchor_ids": [0],
                }
            ],
        }
        model.with_structured_output.return_value = structured
        planner = LangChainResearchPlanner(
            model,
            corpus_catalog={"C001": "Paper A", "T001": "Paper B"},
        )

        with self.assertRaises(ComparisonPlanningError) as raised:
            await planner.plan(
                "Compare C001 and T001: method; result.",
                max_steps=4,
                planning_required=True,
            )

        self.assertEqual(
            raised.exception.reason_code,
            "planner_fact_proposal_invalid",
        )
        self.assertNotIsInstance(
            raised.exception,
            ComparisonTargetResolutionError,
        )
        self.assertEqual(
            [item.attempt for item in raised.exception.attempts],
            [1, 2],
        )

    async def test_required_comparison_rejects_targets_outside_resolved_candidates(self) -> None:
        model = Mock()
        structured = AsyncMock()
        structured.ainvoke.return_value = {
            "selected_corpus_ids": ["C001", "T999"],
            "dimension_labels": ["Method"],
            "facts": [
                {
                    "description": "Paper A method",
                    "dimension_index": 0,
                    "protected_anchor_ids": [0],
                },
            ],
        }
        model.with_structured_output.return_value = structured
        resolver = AsyncMock()
        resolver.resolve.return_value = {"C001": "Paper A", "T001": "Paper B"}
        planner = LangChainResearchPlanner(model, target_resolver=resolver)

        with self.assertRaises(ComparisonPlanningError) as raised:
            await planner.plan("Compare the methods", max_steps=2, planning_required=True)

        self.assertEqual(
            raised.exception.reason_code,
            "planner_target_selection_invalid",
        )
        self.assertEqual(structured.ainvoke.await_count, 2)
        retry_message = structured.ainvoke.await_args_list[1].args[0][-1].content
        self.assertIn(
            "FAILURE_CODE=planner_target_selection_invalid",
            retry_message,
        )
        self.assertIn("LOCAL_CORPUS_CATALOG_JSON", retry_message)


if __name__ == "__main__":
    unittest.main()
