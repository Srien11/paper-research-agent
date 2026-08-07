from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from paper_research_agent.agent.models import ResearchPlan, ResearchStep
from paper_research_agent.agent.planner import LangChainResearchPlanner


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
        planner = LangChainResearchPlanner(model)

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
                    {"target_id": "a", "label": "Paper A"},
                    {"target_id": "b", "label": "Paper B"},
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
                        "target_ids": ["a"],
                        "dimension_ids": ["method"],
                    },
                    {
                        "step_id": "b",
                        "objective": "Paper B method",
                        "query": "Paper B method",
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


if __name__ == "__main__":
    unittest.main()
