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

        model.with_structured_output.assert_called_once_with(ResearchPlan)
        self.assertEqual(plan.steps[0].query, "RAG evaluation methods")
        messages = structured.ainvoke.await_args.args[0]
        self.assertIn("最多 3", messages[0].content)
        self.assertEqual(messages[1].content, "比较 RAG 评测方法")

    async def test_rejects_model_plan_above_requested_budget(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "step budget"):
            await planner.plan("问题", max_steps=1)


if __name__ == "__main__":
    unittest.main()
