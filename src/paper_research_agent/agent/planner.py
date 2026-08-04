"""LangChain structured-output adapter for bounded research planning."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from paper_research_agent.agent.models import ResearchPlan


class LangChainResearchPlanner:
    """Ask a chat model only for ordered corpus-search subquestions."""

    def __init__(self, model: BaseChatModel):
        self._structured_model = model.with_structured_output(
            ResearchPlan,
            method="function_calling",
        )

    async def plan(self, question: str, *, max_steps: int) -> ResearchPlan:
        if max_steps <= 0 or max_steps > 6:
            raise ValueError("max_steps must be between 1 and 6")
        system = SystemMessage(
            content=(
                "你是论文研究任务规划器。只拆分需要在现有本地论文库中检索的子问题，"
                f"最多 {max_steps} 步。每一步必须给出稳定 step_id、简短目标、可独立检索的"
                "查询和 1 到 20 的 top_k。不要回答问题，不要请求网络、代码执行、文件写入"
                "或数据库修改。"
            )
        )
        raw = await self._structured_model.ainvoke([system, HumanMessage(content=question)])
        plan = ResearchPlan.model_validate(raw)
        if len(plan.steps) > max_steps:
            raise ValueError("research plan exceeds the requested step budget")
        return plan
