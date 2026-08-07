"""LangChain structured-output adapter for bounded research planning."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from paper_research_agent.agent.models import ResearchPlan, ResearchStep


class LangChainResearchPlanner:
    """Ask a chat model only for ordered corpus-search subquestions."""

    def __init__(self, model: BaseChatModel):
        self._structured_model = model.with_structured_output(
            ResearchPlan,
            method="function_calling",
        )

    async def plan(
        self,
        question: str,
        *,
        max_steps: int,
        planning_required: bool = False,
    ) -> ResearchPlan:
        if max_steps <= 0 or max_steps > 6:
            raise ValueError("max_steps must be between 1 and 6")
        system = SystemMessage(
            content=(
                "你是论文研究任务规划器。只拆分需要在现有本地论文库中检索的子问题，"
                f"最多 {max_steps} 步。每一步必须给出稳定 step_id、简短目标、可独立检索的"
                "查询和 1 到 20 的 top_k。若问题要求比较两个或更多论文、方法、模型或实验，"
                "必须使用 task_type=comparison：明确列出比较对象 targets 和比较维度 dimensions，"
                "为每个比较对象×维度建立完整证据网格 requirements，并生成逐对象、可独立命中的"
                "检索步骤；禁止用一个宽泛查询代替所有对象的检索。每个 comparison step 必须填写"
                "对应 target_ids 和 dimension_ids，所有 requirement 单元都必须至少由一步覆盖。"
                "非比较问题使用 task_type=direct，且不填写比较元数据。不要回答问题，不要请求"
                "网络、代码执行、文件写入或数据库修改。"
                + (
                    "上游语义路由已确认本题必须进行多对象研究规划，因此必须返回 "
                    "task_type=comparison，不能降为 direct。"
                    if planning_required
                    else ""
                )
            )
        )
        messages = [system, HumanMessage(content=question)]
        for attempt in range(2):
            try:
                raw = await self._structured_model.ainvoke(messages)
                plan = ResearchPlan.model_validate(raw)
                if len(plan.steps) > max_steps:
                    raise ValueError("research plan exceeds the requested step budget")
                if planning_required and plan.task_type != "comparison":
                    raise ValueError("planned research requires a comparison plan")
                return plan
            except ValueError:
                if attempt == 0:
                    messages = [
                        *messages,
                        HumanMessage(
                            content=(
                                "The previous plan violated the structured planning contract. "
                                "Return a valid full target-by-dimension comparison grid within "
                                "the step budget."
                            )
                        ),
                    ]

        return ResearchPlan(
            steps=(
                ResearchStep(
                    step_id="fallback",
                    objective="Retrieve evidence for the requested research question",
                    query=question,
                    top_k=10,
                ),
            )
        )
