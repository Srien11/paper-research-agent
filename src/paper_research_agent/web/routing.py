"""Server-side intent routing and policy enforcement for unified chat requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.agent.intent import requires_research_planning

RouteKind = Literal[
    "normal_chat",
    "local_rag",
    "web_research",
    "attachment_qa",
    "file_edit",
]
RAGMode = Literal["disabled", "preferred", "required"]
ResearchMode = Literal["single", "planned"]

_PURE_SOCIAL_UTTERANCES = frozenset(
    {
        "hi",
        "hello",
        "你好",
        "您好",
        "嗨",
        "谢谢",
        "感谢",
        "再见",
        "拜拜",
    }
)


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: RouteKind
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=160)
    research_mode: ResearchMode = "single"


class CapabilityPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: RouteKind
    use_local_papers: bool = False
    use_web_research: bool = False
    use_dynamic_tools: bool = False
    use_attachments: bool = False
    research_mode: ResearchMode = "single"
    reason: str = Field(min_length=1, max_length=160)

    def enforce(self, context: RouteContext) -> CapabilityPlan:
        route = self.route
        local = self.use_local_papers
        web = self.use_web_research
        dynamic = self.use_dynamic_tools
        attachments = self.use_attachments
        research_mode = self.research_mode
        reason = self.reason

        if context.has_attachments:
            attachments = True
            if route not in {"attachment_qa", "file_edit"}:
                route = "attachment_qa"
                reason = "附件存在，策略层保留附件处理能力"
        else:
            attachments = False
            if route in {"attachment_qa", "file_edit"}:
                route = "local_rag" if local else "normal_chat"
                reason = "没有附件，策略层拒绝附件操作"

        if context.rag_mode == "required":
            if not context.rag_available:
                raise RuntimeError("local RAG is required but unavailable")
            route = "local_rag"
            local = True
            web = False
            dynamic = False
            research_mode = "planned" if requires_research_planning(context.question) else "single"
            reason = "用户明确要求仅使用本地论文库"
        elif context.rag_mode == "disabled":
            local = False
            if route == "local_rag":
                route = "web_research" if web or dynamic else "normal_chat"
                reason = "用户已关闭本地论文库"
        elif context.rag_mode == "preferred" and context.rag_available:
            if not attachments and _is_pure_social_utterance(context.question):
                route = "normal_chat"
                local = False
                web = False
                dynamic = False
                research_mode = "single"
                reason = "纯寒暄无需调用研究能力"
            else:
                # Preferred is a hybrid evidence policy: local retrieval is mandatory,
                # but its evidence does not become the exclusive answer boundary.
                local = True
                if route == "local_rag":
                    route = "normal_chat"
                    reason = "参考本地论文并结合通用能力回答"

        # The current Web research lane is implemented by the bounded dynamic-tool graph.
        if web:
            dynamic = True

        if local and not context.rag_available:
            local = False
            if route == "local_rag":
                route = "web_research" if web or dynamic else "normal_chat"
            reason = "本地论文库不可用，移除本地检索能力"
        if (web or dynamic) and not context.web_available:
            web = False
            dynamic = False
            if route == "web_research":
                route = "local_rag" if local else "normal_chat"
            reason = "动态研究能力不可用，移除外部研究能力"

        if research_mode == "planned" and not context.research_planning_available and not dynamic:
            research_mode = "single"
        return CapabilityPlan(
            route=route,
            use_local_papers=local,
            use_web_research=web,
            use_dynamic_tools=dynamic,
            use_attachments=attachments,
            research_mode=research_mode,
            reason=reason,
        )


@dataclass(frozen=True)
class RouteContext:
    has_attachments: bool
    rag_mode: RAGMode
    rag_available: bool
    web_available: bool
    question: str = ""
    research_planning_available: bool = False


def _is_pure_social_utterance(question: str) -> bool:
    normalized = question.strip().casefold().rstrip("。！？!?~～")
    return normalized in _PURE_SOCIAL_UTTERANCES


ROUTE_LABELS: dict[RouteKind, str] = {
    "normal_chat": "普通交流",
    "local_rag": "本地论文检索",
    "web_research": "联网研究",
    "attachment_qa": "附件问答",
    "file_edit": "文件修改",
}


def enforce_route_policy(decision: RouteDecision, context: RouteContext) -> RouteDecision:
    """Constrain model output to capabilities and explicit user restrictions."""
    route = decision.route
    reason = decision.reason
    research_mode = decision.research_mode

    if context.has_attachments:
        if route not in {"attachment_qa", "file_edit"}:
            route = "attachment_qa"
            reason = "附件存在，策略层限制为只读附件问答"
    elif route in {"attachment_qa", "file_edit"}:
        route = "local_rag" if context.rag_mode == "required" else "normal_chat"
        reason = "没有附件，策略层拒绝文件操作"

    if context.rag_mode == "required" and not context.has_attachments:
        if not context.rag_available:
            raise RuntimeError("local RAG is required but unavailable")
        route = "local_rag"
        reason = "用户明确要求仅使用本地论文库"

    if context.rag_mode == "disabled" and route == "local_rag":
        route = "normal_chat"
        reason = "用户已关闭本地论文库，策略层禁止本地检索"

    if (
        context.rag_mode == "preferred"
        and not context.has_attachments
        and context.rag_available
        and route == "normal_chat"
        and requires_research_planning(context.question)
    ):
        route = "local_rag"
        reason = "比较研究需要论文证据，策略层启用本地研究"
        research_mode = "planned"

    if route == "local_rag" and not context.rag_available:
        route = "normal_chat"
        reason = "本地论文库不可用，策略层回退普通交流"
    if route == "web_research" and not context.web_available:
        route = "normal_chat"
        reason = "联网研究能力不可用，策略层回退普通交流"

    if route != "local_rag":
        research_mode = "single"
    elif requires_research_planning(context.question):
        research_mode = "planned"
    if research_mode == "planned" and not context.research_planning_available:
        research_mode = "single"
        reason = "研究规划能力不可用，保留本地单轮检索"

    return RouteDecision(
        route=route,
        confidence=decision.confidence,
        reason=reason,
        research_mode=research_mode,
    )
