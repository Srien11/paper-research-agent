"""Server-side intent routing and policy enforcement for unified chat requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RouteKind = Literal[
    "normal_chat",
    "local_rag",
    "web_research",
    "attachment_qa",
    "file_edit",
]


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: RouteKind
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=160)


@dataclass(frozen=True)
class RouteContext:
    has_attachments: bool
    local_only: bool
    rag_available: bool
    web_available: bool


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

    if context.has_attachments:
        if route not in {"attachment_qa", "file_edit"}:
            route = "attachment_qa"
            reason = "附件存在，策略层限制为只读附件问答"
    elif route in {"attachment_qa", "file_edit"}:
        route = "local_rag" if context.local_only else "normal_chat"
        reason = "没有附件，策略层拒绝文件操作"

    if context.local_only and not context.has_attachments:
        if not context.rag_available:
            raise RuntimeError("local RAG is required but unavailable")
        route = "local_rag"
        reason = "用户明确要求仅使用本地论文库"

    if route == "local_rag" and not context.rag_available:
        route = "normal_chat"
        reason = "本地论文库不可用，策略层回退普通交流"
    if route == "web_research" and not context.web_available:
        route = "normal_chat"
        reason = "联网研究能力不可用，策略层回退普通交流"

    return RouteDecision(route=route, confidence=decision.confidence, reason=reason)
