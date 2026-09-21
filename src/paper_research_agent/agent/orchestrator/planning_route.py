"""Deterministic allowlist for choosing the main Agent planning path."""

from __future__ import annotations

import re
from typing import Literal

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    FrozenModel,
)

PlanningRoute = Literal["fast_path", "full_planner"]
RAGMode = Literal["disabled", "preferred", "required"]
PlanningRouteReason = Literal[
    "clear_single_local_rag",
    "simple_direct_chat",
    "feature_disabled",
    "existing_workspace",
    "attachments_present",
    "rag_disabled",
    "contract_bounds_exceeded",
    "complex_or_ambiguous",
]


class PlanningRouteDecision(FrozenModel):
    route: PlanningRoute
    reason_code: PlanningRouteReason
    capability: Literal["direct_chat", "local_rag"] | None = None


class SourceRequirements(FrozenModel):
    local_required: bool
    external_required: bool
    local_forbidden: bool = False
    external_forbidden: bool = False
    reason_codes: tuple[str, ...] = ()


class SourcePolicyConflictError(ValueError):
    """An explicit RAG mode contradicts an explicit source request."""

    def __init__(self, reason_code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.reason_code = reason_code
        self.public_message = public_message


_LOCAL_CORPUS_ID = re.compile(r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d)", re.IGNORECASE)
_LOCAL_RESEARCH_OBJECT = re.compile(
    r"(?:本地论文|论文|研究|方法|模型|算法|实验|数据集|指标|架构|"
    r"\bpaper\b|\bstudy\b|\bresearch\b|\bmethod\b|\bmodel\b)",
    re.IGNORECASE,
)
_EXPLICIT_MULTI_OBJECT = re.compile(
    r"(?:两篇|两个(?:论文|研究|方法|模型)|多篇|多个(?:论文|研究|方法|模型)|"
    r"一篇.{0,160}另一篇|一种.{0,160}另一种|这些论文|上述论文|"
    r"\btwo\s+(?:papers|studies|methods|models)\b|\bthese papers\b)",
    re.IGNORECASE,
)
_FILE_OR_CONTROL = re.compile(
    r"(?:修改|编辑|写入|保存|删除|移动|重命名|上传|下载).{0,16}(?:文件|报告|目录)|"
    r"(?:继续|恢复|resume|取消|撤销|修改目标|成功标准|约束|批准|拒绝|审批)",
    re.IGNORECASE,
)
_EXTERNAL_OR_DYNAMIC = re.compile(
    r"(?:最新|实时|今天|当前).{0,20}(?:网页|网站|官网|互联网|状态|新闻|价格)|"
    r"(?:网页|网站|互联网|web|online).{0,20}(?:搜索|查询|检索)|"
    r"(?:运行命令|调用工具|调用\s*API|动态工具|发送消息|发送邮件)",
    re.IGNORECASE,
)
_EXPLICIT_EXTERNAL_SOURCE = re.compile(
    r"(?:联网|网上|在线)(?:搜索|查询|查找|检索|研究|核验|确认)?|"
    r"外部(?:学术)?(?:搜索|查询|检索|研究|核验|资料|来源)|"
    r"(?:搜索|查询|查找|检索|核验|确认|参考|使用).{0,8}(?:网络|外部资料|外部来源|官网)|"
    r"(?:外部资料|外部来源|网络资料|网络来源|官网).{0,8}"
    r"(?:搜索|查询|查找|检索|核验|确认|资料|信息)|"
    r"\b(?:web|online)\s+(?:search|research|sources?)\b",
    re.IGNORECASE,
)
_EXPLICIT_LOCAL_SOURCE = re.compile(
    r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d)|"
    r"(?:使用|用|参考|查询|检索|搜索|根据|基于|结合|调用|来自|从).{0,8}"
    r"(?:本地论文|本地语料|知识库|论文库|私有论文)|"
    r"(?:本地论文|本地语料|知识库|论文库|私有论文).{0,8}"
    r"(?:回答|查找|查询|检索|搜索|证据|资料|内容|论文|中)",
    re.IGNORECASE,
)
_LOCAL_SOURCE_FORBIDDEN = re.compile(
    r"(?:(?:不要|别)(?:再)?(?:使用|检索|参考|查询)?|不使用|不用|无需(?:使用)?|"
    r"禁止(?:使用)?|避免(?:使用)?)\s*(?:本地论文|本地语料|知识库|论文库|私有论文|"
    r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d))|"
    r"(?:do\s+not|don't|without|avoid)\s+(?:using\s+)?(?:the\s+)?"
    r"(?:local\s+(?:papers?|corpus)|knowledge\s+base)",
    re.IGNORECASE,
)
_EXTERNAL_SOURCE_FORBIDDEN = re.compile(
    r"(?:(?:不要|别)(?:再)?(?:使用|搜索|查询|检索)?|不使用|不用|无需(?:使用)?|"
    r"禁止(?:使用)?|避免(?:使用)?)\s*(?:联网|网络|网上|外部资料|外部来源|官网)|"
    r"(?:do\s+not|don't|without|avoid)\s+(?:using\s+)?(?:the\s+)?"
    r"(?:web|internet|online|external\s+sources?)",
    re.IGNORECASE,
)
_MULTI_TASK = re.compile(
    r"(?:两个|多个|分别).{0,16}(?:任务|输出|文件|报告)|"
    r"(?:先|首先).{0,80}(?:再|然后|接着)|"
    r"(?:然后|接着).{0,40}(?:生成|创建|修改|发送|保存)",
    re.IGNORECASE,
)
_VAGUE_RESEARCH = re.compile(
    r"^(?:请)?(?:帮我)?(?:研究|查|看看|分析)(?:一下)?[。.!！]?$",
    re.IGNORECASE,
)
_CONTEXTUAL_FOLLOW_UP = re.compile(
    r"^(?:再|继续|接着|然后|进一步|补充|展开|详细说说|那|那么|为什么|怎么会)|"
    r"(?:上面|刚才|前面|此前|之前|这个|这些|上述|它们|其)(?:的|里|中|呢|吗|如何|为什么)",
    re.IGNORECASE,
)
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "skipped", "cancelled"})


def classify_planning_route(
    envelope: AgentContextEnvelope,
    *,
    enabled: bool,
) -> PlanningRouteDecision:
    """Select a deterministic single-task path when full planning adds no value."""
    if not enabled:
        return _full("feature_disabled")
    if envelope.attachment_ids:
        return _full("attachments_present")

    message = " ".join(envelope.current_message.split())
    if not message:
        return _full("complex_or_ambiguous")
    if len(message) > 1000:
        return _full("contract_bounds_exceeded")
    requirements = infer_source_requirements(message)
    if requirements.external_required:
        return _full("complex_or_ambiguous")
    if envelope.rag_mode == "disabled" and requirements.local_required:
        return _full("rag_disabled")
    if (
        _FILE_OR_CONTROL.search(message)
        or _EXTERNAL_OR_DYNAMIC.search(message)
        or _MULTI_TASK.search(message)
        or _VAGUE_RESEARCH.fullmatch(message)
    ):
        return _full("complex_or_ambiguous")
    has_corpus_id = _LOCAL_CORPUS_ID.search(message) is not None
    is_comparison = requires_research_planning(message)
    has_explicit_multi_object = _EXPLICIT_MULTI_OBJECT.search(message) is not None
    if _workspace_needs_contextual_planning(envelope, message):
        return _full("existing_workspace")
    if is_comparison and not (has_corpus_id or has_explicit_multi_object):
        return _full("complex_or_ambiguous")
    if envelope.rag_mode == "required" and (
        has_corpus_id
        or is_comparison
        or _LOCAL_RESEARCH_OBJECT.search(message) is not None
    ):
        return PlanningRouteDecision(
            route="fast_path",
            reason_code="clear_single_local_rag",
            capability="local_rag",
        )
    if envelope.rag_mode == "preferred" and (
        requirements.local_required
        or has_corpus_id
        or (is_comparison and has_explicit_multi_object)
    ):
        return PlanningRouteDecision(
            route="fast_path",
            reason_code="clear_single_local_rag",
            capability="local_rag",
        )
    if envelope.rag_mode in {"disabled", "preferred"}:
        return PlanningRouteDecision(
            route="fast_path",
            reason_code="simple_direct_chat",
            capability="direct_chat",
        )
    return _full("complex_or_ambiguous")


def _workspace_needs_contextual_planning(
    envelope: AgentContextEnvelope,
    message: str,
) -> bool:
    """Keep active work and context-dependent follow-ups on the full planner."""
    workspace = envelope.workspace
    if workspace.active_goal is None and workspace.task_plan is None:
        return False
    plan = workspace.task_plan
    if plan is None or any(
        task.status not in _TERMINAL_TASK_STATUSES for task in plan.tasks
    ):
        return True
    return _CONTEXTUAL_FOLLOW_UP.search(message) is not None


def infer_source_requirements(message: str) -> SourceRequirements:
    """Identify explicit local and external evidence requirements without planning tasks."""
    normalized = " ".join(message.split())
    local_forbidden = _LOCAL_SOURCE_FORBIDDEN.search(normalized) is not None
    external_forbidden = _EXTERNAL_SOURCE_FORBIDDEN.search(normalized) is not None
    local_required = (
        _EXPLICIT_LOCAL_SOURCE.search(normalized) is not None and not local_forbidden
    )
    external_required = (
        _EXPLICIT_EXTERNAL_SOURCE.search(normalized) is not None and not external_forbidden
    )
    reasons: list[str] = []
    if local_required:
        reasons.append("explicit_local_source")
    if external_required:
        reasons.append("explicit_external_source")
    if local_forbidden:
        reasons.append("explicit_local_source_forbidden")
    if external_forbidden:
        reasons.append("explicit_external_source_forbidden")
    return SourceRequirements(
        local_required=local_required,
        external_required=external_required,
        local_forbidden=local_forbidden,
        external_forbidden=external_forbidden,
        reason_codes=tuple(reasons),
    )


def validate_source_policy(message: str, rag_mode: RAGMode) -> SourceRequirements:
    """Reject language that contradicts an explicit hard RAG mode."""
    requirements = infer_source_requirements(message)
    if rag_mode == "disabled" and requirements.local_required:
        raise SourcePolicyConflictError(
            "rag_disabled_local_requested",
            "当前已关闭本地论文库，但问题明确要求使用知识库；请开启“使用本地论文知识库”后重试。",
        )
    if rag_mode == "required" and requirements.local_forbidden:
        raise SourcePolicyConflictError(
            "rag_required_local_forbidden",
            "当前为“仅依据本地论文回答”，但问题明确要求不使用知识库；请关闭该选项后重试。",
        )
    if rag_mode == "required" and requirements.external_required:
        raise SourcePolicyConflictError(
            "rag_required_external_requested",
            "当前为“仅依据本地论文回答”，但问题明确要求联网或外部资料；请关闭该选项后重试。",
        )
    return requirements


def _full(reason_code: PlanningRouteReason) -> PlanningRouteDecision:
    return PlanningRouteDecision(route="full_planner", reason_code=reason_code)
