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
PlanningRouteReason = Literal[
    "clear_single_local_rag",
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


class SourceRequirements(FrozenModel):
    local_required: bool
    external_required: bool
    reason_codes: tuple[str, ...] = ()


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
    r"(?:联网|网络|网上|外部资料|外部来源|官网|实时|最新|"
    r"\bweb\b|\bonline\b)",
    re.IGNORECASE,
)
_EXPLICIT_LOCAL_SOURCE = re.compile(
    r"(?:本地论文|本地语料|知识库|论文库|私有论文|"
    r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d))",
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


def classify_planning_route(
    envelope: AgentContextEnvelope,
    *,
    enabled: bool,
) -> PlanningRouteDecision:
    """Select fast path only when a single local-paper request is provable."""
    if not enabled:
        return _full("feature_disabled")
    if (
        envelope.workspace.active_goal is not None
        or envelope.workspace.task_plan is not None
    ):
        return _full("existing_workspace")
    if envelope.attachment_ids:
        return _full("attachments_present")

    message = " ".join(envelope.current_message.split())
    if not message:
        return _full("complex_or_ambiguous")
    if len(message) > 1000:
        return _full("contract_bounds_exceeded")
    if infer_source_requirements(message).external_required:
        return _full("complex_or_ambiguous")
    if (
        _FILE_OR_CONTROL.search(message)
        or _EXTERNAL_OR_DYNAMIC.search(message)
        or _MULTI_TASK.search(message)
        or _VAGUE_RESEARCH.fullmatch(message)
    ):
        return _full("complex_or_ambiguous")
    if envelope.rag_mode == "disabled":
        return _full("rag_disabled")

    has_corpus_id = _LOCAL_CORPUS_ID.search(message) is not None
    is_comparison = requires_research_planning(message)
    has_explicit_multi_object = _EXPLICIT_MULTI_OBJECT.search(message) is not None
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
        )
    if envelope.rag_mode == "preferred" and (
        has_corpus_id or (is_comparison and has_explicit_multi_object)
    ):
        return PlanningRouteDecision(
            route="fast_path",
            reason_code="clear_single_local_rag",
        )
    return _full("complex_or_ambiguous")


def infer_source_requirements(message: str) -> SourceRequirements:
    """Identify explicit local and external evidence requirements without planning tasks."""
    normalized = " ".join(message.split())
    local_required = _EXPLICIT_LOCAL_SOURCE.search(normalized) is not None
    external_required = _EXPLICIT_EXTERNAL_SOURCE.search(normalized) is not None
    reasons: list[str] = []
    if local_required:
        reasons.append("explicit_local_source")
    if external_required:
        reasons.append("explicit_external_source")
    return SourceRequirements(
        local_required=local_required,
        external_required=external_required,
        reason_codes=tuple(reasons),
    )


def _full(reason_code: PlanningRouteReason) -> PlanningRouteDecision:
    return PlanningRouteDecision(route="full_planner", reason_code=reason_code)
