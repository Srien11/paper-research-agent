"""Deterministic task selection and policy routing for the main Agent."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentTask,
    Capability,
    ConversationWorkspace,
    FrozenModel,
)


class TaskSelection(FrozenModel):
    task_id: str | None = None
    outcome: Literal["execute", "clarify", "finalize", "blocked"]


class RouteDecision(FrozenModel):
    capability: Capability
    reason: str = Field(min_length=1, max_length=200)
    requires_approval: bool = False


_LOCAL_CORPUS_ID = re.compile(r"\b[CT]\d{3}\b", re.IGNORECASE)


def select_next_task(workspace: ConversationWorkspace) -> TaskSelection:
    """Choose the next task with deterministic priority; no model or storage access."""
    plan = workspace.task_plan
    if plan is None or not plan.tasks:
        return TaskSelection(task_id=None, outcome="finalize")
    tasks = plan.tasks
    waiting_approval = next((t for t in tasks if t.status == "waiting_approval"), None)
    if waiting_approval is not None:
        return TaskSelection(task_id=waiting_approval.task_id, outcome="execute")
    running = next((t for t in tasks if t.status == "running"), None)
    if running is not None:
        return TaskSelection(task_id=running.task_id, outcome="execute")
    completed_ids = {t.task_id for t in tasks if t.status == "completed"}
    available = [
        task
        for task in tasks
        if task.status in {"pending", "ready"}
        and all(dependency in completed_ids for dependency in task.depends_on)
    ]
    if available:
        return TaskSelection(task_id=available[0].task_id, outcome="execute")
    if any(task.status == "waiting_user" for task in tasks):
        return TaskSelection(task_id=None, outcome="clarify")
    failed_ids = {t.task_id for t in tasks if t.status == "failed"}
    dependency_blocked = [
        task
        for task in tasks
        if task.status in {"pending", "ready"}
        and any(dependency in failed_ids for dependency in task.depends_on)
    ]
    if dependency_blocked:
        return TaskSelection(task_id=dependency_blocked[0].task_id, outcome="blocked")
    return TaskSelection(task_id=None, outcome="finalize")


def route_task(
    task: AgentTask, envelope: AgentContextEnvelope
) -> RouteDecision:
    """Enforce RAG-mode and attachment policy on a single task; pure function."""
    capability = task.capability
    rag_mode = envelope.rag_mode
    if capability == "attachment_qa":
        if not envelope.attachment_ids:
            return RouteDecision(capability="direct_chat", reason="没有附件可供分析")
        return RouteDecision(capability="attachment_qa", reason="任务需要读取附件")
    if capability == "file_edit":
        if not envelope.attachment_ids:
            return RouteDecision(capability="direct_chat", reason="没有附件可供修改")
        return RouteDecision(capability="file_edit", reason="任务需要修改附件并产出文件")
    if capability == "local_rag":
        if rag_mode == "disabled":
            return RouteDecision(capability="direct_chat", reason="本地检索已禁用")
        return RouteDecision(capability="local_rag", reason="任务需要本地论文证据")
    if capability == "dynamic_tools":
        if rag_mode == "required":
            return RouteDecision(capability="local_rag", reason="required 模式禁止外部动态研究")
        corpus_ids = {item.upper() for item in _LOCAL_CORPUS_ID.findall(task.objective)}
        if (
            rag_mode != "disabled"
            and len(corpus_ids) >= 2
            and requires_research_planning(task.objective)
        ):
            return RouteDecision(
                capability="local_rag",
                reason="本地多论文比较由固定检索图执行",
            )
        return RouteDecision(capability="dynamic_tools", reason="需要最新信息或非论文工具")
    return RouteDecision(capability="direct_chat", reason="任务不需要外部事实")
