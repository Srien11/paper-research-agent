from __future__ import annotations

import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentTask,
    ConversationWorkspace,
    GoalState,
    TaskPlan,
)
from paper_research_agent.agent.orchestrator.router import (
    CAPABILITIES,
    RouteDecision,
    route_task,
    select_next_task,
)


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal() -> GoalState:
    return GoalState(
        goal_id="a" * 32,
        objective="比较 RAG 与 GraphRAG",
        status="active",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )


def _task(**overrides: object) -> AgentTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "goal_id": "a" * 32,
        "title": "任务",
        "objective": "完成目标",
        "success_criteria": ("完成",),
        "capability": "local_rag",
        "status": "pending",
        "depends_on": (),
    }
    values.update(overrides)
    return AgentTask(**values)


def _plan(tasks: tuple[AgentTask, ...], **overrides: object) -> TaskPlan:
    values: dict[str, object] = {
        "plan_id": "c" * 32,
        "goal_id": "a" * 32,
        "revision": 1,
        "tasks": tasks,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return TaskPlan(**values)


def _workspace(**overrides: object) -> ConversationWorkspace:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "version": 0,
        "active_goal": _goal(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return ConversationWorkspace(**values)


def _envelope(**overrides: object) -> AgentContextEnvelope:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "request_id": "request-1",
        "turn_id": "d" * 32,
        "current_message": "继续",
        "rag_mode": "preferred",
        "workspace": _workspace(),
        "prepared_at": _utc(),
    }
    values.update(overrides)
    return AgentContextEnvelope(**values)


class SelectNextTaskTests(unittest.TestCase):
    def test_waiting_approval_has_priority(self) -> None:
        workspace = _workspace(
            task_plan=_plan(
                tasks=(
                    _task(task_id="ready-task", capability="local_rag", status="ready"),
                    _task(
                        task_id="approval-task",
                        capability="dynamic_tools",
                        status="waiting_approval",
                    ),
                )
            )
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "execute")
        self.assertEqual(selection.task_id, "approval-task")

    def test_running_task_continues(self) -> None:
        workspace = _workspace(
            task_plan=_plan(
                tasks=(
                    _task(task_id="running-task", status="running"),
                    _task(task_id="pending-task", status="pending"),
                )
            )
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "execute")
        self.assertEqual(selection.task_id, "running-task")

    def test_pending_with_completed_dependencies_is_selected(self) -> None:
        workspace = _workspace(
            task_plan=_plan(
                tasks=(
                    _task(task_id="t1", status="completed"),
                    _task(task_id="t2", status="pending", depends_on=("t1",)),
                    _task(task_id="t3", status="pending", depends_on=("t2",)),
                )
            )
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.task_id, "t2")
        self.assertEqual(selection.outcome, "execute")

    def test_waiting_user_returns_clarify(self) -> None:
        workspace = _workspace(
            task_plan=_plan(tasks=(_task(task_id="t1", status="waiting_user"),))
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "clarify")

    def test_all_terminal_finalizes(self) -> None:
        workspace = _workspace(
            task_plan=_plan(
                tasks=(
                    _task(task_id="t1", status="completed"),
                    _task(task_id="t2", status="cancelled"),
                    _task(task_id="t3", status="skipped"),
                )
            )
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "finalize")

    def test_failed_dependency_blocks_task(self) -> None:
        workspace = _workspace(
            task_plan=_plan(
                tasks=(
                    _task(task_id="t1", status="failed"),
                    _task(task_id="t2", status="pending", depends_on=("t1",)),
                )
            )
        )
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "blocked")
        self.assertEqual(selection.task_id, "t2")

    def test_empty_plan_finalizes(self) -> None:
        workspace = _workspace(task_plan=_plan(tasks=()))
        selection = select_next_task(workspace)
        self.assertEqual(selection.outcome, "finalize")


class RouteTaskTests(unittest.TestCase):
    def test_all_main_agent_capabilities_are_dispatchable(self) -> None:
        self.assertEqual(
            CAPABILITIES,
            {
                "direct_chat",
                "local_rag",
                "dynamic_tools",
                "attachment_qa",
                "file_edit",
            },
        )

    def test_attachment_qa_requires_attachment(self) -> None:
        task = _task(task_id="att", capability="attachment_qa")
        decision = route_task(task, _envelope(attachment_ids=("file-1",)))
        self.assertEqual(decision.capability, "attachment_qa")
        no_attachment = route_task(task, _envelope(attachment_ids=()))
        self.assertEqual(no_attachment.capability, "direct_chat")

    def test_file_edit_requires_attachment(self) -> None:
        task = _task(task_id="edit", capability="file_edit")
        decision = route_task(task, _envelope(attachment_ids=("file-1",)))
        self.assertEqual(decision.capability, "file_edit")
        no_attachment = route_task(task, _envelope(attachment_ids=()))
        self.assertEqual(no_attachment.capability, "direct_chat")

    def test_local_rag_policy_by_rag_mode(self) -> None:
        task = _task(task_id="local", capability="local_rag")
        self.assertEqual(
            route_task(task, _envelope(rag_mode="preferred")).capability, "local_rag"
        )
        self.assertEqual(
            route_task(task, _envelope(rag_mode="required")).capability, "local_rag"
        )
        self.assertEqual(
            route_task(task, _envelope(rag_mode="disabled")).capability, "direct_chat"
        )

    def test_dynamic_tools_forbidden_in_required_mode(self) -> None:
        task = _task(task_id="web", capability="dynamic_tools")
        self.assertEqual(
            route_task(task, _envelope(rag_mode="required")).capability, "local_rag"
        )
        self.assertEqual(
            route_task(task, _envelope(rag_mode="preferred")).capability, "dynamic_tools"
        )
        self.assertEqual(
            route_task(task, _envelope(rag_mode="disabled")).capability, "dynamic_tools"
        )

    def test_local_corpus_comparison_is_forced_to_fixed_rag_graph(self) -> None:
        task = _task(
            capability="dynamic_tools",
            objective="比较 C001 与 T001 两篇论文的评测指标",
        )

        decision = route_task(task, _envelope(rag_mode="preferred"))

        self.assertEqual(decision.capability, "local_rag")
        self.assertIn("固定检索图", decision.reason)

    def test_direct_chat_needs_no_external_facts(self) -> None:
        task = _task(task_id="chat", capability="direct_chat")
        decision = route_task(task, _envelope())
        self.assertEqual(decision.capability, "direct_chat")

    def test_single_task_single_capability(self) -> None:
        task = _task(task_id="one", capability="direct_chat")
        decision = route_task(task, _envelope())
        self.assertIsInstance(decision, RouteDecision)
        self.assertIsInstance(decision.capability, str)


if __name__ == "__main__":
    unittest.main()
