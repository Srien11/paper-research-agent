from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from paper_research_agent.agent.orchestrator.control import (
    PlanEdit,
    RunControlCommand,
    RunControlConflict,
)
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ConversationWorkspace,
    GoalState,
    MainAgentResult,
    TaskPlan,
)
from paper_research_agent.conversation.store import (
    InMemoryConversationStore,
    SQLiteConversationStore,
)


@pytest.fixture(params=("sqlite", "memory"))
def store(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "sqlite":
        return SQLiteConversationStore(tmp_path / "control.sqlite3")
    return InMemoryConversationStore()


def _start(store):
    return store.begin_agent_run(
        request_id="request-control-1234",
        conversation_id="conversation-control",
        user_question="执行可编辑计划",
    )


def _planned_workspace(base: ConversationWorkspace) -> ConversationWorkspace:
    now = datetime(2026, 8, 11, tzinfo=UTC)
    goal_id = "a" * 32
    return base.model_copy(
        update={
            "active_goal": GoalState(
                goal_id=goal_id,
                objective="原目标",
                origin_turn_id="b" * 32,
                created_at=now,
                updated_at=now,
            ),
            "task_plan": TaskPlan(
                plan_id="c" * 32,
                goal_id=goal_id,
                revision=1,
                tasks=(
                    AgentTask(
                        task_id="done",
                        goal_id=goal_id,
                        title="完成",
                        objective="保留",
                        success_criteria=("完成",),
                        capability="direct_chat",
                        status="completed",
                        result_ref="artifact:1",
                    ),
                    AgentTask(
                        task_id="failed",
                        goal_id=goal_id,
                        title="失败",
                        objective="重试",
                        success_criteria=("成功",),
                        capability="dynamic_tools",
                        status="failed",
                        depends_on=("done",),
                    ),
                ),
                created_at=now,
                updated_at=now,
            ),
        }
    )


def _force_paused(store, start) -> None:
    workspace = _planned_workspace(start.workspace)
    result = MainAgentResult(
        run_id=start.run_id,
        request_id=start.request_id,
        conversation_id=start.conversation_id,
        status="paused",
        answer="已暂停",
        workspace_version=workspace.version,
    )
    if isinstance(store, SQLiteConversationStore):
        with closing(sqlite3.connect(store.path)) as connection:
            connection.execute(
                "UPDATE conversation_workspaces SET state_json = ? WHERE conversation_id = ?",
                (workspace.model_dump_json(), start.conversation_id),
            )
            connection.execute(
                "UPDATE main_agent_runs SET status = 'paused', result_json = ? WHERE request_id = ?",
                (result.model_dump_json(), start.request_id),
            )
            connection.execute(
                "UPDATE main_agent_controls SET status = 'paused', revision = 1 "
                "WHERE request_id = ?",
                (start.request_id,),
            )
            connection.commit()
        return
    store._workspaces[start.conversation_id] = workspace
    store._runs[start.request_id].status = "paused"
    store._runs[start.request_id].result = result
    store._controls[start.request_id] = store._controls[start.request_id].model_copy(
        update={"status": "paused", "revision": 1}
    )


def test_control_is_persistent_optimistic_and_resumable(store) -> None:
    start = _start(store)
    initial = store.load_agent_control(request_id=start.request_id)
    assert initial is not None
    assert (initial.status, initial.revision) == ("running", 0)

    requested = store.command_agent_run(
        request_id=start.request_id,
        command=RunControlCommand(action="pause", expected_revision=0),
    )
    assert (requested.status, requested.revision) == ("pause_requested", 1)
    with pytest.raises(RunControlConflict, match="revision conflict"):
        store.command_agent_run(
            request_id=start.request_id,
            command=RunControlCommand(action="cancel", expected_revision=0),
        )

    _force_paused(store, start)
    resumed = store.command_agent_run(
        request_id=start.request_id,
        command=RunControlCommand(action="resume", expected_revision=1),
    )
    assert resumed.status == "resuming"
    reopened = store.begin_agent_run(
        request_id=start.request_id,
        conversation_id=start.conversation_id,
        user_question="执行可编辑计划",
    )
    assert reopened.outcome == "resuming"
    assert reopened.run_id == start.run_id


def test_plan_edit_requires_pause_and_preserves_completed_result(store) -> None:
    start = _start(store)
    with pytest.raises(RunControlConflict, match="paused run"):
        store.edit_agent_plan(
            request_id=start.request_id,
            edit=PlanEdit(expected_revision=1, retry_task_ids=("failed",)),
        )

    _force_paused(store, start)
    edited = store.edit_agent_plan(
        request_id=start.request_id,
        edit=PlanEdit(
            expected_revision=1,
            objective="调整后的目标",
            retry_task_ids=("failed",),
        ),
    )
    assert edited.active_goal is not None
    assert edited.active_goal.objective == "调整后的目标"
    assert edited.task_plan is not None
    assert edited.task_plan.revision == 2
    assert edited.task_plan.tasks[0].result_ref == "artifact:1"
    assert edited.task_plan.tasks[0].status == "completed"
    assert edited.task_plan.tasks[1].status == "pending"


def test_cancel_of_paused_run_reopens_graph_to_finalize_cancellation(store) -> None:
    start = _start(store)
    _force_paused(store, start)
    requested = store.command_agent_run(
        request_id=start.request_id,
        command=RunControlCommand(action="cancel", expected_revision=1),
    )
    assert requested.status == "cancel_requested"
    reopened = store.begin_agent_run(
        request_id=start.request_id,
        conversation_id=start.conversation_id,
        user_question="执行可编辑计划",
    )
    assert reopened.outcome == "resuming"
