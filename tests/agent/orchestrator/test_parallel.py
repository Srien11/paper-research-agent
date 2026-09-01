from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ChildTaskRequest,
    ChildTaskResult,
    ConversationWorkspace,
    TaskPlan,
)
from paper_research_agent.agent.orchestrator.parallel import (
    ParallelBatchCoordinator,
    select_next_batch,
)

SCHOLARLY_NAMES = (
    "search_scholarly_sources",
    "resolve_paper_identifier",
    "get_citation_graph",
    "check_paper_status",
)


def _utc() -> datetime:
    return datetime(2026, 9, 1, tzinfo=UTC)


def _task(*, capability: str, task_id: str, status: str = "pending") -> AgentTask:
    values: dict[str, object] = {
        "task_id": task_id,
        "goal_id": "a" * 32,
        "title": task_id,
        "objective": task_id,
        "success_criteria": ("完成",),
        "capability": capability,
        "status": status,
        "parallel_group_id": "hybrid-research",
    }
    if capability == "dynamic_tools":
        values["allowed_tool_risks"] = ("network_read",)
        values["allowed_tool_names"] = SCHOLARLY_NAMES
    return AgentTask(**values)  # type: ignore[arg-type]


def _workspace(*tasks: AgentTask) -> ConversationWorkspace:
    return ConversationWorkspace(
        conversation_id="conversation-1",
        task_plan=TaskPlan(
            plan_id="b" * 32,
            goal_id="a" * 32,
            tasks=tasks,
            created_at=_utc(),
            updated_at=_utc(),
        ),
        updated_at=_utc(),
    )


def _request(task: AgentTask) -> ChildTaskRequest:
    return ChildTaskRequest(
        run_id="run-1",
        request_id="request-1",
        conversation_id="conversation-1",
        turn_id="c" * 32,
        goal_id="a" * 32,
        task_id=task.task_id,
        objective=task.objective,
        success_criteria=task.success_criteria,
        capability=task.capability,
        current_message="结合本地论文和外部学术信息回答",
        rag_mode="preferred",
        parallel_group_id=task.parallel_group_id,
        allowed_tool_risks=task.allowed_tool_risks,
        allowed_tool_names=task.allowed_tool_names,
    )


def _completed(request: ChildTaskRequest) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=f"child-{request.task_id}",
        task_id=request.task_id,
        capability=request.capability,
        status="completed",
        citation_kind="local_paper" if request.capability == "local_rag" else "external",
    )


def test_select_next_batch_returns_parallel_group_in_deterministic_order() -> None:
    dynamic = _task(capability="dynamic_tools", task_id="external-research")
    local = _task(capability="local_rag", task_id="local-research")

    batch = select_next_batch(_workspace(dynamic, local))

    assert batch is not None
    assert batch.execution_mode == "parallel"
    assert batch.task_ids == ("local-research", "external-research")


def test_select_next_batch_returns_only_unfinished_member_for_retry() -> None:
    local = _task(capability="local_rag", task_id="local-research", status="completed")
    dynamic = _task(capability="dynamic_tools", task_id="external-research")

    batch = select_next_batch(_workspace(local, dynamic))

    assert batch is not None
    assert batch.execution_mode == "single"
    assert batch.task_ids == ("external-research",)


def test_parallel_batch_starts_both_before_either_finishes() -> None:
    local = _task(capability="local_rag", task_id="local-research")
    dynamic = _task(capability="dynamic_tools", task_id="external-research")
    batch = select_next_batch(_workspace(local, dynamic))
    assert batch is not None
    requests = (_request(local), _request(dynamic))
    started: set[str] = set()
    both_started = asyncio.Event()

    async def execute(request: ChildTaskRequest) -> ChildTaskResult:
        started.add(request.task_id)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=0.2)
        await asyncio.sleep(0.2)
        return _completed(request)

    async def run() -> tuple[float, tuple[ChildTaskResult, ...]]:
        loop = asyncio.get_running_loop()
        before = loop.time()
        result = await ParallelBatchCoordinator().dispatch(
            batch,
            requests,
            execute,
            remaining_calls=2,
        )
        return loop.time() - before, result.results

    elapsed, results = asyncio.run(run())
    assert started == {"local-research", "external-research"}
    assert elapsed < 0.34
    assert tuple(result.capability for result in results) == ("local_rag", "dynamic_tools")


def test_parallel_batch_twenty_run_p95_performance_gate() -> None:
    local = _task(capability="local_rag", task_id="local-research")
    dynamic = _task(capability="dynamic_tools", task_id="external-research")
    batch = select_next_batch(_workspace(local, dynamic))
    assert batch is not None
    requests = (_request(local), _request(dynamic))

    async def run_gate() -> tuple[list[float], list[float]]:
        wall_times: list[float] = []
        start_gaps: list[float] = []
        coordinator = ParallelBatchCoordinator()
        loop = asyncio.get_running_loop()
        for _ in range(20):
            starts: list[float] = []

            async def execute(
                request: ChildTaskRequest,
                current_starts: list[float] = starts,
            ) -> ChildTaskResult:
                current_starts.append(loop.time())
                await asyncio.sleep(0.2)
                return _completed(request)

            before = loop.time()
            await coordinator.dispatch(
                batch,
                requests,
                execute,
                remaining_calls=2,
            )
            wall_times.append(loop.time() - before)
            start_gaps.append(max(starts) - min(starts))
        return wall_times, start_gaps

    wall_times, start_gaps = asyncio.run(run_gate())
    p95_index = 18
    assert sorted(wall_times)[p95_index] < 0.32
    assert sorted(start_gaps)[p95_index] < 0.05


def test_parallel_branch_failure_does_not_cancel_sibling() -> None:
    local = _task(capability="local_rag", task_id="local-research")
    dynamic = _task(capability="dynamic_tools", task_id="external-research")
    batch = select_next_batch(_workspace(local, dynamic))
    assert batch is not None

    async def execute(request: ChildTaskRequest) -> ChildTaskResult:
        if request.capability == "local_rag":
            raise RuntimeError("private detail")
        await asyncio.sleep(0.01)
        return _completed(request)

    result = asyncio.run(
        ParallelBatchCoordinator().dispatch(
            batch,
            (_request(local), _request(dynamic)),
            execute,
            remaining_calls=2,
        )
    )

    assert result.results[0].status == "failed"
    assert result.results[0].error_code == "parallel_child_failed"
    assert result.results[1].status == "completed"


def test_parallel_batch_rejects_insufficient_call_budget_before_start() -> None:
    local = _task(capability="local_rag", task_id="local-research")
    dynamic = _task(capability="dynamic_tools", task_id="external-research")
    batch = select_next_batch(_workspace(local, dynamic))
    assert batch is not None
    calls = 0

    async def execute(request: ChildTaskRequest) -> ChildTaskResult:
        nonlocal calls
        calls += 1
        return _completed(request)

    with pytest.raises(ValueError, match="call budget"):
        asyncio.run(
            ParallelBatchCoordinator().dispatch(
                batch,
                (_request(local), _request(dynamic)),
                execute,
                remaining_calls=1,
            )
        )
    assert calls == 0
