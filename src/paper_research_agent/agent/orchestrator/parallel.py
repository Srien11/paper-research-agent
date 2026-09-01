"""Pure batch selection and structured concurrency for main-Agent child tasks."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from paper_research_agent.agent.orchestrator.models import (
    ChildTaskRequest,
    ChildTaskResult,
    ConversationWorkspace,
    FrozenModel,
)
from paper_research_agent.agent.orchestrator.router import select_next_task


class TaskBatch(FrozenModel):
    batch_id: str = Field(min_length=1, max_length=128)
    task_ids: tuple[str, ...] = Field(min_length=1, max_length=2)
    execution_mode: Literal["single", "parallel"]

    @model_validator(mode="after")
    def validate_mode(self) -> TaskBatch:
        expected = "parallel" if len(self.task_ids) == 2 else "single"
        if self.execution_mode != expected:
            raise ValueError("task batch execution mode does not match task count")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task batch IDs must be unique")
        return self


class BatchDispatchResult(FrozenModel):
    batch: TaskBatch
    results: tuple[ChildTaskResult, ...] = Field(min_length=1, max_length=2)
    elapsed_seconds: float = Field(ge=0)


ChildExecutor = Callable[[ChildTaskRequest], Awaitable[ChildTaskResult]]


def select_next_batch(workspace: ConversationWorkspace) -> TaskBatch | None:
    """Select one ready task or one complete two-member parallel group."""
    selection = select_next_task(workspace)
    if selection.outcome != "execute" or selection.task_id is None:
        return None
    plan = workspace.task_plan
    if plan is None:
        return None
    selected = next(task for task in plan.tasks if task.task_id == selection.task_id)
    group_id = selected.parallel_group_id
    if group_id is None:
        return TaskBatch(
            batch_id=f"single::{selected.task_id}",
            task_ids=(selected.task_id,),
            execution_mode="single",
        )
    members = tuple(task for task in plan.tasks if task.parallel_group_id == group_id)
    if any(task.status == "waiting_approval" for task in members):
        raise ValueError("parallel group must not contain a waiting approval task")
    completed_ids = {task.task_id for task in plan.tasks if task.status == "completed"}
    ready = tuple(
        task
        for task in members
        if task.status in {"pending", "ready"}
        and all(dependency in completed_ids for dependency in task.depends_on)
    )
    if not ready:
        return None
    if len(ready) == 1:
        task = ready[0]
        return TaskBatch(
            batch_id=f"single::{task.task_id}",
            task_ids=(task.task_id,),
            execution_mode="single",
        )
    ordered = tuple(
        task.task_id
        for capability in ("local_rag", "dynamic_tools")
        for task in ready
        if task.capability == capability
    )
    return TaskBatch(
        batch_id=group_id,
        task_ids=ordered,
        execution_mode="parallel",
    )


class ParallelBatchCoordinator:
    """Run one batch with sibling failure isolation and outer cancellation propagation."""

    async def dispatch(
        self,
        batch: TaskBatch,
        requests: Sequence[ChildTaskRequest],
        execute: ChildExecutor,
        *,
        remaining_calls: int,
        timeout_seconds: Mapping[str, float | None] | None = None,
    ) -> BatchDispatchResult:
        if remaining_calls < len(batch.task_ids):
            raise ValueError("insufficient child call budget for task batch")
        by_id = {request.task_id: request for request in requests}
        if len(by_id) != len(requests) or set(by_id) != set(batch.task_ids):
            raise ValueError("batch requests must match task IDs exactly")
        ordered = tuple(by_id[task_id] for task_id in batch.task_ids)
        timeouts = timeout_seconds or {}
        started = time.perf_counter()
        results: tuple[ChildTaskResult, ...]
        if batch.execution_mode == "single":
            result = await _run_one(
                ordered[0],
                execute,
                timeout_seconds=timeouts.get(ordered[0].task_id),
            )
            results = (result,)
        else:
            async with asyncio.TaskGroup() as group:
                pending = {
                    request.task_id: group.create_task(
                        _run_one(
                            request,
                            execute,
                            timeout_seconds=timeouts.get(request.task_id),
                        )
                    )
                    for request in ordered
                }
            results = tuple(pending[task_id].result() for task_id in batch.task_ids)
        return BatchDispatchResult(
            batch=batch,
            results=results,
            elapsed_seconds=time.perf_counter() - started,
        )


async def _run_one(
    request: ChildTaskRequest,
    execute: ChildExecutor,
    *,
    timeout_seconds: float | None,
) -> ChildTaskResult:
    try:
        if timeout_seconds is None:
            return await execute(request)
        async with asyncio.timeout(timeout_seconds):
            return await execute(request)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return _failed_result(request, "parallel_child_timeout")
    except Exception:  # noqa: BLE001 - branch isolation deliberately hides private errors
        return _failed_result(request, "parallel_child_failed")


def _failed_result(request: ChildTaskRequest, error_code: str) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=f"{request.run_id}:{request.task_id}",
        task_id=request.task_id,
        capability=request.capability,
        status="failed",
        summary="研究分支执行失败。",
        error_code=error_code,
    )
