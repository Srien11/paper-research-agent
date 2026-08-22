"""Main Agent LangGraph: cross-turn goals, task plans, and child routing."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paper_research_agent.agent.dynamic.models import PendingApproval
from paper_research_agent.agent.orchestrator.children import ChildGraphDispatcher
from paper_research_agent.agent.orchestrator.control import task_budget_exhausted
from paper_research_agent.agent.orchestrator.evaluator import (
    MAX_CHILD_CALLS_PER_RUN,
    MAX_REPLANS_PER_RUN,
    TaskEvaluation,
    evaluate_task,
    reduce_workspace,
)
from paper_research_agent.agent.orchestrator.hydrator import ContextHydrator
from paper_research_agent.agent.orchestrator.interpreter import TurnInterpreter
from paper_research_agent.agent.orchestrator.models import (
    AgentApprovalClaim,
    AgentContextEnvelope,
    AgentRunStart,
    AgentTask,
    Capability,
    ChildTaskRequest,
    ChildTaskResult,
    ConversationWorkspace,
    GoalDecision,
    MainAgentRequest,
    MainAgentResult,
    MainAgentResumeRequest,
    RecalledContext,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.planner import (
    GoalReconciler,
    TaskPlanner,
    build_single_local_rag_decisions,
)
from paper_research_agent.agent.orchestrator.planning_route import (
    classify_planning_route as classify_planning_route_pure,
)
from paper_research_agent.agent.orchestrator.router import CAPABILITIES
from paper_research_agent.agent.orchestrator.router import route_task as route_task_pure
from paper_research_agent.agent.orchestrator.router import (
    select_next_task as select_next_task_pure,
)
from paper_research_agent.agent.orchestrator.state import MainAgentGraphState
from paper_research_agent.agent.orchestrator.synthesizer import AnswerSynthesizer
from paper_research_agent.conversation.models import ConversationResolution, ConversationStatus
from paper_research_agent.conversation.store import ConversationStore
from paper_research_agent.web.events import (
    AgentStreamEventDraft,
    AgentStreamEventType,
    RunNodeStatus,
    SafeRunEventDetail,
)


def build_main_agent_graph(
    *,
    repository: ConversationStore,
    hydrator: ContextHydrator,
    interpreter: TurnInterpreter,
    goal_reconciler: GoalReconciler,
    task_planner: TaskPlanner,
    dispatcher: ChildGraphDispatcher,
    synthesizer: AnswerSynthesizer | None = None,
    max_child_calls: int = MAX_CHILD_CALLS_PER_RUN,
    max_replans: int = MAX_REPLANS_PER_RUN,
    checkpointer: Any | None = None,
    run_event_publisher: Any | None = None,
    fast_path_enabled: bool = False,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the graph; only commit_turn and abort_turn write storage."""
    if max_child_calls <= 0 or max_child_calls > 12:
        raise ValueError("max_child_calls must be between 1 and 12")
    if max_replans <= 0 or max_replans > 3:
        raise ValueError("max_replans must be between 1 and 3")
    answer_synthesizer = synthesizer or AnswerSynthesizer()

    async def publish_product_event(
        state: MainAgentGraphState,
        event_type: AgentStreamEventType,
        *,
        node_id: str,
        status: RunNodeStatus | None = None,
        title: str | None = None,
        summary: str | None = None,
        detail: SafeRunEventDetail | None = None,
        task_id: str | None = None,
        idempotency_key: str,
    ) -> None:
        if run_event_publisher is None:
            return
        request = MainAgentRequest.model_validate(state["request"])
        run_id = str(state.get("run_id", ""))
        turn_id = str(state.get("turn_id", ""))
        if not run_id or not turn_id:
            return
        await run_event_publisher.publish(
            AgentStreamEventDraft(
                type=event_type,
                occurred_at=datetime.now(UTC),
                request_id=request.request_id,
                run_id=run_id,
                turn_id=turn_id,
                node_id=node_id,
                parent_node_id=f"task:{task_id}" if task_id else None,
                task_id=task_id,
                status=status,
                title=title,
                summary=summary,
                detail=detail or SafeRunEventDetail(delivery_mode="event_only"),
            ),
            idempotency_key=idempotency_key,
        )

    async def initialize_turn(state: MainAgentGraphState) -> MainAgentGraphState:
        request = MainAgentRequest.model_validate(state["request"])
        raw_start = state.get("run_start")
        start = (
            AgentRunStart.model_validate(raw_start)
            if raw_start is not None
            else await asyncio.to_thread(
                repository.begin_agent_run,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                user_question=request.message,
            )
        )
        if start.outcome == "completed_cached":
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "final_answer": start.result.answer if start.result is not None else "",
                "termination_reason": "cached",
            }
        if start.outcome == "running_reused":
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "termination_reason": "running_reused",
            }
        if start.outcome == "failed_cached":
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "final_answer": "该请求此前未通过提交校验。",
                "termination_reason": "failed",
            }
        if start.outcome == "cancelled_cached":
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "final_answer": start.result.answer if start.result is not None else "运行已取消。",
                "termination_reason": "cancelled_cached",
            }
        if start.outcome == "paused_cached":
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "final_answer": start.result.answer if start.result is not None else "运行已暂停。",
                "child_results": list(start.result.child_results) if start.result is not None else [],
                "termination_reason": "paused_cached",
            }
        if start.outcome == "waiting_approval_cached":
            cached = start.result
            update: MainAgentGraphState = {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "final_answer": cached.answer if cached is not None else "等待敏感工具审批。",
                "child_results": list(cached.child_results) if cached is not None else [],
                "termination_reason": "waiting_approval_cached",
            }
            if cached is not None and cached.pending_approval is not None:
                update["pending_approval"] = cached.pending_approval
            return update
        if start.outcome == "resuming":
            used_calls = sum(task.usage.call_count for task in (start.workspace.task_plan.tasks if start.workspace.task_plan is not None else ()))
            return {
                "run_id": start.run_id,
                "turn_id": start.turn_id,
                "base_workspace_version": start.workspace.version,
                "workspace_draft": start.workspace,
                "child_results": list(start.result.child_results) if start.result is not None else [],
                "remaining_child_calls": max(0, max_child_calls - used_calls),
                "remaining_replans": max_replans,
                "resuming": True,
            }
        return {
            "run_id": start.run_id,
            "turn_id": start.turn_id,
            "base_workspace_version": start.workspace.version,
            "workspace_draft": start.workspace,
            "remaining_child_calls": max_child_calls,
            "remaining_replans": max_replans,
        }

    async def hydrate_context(state: MainAgentGraphState) -> MainAgentGraphState:
        await publish_product_event(
            state,
            "reasoning_started",
            node_id="reasoning:main",
            status="running",
            title="理解当前对话",
            summary="正在准备对话上下文",
            idempotency_key="reasoning:start",
        )
        request = MainAgentRequest.model_validate(state["request"])
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        envelope = await hydrator.hydrate(
            request,
            workspace,
            turn_id=str(state["turn_id"]),
            run_id=str(state["run_id"]),
        )
        await publish_product_event(
            state,
            "reasoning_summary",
            node_id="reasoning:main",
            status="running",
            title="上下文已准备",
            summary=f"已选择 {len(envelope.recalled_context)} 条相关记忆",
            idempotency_key="reasoning:hydrate",
        )
        return {"context": envelope}

    async def classify_planning_route(
        state: MainAgentGraphState,
    ) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        decision = classify_planning_route_pure(
            envelope,
            enabled=fast_path_enabled,
        )
        return {
            "planning_route": decision.route,
            "planning_route_reason": decision.reason_code,
        }

    async def materialize_fast_path(
        state: MainAgentGraphState,
    ) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation, goal_decision, plan_decision = (
            build_single_local_rag_decisions(envelope)
        )
        workspace = reduce_workspace(
            ConversationWorkspace.model_validate(state["workspace_draft"]),
            goal_decision=goal_decision,
        )
        workspace = reduce_workspace(
            workspace,
            plan_decision=plan_decision,
        )
        await publish_product_event(
            state,
            "goal_updated",
            node_id="goal:active",
            status="completed",
            title="目标已更新",
            summary="目标动作：create",
            detail=SafeRunEventDetail(goal_action="create"),
            idempotency_key="goal:update",
        )
        await publish_product_event(
            state,
            "plan_updated",
            node_id="plan:active",
            status="completed",
            title="研究计划已更新",
            summary="计划动作：create，共 1 个任务",
            detail=SafeRunEventDetail(plan_action="create"),
            idempotency_key="plan:update:1",
        )
        return {
            "interpretation": interpretation,
            "goal_decision": goal_decision,
            "plan_decision": plan_decision,
            "workspace_draft": workspace,
        }

    async def interpret_turn(state: MainAgentGraphState) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation = await interpreter.interpret(envelope)
        await publish_product_event(
            state,
            "reasoning_summary",
            node_id="reasoning:main",
            status="running",
            title="已理解请求",
            summary=(
                f"请求类型：{interpretation.relation}；需要补充信息"
                if interpretation.needs_clarification
                else f"请求类型：{interpretation.relation}"
            ),
            detail=SafeRunEventDetail(route=interpretation.relation),
            idempotency_key="reasoning:interpret",
        )
        return {"interpretation": interpretation}

    async def reconcile_goal(state: MainAgentGraphState) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation = TurnInterpretationV2.model_validate(state["interpretation"])
        decision = await goal_reconciler.reconcile(envelope, interpretation)
        workspace = reduce_workspace(
            ConversationWorkspace.model_validate(state["workspace_draft"]),
            goal_decision=decision,
        )
        await publish_product_event(
            state,
            "goal_updated",
            node_id="goal:active",
            status="completed",
            title="目标已更新",
            summary=f"目标动作：{decision.action}",
            detail=SafeRunEventDetail(goal_action=decision.action),
            idempotency_key="goal:update",
        )
        return {"workspace_draft": workspace, "goal_decision": decision}

    async def plan_tasks(state: MainAgentGraphState) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation = TurnInterpretationV2.model_validate(state["interpretation"])
        goal_decision = GoalDecision.model_validate(state["goal_decision"])
        decision = await task_planner.plan(envelope, interpretation, goal_decision)
        workspace = reduce_workspace(
            ConversationWorkspace.model_validate(state["workspace_draft"]),
            plan_decision=decision,
        )
        task_count = len(decision.plan.tasks) if decision.plan is not None else 0
        await publish_product_event(
            state,
            "plan_updated",
            node_id="plan:active",
            status="completed",
            title="研究计划已更新",
            summary=f"计划动作：{decision.action}，共 {task_count} 个任务",
            detail=SafeRunEventDetail(plan_action=decision.action),
            idempotency_key=f"plan:update:{decision.plan.revision if decision.plan else 0}",
        )
        return {"workspace_draft": workspace, "plan_decision": decision}

    async def select_next_task(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        control = await asyncio.to_thread(
            repository.load_agent_control, run_id=str(state.get("run_id", ""))
        )
        if control is not None and control.status == "pause_requested":
            return {
                "workspace_draft": workspace,
                "final_answer": "运行已暂停；已完成步骤及其结果均已保留。",
                "termination_reason": "paused",
                "next_action": "control_stop",
            }
        if control is not None and control.status == "cancel_requested":
            return {
                "workspace_draft": _cancel_open_tasks(workspace),
                "final_answer": "运行已取消；已完成步骤及其结果均已保留。",
                "termination_reason": "cancelled",
                "next_action": "control_stop",
            }
        selection = select_next_task_pure(workspace)
        if selection.outcome == "finalize":
            return {"next_action": "synthesize"}
        if selection.outcome in {"clarify", "blocked"}:
            return {"next_action": "clarify"}
        remaining = int(state.get("remaining_child_calls", max_child_calls))
        if remaining <= 0:
            task_id = str(selection.task_id)
            selected_task = _task_by_id(workspace, task_id)
            evaluation = TaskEvaluation(
                task_id=task_id,
                outcome="fail",
                reason="整轮子图调用预算耗尽",
            )
            workspace = reduce_workspace(
                workspace, task_id=task_id, evaluation=evaluation
            )
            child_results = list(state.get("child_results", []))
            child_results.append(
                _budget_failure_result(
                    selected_task,
                    reason="run_call_budget_exhausted",
                    summary="整轮子任务调用预算已耗尽，当前任务未执行。",
                )
            )
            return {
                "workspace_draft": workspace,
                "child_results": child_results,
                "next_action": "synthesize",
            }
        if selection.task_id is None:
            raise ValueError("execute selection requires a task id")
        selected_task = _task_by_id(workspace, selection.task_id)
        budget_reason = task_budget_exhausted(selected_task)
        if budget_reason is not None:
            child_results = list(state.get("child_results", []))
            child_results.append(
                _budget_failure_result(
                    selected_task,
                    reason=budget_reason,
                    summary=f"任务因预算限制未执行：{budget_reason}。",
                )
            )
            return {
                "workspace_draft": _fail_task_for_budget(
                    workspace, selected_task.task_id, budget_reason
                ),
                "child_results": child_results,
                "next_action": "select_next_task",
            }
        await publish_product_event(
            state,
            "task_started",
            node_id=f"task:{selected_task.task_id}",
            status="running",
            title=selected_task.title,
            summary=selected_task.execution_reason,
            detail=SafeRunEventDetail(capability=selected_task.capability),
            task_id=selected_task.task_id,
            idempotency_key=(
                f"task:{selected_task.task_id}:attempt:{selected_task.attempt_count}:started"
            ),
        )
        return {"active_task_id": selection.task_id, "next_action": "route"}

    async def route_task(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        task = _task_by_id(workspace, str(state["active_task_id"]))
        envelope = AgentContextEnvelope.model_validate(state["context"])
        decision = route_task_pure(task, envelope)
        return {"route": decision.capability}

    async def dispatch_child(state: MainAgentGraphState) -> MainAgentGraphState:
        child_request = _child_request(state)
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        task = _task_by_id(workspace, child_request.task_id)
        remaining_seconds = (
            task.budget.max_seconds - task.usage.elapsed_seconds
            if task.budget.max_seconds is not None
            else None
        )
        started = time.perf_counter()
        try:
            if remaining_seconds is not None:
                async with asyncio.timeout(max(0.001, remaining_seconds)):
                    result = await dispatcher.dispatch(child_request)
            else:
                result = await dispatcher.dispatch(child_request)
        except TimeoutError:
            result = ChildTaskResult(
                child_run_id=f"budget-{task.task_id}",
                task_id=task.task_id,
                capability=task.capability,
                status="failed",
                summary="任务超过单步时间预算。",
                error_code="task_time_budget_exhausted",
            )
        elapsed = max(0.0, time.perf_counter() - started)
        workspace = _record_task_usage(
            workspace,
            task.task_id,
            elapsed_seconds=elapsed,
            cost_usd=result.estimated_cost_usd,
        )
        child_results = list(state.get("child_results", []))
        child_results.append(result)
        return {
            "workspace_draft": workspace,
            "child_results": child_results,
            "child_result": result,
            "remaining_child_calls": int(
                state.get("remaining_child_calls", max_child_calls)
            )
            - 1,
        }

    async def evaluate_result(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        result = ChildTaskResult.model_validate(state["child_result"])
        task = _task_by_id(workspace, str(state["active_task_id"]))
        used_calls = max_child_calls - int(
            state.get("remaining_child_calls", max_child_calls)
        )
        used_replans = max_replans - int(
            state.get("remaining_replans", max_replans)
        )
        evaluation = evaluate_task(
            task,
            result,
            child_calls_used=used_calls,
            replans_used=used_replans,
        )
        return {"evaluation": evaluation}

    async def update_task_state(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        evaluation = TaskEvaluation.model_validate(state["evaluation"])
        raw_result = state.get("child_result")
        result = (
            ChildTaskResult.model_validate(raw_result)
            if raw_result is not None
            else None
        )
        workspace = reduce_workspace(
            workspace,
            task_id=str(state["active_task_id"]),
            evaluation=evaluation,
            result=result,
        )
        active_task_id = str(state["active_task_id"])
        completed = evaluation.outcome == "complete"
        await publish_product_event(
            state,
            "task_completed" if completed else "task_failed",
            node_id=f"task:{active_task_id}",
            status="completed" if completed else "failed",
            title="任务完成" if completed else "任务未完成",
            summary=evaluation.reason,
            task_id=active_task_id,
            idempotency_key=(
                f"task:{active_task_id}:evaluation:{evaluation.outcome}:"
                f"{int(state.get('remaining_replans', max_replans))}"
            ),
        )
        update: MainAgentGraphState = {"workspace_draft": workspace}
        if evaluation.outcome == "replan":
            update["remaining_replans"] = int(
                state.get("remaining_replans", max_replans)
            ) - 1
            update["next_action"] = "plan_tasks"
        elif evaluation.outcome == "wait_user":
            if (
                result is not None
                and result.status == "waiting_approval"
                and result.pending_approval is not None
            ):
                update["pending_approval"] = result.pending_approval
                update["next_action"] = "pause_approval"
            else:
                update["next_action"] = "synthesize"
        else:
            update["next_action"] = "select_next_task"
        return update

    async def clarify_response(state: MainAgentGraphState) -> MainAgentGraphState:
        interpretation = state.get("interpretation")
        question = (
            TurnInterpretationV2.model_validate(interpretation).clarification_question
            if interpretation is not None
            else None
        )
        answer = question or "请说明你希望我做什么。"
        return {"direct_answer": answer, "final_answer": answer}

    async def synthesize_response(state: MainAgentGraphState) -> MainAgentGraphState:
        direct = state.get("direct_answer")
        if isinstance(direct, str) and direct.strip():
            return {"final_answer": direct[:20_000]}
        context = AgentContextEnvelope.model_validate(state["context"])
        child_results = tuple(
            ChildTaskResult.model_validate(raw)
            for raw in state.get("child_results", [])
        )
        answer = await answer_synthesizer.synthesize(context, child_results)
        return {"final_answer": answer.text[:20_000]}

    async def update_workspace_summary(
        state: MainAgentGraphState,
    ) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        summary = _build_summary(workspace)
        updated = workspace.model_copy(update={"summary": summary[:3_000]})
        return {"workspace_draft": updated}

    async def validate_commit(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        errors = _validate_commit_state(workspace, state)
        return {"validation_errors": errors}

    async def abort_turn(state: MainAgentGraphState) -> MainAgentGraphState:
        outcome = await asyncio.to_thread(
            repository.fail_agent_run,
            run_id=str(state.get("run_id", "")),
            turn_id=str(state.get("turn_id", "")),
            reason_code="commit_validation_failed",
        )
        return {
            "commit_outcome": outcome,
            "final_answer": "主 Agent 状态未通过提交校验，本次运行已安全终止。",
            "termination_reason": "failed",
        }

    async def commit_turn(state: MainAgentGraphState) -> MainAgentGraphState:
        request = MainAgentRequest.model_validate(state["request"])
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        pending = state.get("pending_approval")
        termination = str(state.get("termination_reason", ""))
        status: Literal["paused", "cancelled", "waiting_approval", "completed"]
        if termination == "paused":
            status = "paused"
        elif termination == "cancelled":
            status = "cancelled"
        elif pending is not None:
            status = "waiting_approval"
        else:
            status = "completed"
        turn_status: ConversationStatus
        if status in {"paused", "waiting_approval"}:
            turn_status = "pending"
        elif status == "cancelled":
            turn_status = "cancelled"
        else:
            turn_status = "completed"
        answer = str(state.get("final_answer", ""))
        if not answer and status == "waiting_approval":
            answer = "等待敏感工具审批。"
        child_results = tuple(
            ChildTaskResult.model_validate(item)
            for item in state.get("child_results", [])
        )
        source_ids = _source_ids(child_results)
        result = MainAgentResult(
            run_id=str(state.get("run_id", "")),
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            status=status,
            answer=answer,
            route_trace=tuple(_route_trace(state)),
            child_results=child_results,
            pending_approval=pending,
            workspace_version=workspace.version + 1,
        )
        resolution = ConversationResolution(
            original_question=request.message,
            standalone_question=request.message,
            chinese_query=request.message,
            confidence=0.9,
        )
        outcome = await asyncio.to_thread(
            repository.commit_agent_run,
            run_id=str(state.get("run_id", "")),
            turn_id=str(state.get("turn_id", "")),
            expected_workspace_version=workspace.version,
            workspace=workspace,
            route="main_agent",
            status=turn_status,
            resolution=resolution,
            assistant_summary=workspace.summary,
            source_ids=source_ids,
            result=result,
        )
        return {
            "commit_outcome": outcome,
            "final_answer": answer,
            "termination_reason": status,
        }

    def after_initialize(state: MainAgentGraphState) -> str:
        if state.get("termination_reason") in {
            "cached",
            "running_reused",
            "waiting_approval",
            "waiting_approval_cached",
            "failed",
            "paused_cached",
            "cancelled_cached",
        }:
            return END
        return "hydrate_context"

    def after_hydrate(state: MainAgentGraphState) -> str:
        return (
            "select_next_task"
            if state.get("resuming")
            else "classify_planning_route"
        )

    def after_planning_route(state: MainAgentGraphState) -> str:
        return (
            "materialize_fast_path"
            if state.get("planning_route") == "fast_path"
            else "interpret_turn"
        )

    def after_interpret(state: MainAgentGraphState) -> str:
        raw = state.get("interpretation")
        if raw is not None and TurnInterpretationV2.model_validate(raw).needs_clarification:
            return "clarify_response"
        return "reconcile_goal"

    def after_select(state: MainAgentGraphState) -> str:
        action = state.get("next_action")
        if action == "route":
            return "route_task"
        if action == "clarify":
            return "clarify_response"
        if action == "control_stop":
            return "update_workspace_summary"
        if action == "select_next_task":
            return "select_next_task"
        return "synthesize_response"

    def after_route(state: MainAgentGraphState) -> str:
        route = str(state.get("route", ""))
        if route in CAPABILITIES:
            return "dispatch_child"
        raise ValueError(f"unsupported main-agent route: {route}")

    def after_update(state: MainAgentGraphState) -> str:
        action = state.get("next_action")
        if action == "plan_tasks":
            return "plan_tasks"
        if action == "pause_approval":
            return "pause_approval"
        if action == "synthesize":
            return "synthesize_response"
        return "select_next_task"

    def after_validate(state: MainAgentGraphState) -> str:
        return "abort_turn" if state.get("validation_errors") else "commit_turn"

    builder = StateGraph(MainAgentGraphState)
    builder.add_node("initialize_turn", initialize_turn)
    builder.add_node("hydrate_context", hydrate_context)
    builder.add_node("classify_planning_route", classify_planning_route)
    builder.add_node("materialize_fast_path", materialize_fast_path)
    builder.add_node("interpret_turn", interpret_turn)
    builder.add_node("reconcile_goal", reconcile_goal)
    builder.add_node("plan_tasks", plan_tasks)
    builder.add_node("select_next_task", select_next_task)
    builder.add_node("route_task", route_task)
    builder.add_node("dispatch_child", dispatch_child)
    builder.add_node("evaluate_result", evaluate_result)
    builder.add_node("update_task_state", update_task_state)
    builder.add_node("clarify_response", clarify_response)
    builder.add_node("synthesize_response", synthesize_response)
    builder.add_node("update_workspace_summary", update_workspace_summary)
    builder.add_node("validate_commit", validate_commit)
    builder.add_node("abort_turn", abort_turn)
    builder.add_node("commit_turn", commit_turn)
    builder.add_edge(START, "initialize_turn")
    builder.add_conditional_edges("initialize_turn", after_initialize, {END: END, "hydrate_context": "hydrate_context"})
    builder.add_conditional_edges(
        "hydrate_context",
        after_hydrate,
        {
            "classify_planning_route": "classify_planning_route",
            "select_next_task": "select_next_task",
        },
    )
    builder.add_conditional_edges(
        "classify_planning_route",
        after_planning_route,
        {
            "materialize_fast_path": "materialize_fast_path",
            "interpret_turn": "interpret_turn",
        },
    )
    builder.add_edge("materialize_fast_path", "select_next_task")
    builder.add_conditional_edges(
        "interpret_turn",
        after_interpret,
        {
            "reconcile_goal": "reconcile_goal",
            "clarify_response": "clarify_response",
        },
    )
    builder.add_edge("reconcile_goal", "plan_tasks")
    builder.add_edge("plan_tasks", "select_next_task")
    builder.add_conditional_edges(
        "select_next_task",
        after_select,
        {
            "route_task": "route_task",
            "clarify_response": "clarify_response",
            "synthesize_response": "synthesize_response",
            "update_workspace_summary": "update_workspace_summary",
            "select_next_task": "select_next_task",
        },
    )
    builder.add_conditional_edges(
        "route_task",
        after_route,
        {"dispatch_child": "dispatch_child"},
    )
    builder.add_edge("dispatch_child", "evaluate_result")
    builder.add_edge("evaluate_result", "update_task_state")
    builder.add_conditional_edges(
        "update_task_state",
        after_update,
        {
            "select_next_task": "select_next_task",
            "plan_tasks": "plan_tasks",
            "synthesize_response": "synthesize_response",
            "pause_approval": "update_workspace_summary",
        },
    )
    builder.add_edge("clarify_response", "update_workspace_summary")
    builder.add_edge("synthesize_response", "update_workspace_summary")
    builder.add_edge("update_workspace_summary", "validate_commit")
    builder.add_conditional_edges(
        "validate_commit",
        after_validate,
        {"abort_turn": "abort_turn", "commit_turn": "commit_turn"},
    )
    builder.add_edge("abort_turn", END)
    builder.add_edge("commit_turn", END)
    return builder.compile(checkpointer=checkpointer, name="paper_research_main_v1")


class MainAgentApprovalResumer:
    """Atomically claim and resume exactly one waiting dynamic child task."""

    def __init__(
        self,
        *,
        repository: ConversationStore,
        dispatcher: ChildGraphDispatcher,
        synthesizer: AnswerSynthesizer | None = None,
    ) -> None:
        self._repository = repository
        self._dispatcher = dispatcher
        self._synthesizer = synthesizer or AnswerSynthesizer()

    async def resume(self, request_id: str, approved: bool) -> MainAgentResult:
        request = MainAgentResumeRequest(request_id=request_id, approved=approved)
        stored = await asyncio.to_thread(
            self._repository.load_agent_run, request.request_id
        )
        if stored is None or stored.status != "waiting_approval":
            raise RuntimeError("approval request is not waiting")
        pending_payload = stored.pending_approval
        if pending_payload is None:
            raise RuntimeError("waiting run has no pending approval")
        approval_id = str(pending_payload.get("approval_request_id", ""))
        claim = await asyncio.to_thread(
            self._repository.claim_agent_approval,
            request_id=request.request_id,
            approval_request_id=approval_id,
        )
        if claim is None:
            raise RuntimeError("approval request was already consumed or does not match")
        try:
            task, _pending = _validate_approval_claim(claim.workspace, claim.result)
            child_request = _resume_child_request(
                claim.result, claim.workspace, task, claim.turn_id
            )
            resumed = await self._dispatcher.resume_dynamic_tools(
                child_request,
                approved=request.approved,
            )
            return await self._finish_resume(claim, task, resumed)
        except Exception:
            await asyncio.to_thread(
                self._repository.fail_agent_run,
                run_id=claim.result.run_id,
                turn_id=claim.turn_id,
                reason_code="approval_resume_failed",
            )
            raise

    async def _finish_resume(
        self,
        claim: AgentApprovalClaim,
        task: AgentTask,
        resumed: ChildTaskResult,
    ) -> MainAgentResult:
        if resumed.status == "failed":
            evaluation = TaskEvaluation(
                task_id=task.task_id,
                outcome="fail",
                missing_criteria=task.success_criteria,
                summary=resumed.summary,
                reason="审批未执行或已失效",
            )
        else:
            evaluation = evaluate_task(
                task,
                resumed,
                child_calls_used=MAX_CHILD_CALLS_PER_RUN,
                replans_used=MAX_REPLANS_PER_RUN,
            )
        workspace = reduce_workspace(
            claim.workspace,
            task_id=task.task_id,
            evaluation=evaluation,
            result=resumed,
        )
        child_results = tuple(
            resumed if item.task_id == task.task_id else item
            for item in claim.result.child_results
        )
        pending = resumed.pending_approval if resumed.status == "waiting_approval" else None
        if pending is None:
            context = AgentContextEnvelope(
                conversation_id=claim.result.conversation_id,
                request_id=claim.result.request_id,
                turn_id=claim.turn_id,
                current_message=task.objective,
                rag_mode="preferred",
                workspace=workspace,
                prepared_at=datetime.now(UTC),
            )
            synthesized = await self._synthesizer.synthesize(context, child_results)
            answer = synthesized.text[:20_000]
            status: Literal["completed", "waiting_approval"] = "completed"
        else:
            answer = "等待敏感工具审批。"
            status = "waiting_approval"
        workspace = workspace.model_copy(update={"summary": _build_summary(workspace)[:3_000]})
        validation_state: MainAgentGraphState = {
            "child_results": list(child_results),
        }
        if pending is not None:
            validation_state["pending_approval"] = pending
        errors = _validate_commit_state(workspace, validation_state)
        if errors:
            raise ValueError(f"resumed state failed commit validation: {errors}")
        result = MainAgentResult(
            run_id=claim.result.run_id,
            request_id=claim.result.request_id,
            conversation_id=claim.result.conversation_id,
            status=status,
            answer=answer,
            route_trace=claim.result.route_trace,
            child_results=child_results,
            pending_approval=pending,
            workspace_version=workspace.version + 1,
        )
        resolution = ConversationResolution(
            original_question=task.objective,
            standalone_question=task.objective,
            chinese_query=task.objective,
            confidence=1,
        )
        outcome = await asyncio.to_thread(
            self._repository.commit_agent_run,
            run_id=result.run_id,
            turn_id=claim.turn_id,
            expected_workspace_version=workspace.version,
            workspace=workspace,
            route="main_agent",
            status="pending" if status == "waiting_approval" else "completed",
            resolution=resolution,
            assistant_summary=workspace.summary,
            source_ids=_source_ids(child_results),
            result=result,
        )
        if not outcome.committed:
            raise RuntimeError(f"approval resume commit failed: {outcome.reason}")
        return result


def _validate_approval_claim(
    workspace: ConversationWorkspace,
    result: MainAgentResult,
) -> tuple[AgentTask, PendingApproval]:
    plan = workspace.task_plan
    if plan is None:
        raise ValueError("waiting approval has no task plan")
    waiting = tuple(task for task in plan.tasks if task.status == "waiting_approval")
    if len(waiting) != 1:
        raise ValueError("waiting approval requires exactly one waiting task")
    task = waiting[0]
    pending_payload = result.pending_approval
    if pending_payload is None:
        raise ValueError("waiting approval result has no pending payload")
    task_id = str(pending_payload.get("task_id", ""))
    if task_id != task.task_id:
        raise ValueError("pending approval task ID does not match waiting task")
    pending_values = dict(pending_payload)
    pending_values.pop("task_id", None)
    pending = PendingApproval.model_validate(pending_values)
    waiting_children = tuple(
        child
        for child in result.child_results
        if child.status == "waiting_approval" and child.task_id == task.task_id
    )
    if len(waiting_children) != 1 or waiting_children[0].pending_approval is None:
        raise ValueError("waiting approval child result is missing or ambiguous")
    child_payload = waiting_children[0].pending_approval
    for key in (
        "task_id",
        "tool_name",
        "approval_request_id",
        "arguments_sha256",
        "expires_at_epoch",
    ):
        if child_payload.get(key) != pending_payload.get(key):
            raise ValueError(f"pending approval {key} does not match child result")
    return task, pending


def _resume_child_request(
    result: MainAgentResult,
    workspace: ConversationWorkspace,
    task: AgentTask,
    turn_id: str,
) -> ChildTaskRequest:
    goal = workspace.active_goal
    if goal is None or goal.goal_id != task.goal_id:
        raise ValueError("waiting approval task has no matching active goal")
    return ChildTaskRequest(
        run_id=result.run_id,
        request_id=result.request_id,
        conversation_id=result.conversation_id,
        turn_id=turn_id,
        goal_id=task.goal_id,
        goal_objective=goal.objective,
        task_id=task.task_id,
        attempt_count=task.attempt_count,
        objective=task.objective,
        success_criteria=task.success_criteria,
        capability="dynamic_tools",
        current_message=task.objective,
        conversation_summary=workspace.summary,
        constraints=goal.constraints,
        rag_mode="preferred",
    )


def _task_by_id(workspace: ConversationWorkspace, task_id: str) -> AgentTask:
    plan = workspace.task_plan
    if plan is None:
        raise ValueError("no task plan in workspace")
    for task in plan.tasks:
        if task.task_id == task_id:
            return task
    raise ValueError(f"unknown task id: {task_id}")


def _replace_task(
    workspace: ConversationWorkspace, task_id: str, updated_task: AgentTask
) -> ConversationWorkspace:
    plan = workspace.task_plan
    if plan is None:
        raise ValueError("no task plan in workspace")
    tasks = tuple(
        updated_task if task.task_id == task_id else task for task in plan.tasks
    )
    now = datetime.now(UTC)
    return workspace.model_copy(
        update={
            "task_plan": plan.model_copy(update={"tasks": tasks, "updated_at": now}),
            "updated_at": now,
        }
    )


def _record_task_usage(
    workspace: ConversationWorkspace,
    task_id: str,
    *,
    elapsed_seconds: float,
    cost_usd: float,
) -> ConversationWorkspace:
    task = _task_by_id(workspace, task_id)
    usage = task.usage.model_copy(
        update={
            "elapsed_seconds": task.usage.elapsed_seconds + elapsed_seconds,
            "call_count": task.usage.call_count + 1,
            "cost_usd": task.usage.cost_usd + cost_usd,
        }
    )
    return _replace_task(workspace, task_id, task.model_copy(update={"usage": usage}))


def _fail_task_for_budget(
    workspace: ConversationWorkspace, task_id: str, reason: str
) -> ConversationWorkspace:
    task = _task_by_id(workspace, task_id)
    failed = task.model_copy(update={"status": "failed", "blocked_reason": reason})
    return _replace_task(workspace, task_id, failed)


def _budget_failure_result(
    task: AgentTask, *, reason: str, summary: str
) -> ChildTaskResult:
    return ChildTaskResult(
        child_run_id=f"budget-{task.task_id}",
        task_id=task.task_id,
        capability=task.capability,
        status="failed",
        summary=summary,
        error_code=reason,
    )


def _cancel_open_tasks(workspace: ConversationWorkspace) -> ConversationWorkspace:
    plan = workspace.task_plan
    if plan is None:
        return workspace
    tasks = tuple(
        task
        if task.status in {"completed", "skipped", "cancelled"}
        else task.model_copy(
            update={"status": "cancelled", "blocked_reason": "用户取消运行"}
        )
        for task in plan.tasks
    )
    now = datetime.now(UTC)
    return workspace.model_copy(
        update={
            "task_plan": plan.model_copy(update={"tasks": tasks, "updated_at": now}),
            "updated_at": now,
        }
    )


def _child_request(state: MainAgentGraphState) -> ChildTaskRequest:
    request = MainAgentRequest.model_validate(state["request"])
    workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
    envelope = AgentContextEnvelope.model_validate(state["context"])
    raw_interpretation = state.get("interpretation")
    interpretation = (
        TurnInterpretationV2.model_validate(raw_interpretation)
        if raw_interpretation is not None
        else None
    )
    task = _task_by_id(workspace, str(state["active_task_id"]))
    goal_constraints = (
        workspace.active_goal.constraints if workspace.active_goal is not None else ()
    )
    return ChildTaskRequest(
        run_id=str(state.get("run_id", "")),
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        turn_id=str(state.get("turn_id", "")),
        goal_id=task.goal_id,
        goal_objective=(
            workspace.active_goal.objective
            if workspace.active_goal is not None
            else ""
        ),
        task_id=task.task_id,
        attempt_count=task.attempt_count,
        objective=task.objective,
        success_criteria=task.success_criteria,
        capability=_routed_capability(state),
        current_message=request.message,
        conversation_summary=workspace.summary,
        constraints=task.constraints
        if hasattr(task, "constraints")
        else goal_constraints,
        recent_messages=envelope.recent_messages,
        selected_context=_selected_recalled_context(envelope, interpretation),
        rag_mode=request.rag_mode,
        attachment_ids=request.attachment_ids,
    )


def _selected_recalled_context(
    envelope: AgentContextEnvelope,
    interpretation: TurnInterpretationV2 | None,
) -> tuple[RecalledContext, ...]:
    if interpretation is None:
        return ()
    selected = set(interpretation.selected_context_ids)
    return tuple(
        item for item in envelope.recalled_context if item.source_id in selected
    )


def _routed_capability(state: MainAgentGraphState) -> Capability:
    route = str(state.get("route", ""))
    if route not in CAPABILITIES:
        raise ValueError(f"unsupported main-agent route: {route}")
    return cast(Capability, route)


def _build_summary(workspace: ConversationWorkspace) -> str:
    parts: list[str] = []
    if workspace.active_goal is not None:
        parts.append(f"目标：{workspace.active_goal.objective}")
    if workspace.stable_constraints:
        parts.append(f"约束：{'；'.join(workspace.stable_constraints)}")
    if workspace.task_plan is not None:
        completed = [
            task.title for task in workspace.task_plan.tasks if task.status == "completed"
        ]
        pending = [
            task.title
            for task in workspace.task_plan.tasks
            if task.status not in {"completed", "cancelled", "skipped", "failed"}
        ]
        if completed:
            parts.append(f"已完成：{'；'.join(completed)}")
        if pending:
            parts.append(f"未完成：{'；'.join(pending)}")
    return "。".join(parts)


def _validate_commit_state(
    workspace: ConversationWorkspace, state: MainAgentGraphState
) -> tuple[str, ...]:
    errors: list[str] = []
    plan = workspace.task_plan
    if (
        plan is not None
        and workspace.active_goal is not None
        and plan.goal_id != workspace.active_goal.goal_id
    ):
        errors.append("plan goal ID does not match the active goal")
    if plan is not None:
        running = [task for task in plan.tasks if task.status == "running"]
        if len(running) > 1:
            errors.append("more than one running task")
        pending_approval = state.get("pending_approval")
        approval_task = [
            task
            for task in plan.tasks
            if task.status == "waiting_approval" or task.status == "waiting_user"
        ]
        if pending_approval is not None and not approval_task:
            errors.append("pending approval without a waiting task")
    source_ids = tuple(
        source_id
        for item in state.get("child_results", [])
        for source_id in ChildTaskResult.model_validate(item).source_ids
    )
    if len(source_ids) != len(set(source_ids)):
        errors.append("child source IDs must be unique")
    return tuple(errors)


def _source_ids(child_results: tuple[ChildTaskResult, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for result in child_results:
        for source_id in result.source_ids:
            if source_id not in seen:
                seen.append(source_id)
    return tuple(seen)


def _route_trace(state: MainAgentGraphState) -> list[str]:
    raw = state.get("route_trace", [])
    trace = [str(item) for item in raw] if isinstance(raw, list) else []
    route = state.get("route")
    if isinstance(route, str) and route not in trace:
        trace.append(route)
    return trace
