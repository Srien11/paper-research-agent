"""Main Agent LangGraph: cross-turn goals, task plans, and child routing."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from paper_research_agent.agent.dynamic.models import PendingApproval
from paper_research_agent.agent.orchestrator.children import ChildGraphDispatcher
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
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.planner import GoalReconciler, TaskPlanner
from paper_research_agent.agent.orchestrator.router import CAPABILITIES
from paper_research_agent.agent.orchestrator.router import route_task as route_task_pure
from paper_research_agent.agent.orchestrator.router import (
    select_next_task as select_next_task_pure,
)
from paper_research_agent.agent.orchestrator.state import MainAgentGraphState
from paper_research_agent.agent.orchestrator.synthesizer import AnswerSynthesizer
from paper_research_agent.conversation.models import ConversationResolution, ConversationStatus
from paper_research_agent.conversation.store import ConversationStore


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
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the graph; only commit_turn and abort_turn write storage."""
    if max_child_calls <= 0 or max_child_calls > 12:
        raise ValueError("max_child_calls must be between 1 and 12")
    if max_replans <= 0 or max_replans > 3:
        raise ValueError("max_replans must be between 1 and 3")
    answer_synthesizer = synthesizer or AnswerSynthesizer()

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
        return {
            "run_id": start.run_id,
            "turn_id": start.turn_id,
            "base_workspace_version": start.workspace.version,
            "workspace_draft": start.workspace,
            "remaining_child_calls": max_child_calls,
            "remaining_replans": max_replans,
        }

    async def hydrate_context(state: MainAgentGraphState) -> MainAgentGraphState:
        request = MainAgentRequest.model_validate(state["request"])
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        envelope = await hydrator.hydrate(
            request, workspace, turn_id=str(state["turn_id"])
        )
        return {"context": envelope}

    async def interpret_turn(state: MainAgentGraphState) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation = await interpreter.interpret(envelope)
        return {"interpretation": interpretation}

    async def reconcile_goal(state: MainAgentGraphState) -> MainAgentGraphState:
        envelope = AgentContextEnvelope.model_validate(state["context"])
        interpretation = TurnInterpretationV2.model_validate(state["interpretation"])
        decision = await goal_reconciler.reconcile(envelope, interpretation)
        workspace = reduce_workspace(
            ConversationWorkspace.model_validate(state["workspace_draft"]),
            goal_decision=decision,
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
        return {"workspace_draft": workspace, "plan_decision": decision}

    async def select_next_task(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        selection = select_next_task_pure(workspace)
        if selection.outcome == "finalize":
            return {"next_action": "synthesize"}
        if selection.outcome in {"clarify", "blocked"}:
            return {"next_action": "clarify"}
        remaining = int(state.get("remaining_child_calls", max_child_calls))
        if remaining <= 0:
            evaluation = TaskEvaluation(
                task_id=str(selection.task_id),
                outcome="fail",
                reason="整轮子图调用预算耗尽",
            )
            workspace = reduce_workspace(
                workspace, task_id=str(selection.task_id), evaluation=evaluation
            )
            return {"workspace_draft": workspace, "next_action": "synthesize"}
        if selection.task_id is None:
            raise ValueError("execute selection requires a task id")
        return {"active_task_id": selection.task_id, "next_action": "route"}

    async def route_task(state: MainAgentGraphState) -> MainAgentGraphState:
        workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
        task = _task_by_id(workspace, str(state["active_task_id"]))
        envelope = AgentContextEnvelope.model_validate(state["context"])
        decision = route_task_pure(task, envelope)
        return {"route": decision.capability}

    async def dispatch_child(state: MainAgentGraphState) -> MainAgentGraphState:
        child_request = _child_request(state)
        result = await dispatcher.dispatch(child_request)
        child_results = list(state.get("child_results", []))
        child_results.append(result)
        return {
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
        status: Literal["waiting_approval", "completed"] = (
            "waiting_approval" if pending is not None else "completed"
        )
        turn_status: ConversationStatus = (
            "pending" if pending is not None else "completed"
        )
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
        }:
            return END
        return "hydrate_context"

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
    builder.add_edge("hydrate_context", "interpret_turn")
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
            child_request = _resume_child_request(claim.result, claim.workspace, task)
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
) -> ChildTaskRequest:
    goal = workspace.active_goal
    if goal is None or goal.goal_id != task.goal_id:
        raise ValueError("waiting approval task has no matching active goal")
    return ChildTaskRequest(
        run_id=result.run_id,
        conversation_id=result.conversation_id,
        goal_id=task.goal_id,
        goal_objective=goal.objective,
        task_id=task.task_id,
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


def _child_request(state: MainAgentGraphState) -> ChildTaskRequest:
    request = MainAgentRequest.model_validate(state["request"])
    workspace = ConversationWorkspace.model_validate(state["workspace_draft"])
    envelope = AgentContextEnvelope.model_validate(state["context"])
    task = _task_by_id(workspace, str(state["active_task_id"]))
    goal_constraints = (
        workspace.active_goal.constraints if workspace.active_goal is not None else ()
    )
    return ChildTaskRequest(
        run_id=str(state.get("run_id", "")),
        conversation_id=request.conversation_id,
        goal_id=task.goal_id,
        goal_objective=(
            workspace.active_goal.objective
            if workspace.active_goal is not None
            else ""
        ),
        task_id=task.task_id,
        objective=task.objective,
        success_criteria=task.success_criteria,
        capability=_routed_capability(state),
        current_message=request.message,
        conversation_summary=workspace.summary,
        constraints=task.constraints
        if hasattr(task, "constraints")
        else goal_constraints,
        recent_messages=envelope.recent_messages,
        selected_context=envelope.recalled_context,
        rag_mode=request.rag_mode,
        attachment_ids=request.attachment_ids,
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
