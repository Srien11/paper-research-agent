from __future__ import annotations

import unittest
from collections import deque
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.graph import build_main_agent_graph
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentTask,
    ChildTaskResult,
    ConversationWorkspace,
    GoalDecision,
    GoalState,
    MainAgentRequest,
    TaskPlan,
    TaskPlanDecision,
    TurnInterpretationV2,
)
from paper_research_agent.conversation.store import InMemoryConversationStore


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal(**overrides: object) -> GoalState:
    values: dict[str, object] = {
        "goal_id": "a" * 32,
        "objective": "比较 RAG 与 GraphRAG",
        "status": "active",
        "origin_turn_id": "b" * 32,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return GoalState(**values)


def _task(**overrides: object) -> AgentTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "goal_id": "a" * 32,
        "title": "检索证据",
        "objective": "比较 RAG 与 GraphRAG",
        "success_criteria": ("找到证据",),
        "capability": "local_rag",
        "status": "pending",
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


def _plan_decision(tasks: tuple[AgentTask, ...], action: str = "create") -> TaskPlanDecision:
    return TaskPlanDecision(
        action=action,
        plan=_plan(tasks),
        rationale="测试计划",
    )


def _result(
    status: str = "completed",
    *,
    capability: str = "local_rag",
    task_id: str = "task-1",
    citation_kind: str | None = None,
    source_id: str | None = None,
) -> ChildTaskResult:
    values: dict[str, object] = {
        "child_run_id": "r" * 32,
        "task_id": task_id,
        "capability": capability,
        "status": status,
        "summary": "任务完成",
        "citation_kind": citation_kind or ("local_paper" if status == "completed" else "none"),
        "source_ids": (source_id,) if source_id else (),
    }
    if status == "waiting_approval":
        values["pending_approval"] = {"approval_request_id": "a" * 32}
    if status == "failed":
        values["error_code"] = "test_failure"
    return ChildTaskResult(**values)


def _interpretation(**overrides: object) -> TurnInterpretationV2:
    values: dict[str, object] = {
        "relation": "new_goal",
        "resolved_request": "比较 RAG 与 GraphRAG",
        "confidence": 0.9,
    }
    values.update(overrides)
    return TurnInterpretationV2(**values)


def _goal_decision(goal: GoalState | None = None) -> GoalDecision:
    return GoalDecision(
        action="create",
        goal=goal or _goal(),
        rationale="建立目标",
    )


class FakeHydrator:
    async def hydrate(
        self, request: MainAgentRequest, workspace: ConversationWorkspace, *, turn_id: str
    ) -> AgentContextEnvelope:
        return AgentContextEnvelope(
            conversation_id=request.conversation_id,
            request_id=request.request_id,
            turn_id=turn_id,
            current_message=request.message,
            rag_mode=request.rag_mode,
            attachment_ids=request.attachment_ids,
            workspace=workspace,
            recent_messages=(),
            recalled_context=(),
            prepared_at=_utc(),
        )


class FakeInterpreter:
    def __init__(self, interpretation: TurnInterpretationV2) -> None:
        self.interpretation = interpretation

    async def interpret(self, envelope: AgentContextEnvelope) -> TurnInterpretationV2:
        del envelope
        return self.interpretation


class FakeReconciler:
    def __init__(self, decision: GoalDecision) -> None:
        self.decision = decision

    async def reconcile(
        self, envelope: AgentContextEnvelope, interpretation: TurnInterpretationV2
    ) -> GoalDecision:
        del envelope, interpretation
        return self.decision


class FakePlanner:
    def __init__(self, *decisions: TaskPlanDecision) -> None:
        self.decisions = deque(decisions)
        self.calls = 0

    async def plan(
        self,
        envelope: AgentContextEnvelope,
        interpretation: TurnInterpretationV2,
        goal_decision: GoalDecision,
    ) -> TaskPlanDecision:
        del envelope, interpretation, goal_decision
        self.calls += 1
        if not self.decisions:
            raise AssertionError("planner exhausted its decisions")
        return self.decisions.popleft()


class FakeDispatcher:
    def __init__(self, *results: ChildTaskResult) -> None:
        self.results = deque(results)
        self.calls: list[object] = []

    async def dispatch(self, request: object) -> ChildTaskResult:
        self.calls.append(request)
        if not self.results:
            raise AssertionError("dispatcher exhausted its results")
        return self.results.popleft()


class MainAgentGraphTests(unittest.IsolatedAsyncioTestCase):
    def _build(
        self,
        *,
        interpretation: TurnInterpretationV2 | None = None,
        plan_decisions: tuple[TaskPlanDecision, ...] = (),
        dispatch_results: tuple[ChildTaskResult, ...] = (),
        max_child_calls: int = 3,
    ) -> tuple[object, InMemoryConversationStore, FakeDispatcher, FakePlanner]:
        store = InMemoryConversationStore()
        planner = FakePlanner(*plan_decisions)
        dispatcher = FakeDispatcher(*dispatch_results)
        graph = build_main_agent_graph(
            repository=store,
            hydrator=FakeHydrator(),
            interpreter=FakeInterpreter(
                interpretation or _interpretation()
            ),
            goal_reconciler=FakeReconciler(_goal_decision()),
            task_planner=planner,
            dispatcher=dispatcher,
            max_child_calls=max_child_calls,
        )
        return graph, store, dispatcher, planner

    async def _run(
        self, graph: object, request: MainAgentRequest
    ) -> dict[str, object]:
        return await graph.ainvoke({"request": request.model_dump(mode="json")})  # type: ignore[attr-defined]

    async def test_direct_chat_flow(self) -> None:
        plan = _plan_decision((_task(task_id="chat", capability="direct_chat"),))
        graph, store, dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="你好",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(state["final_answer"], "你好")
        self.assertEqual(dispatcher.calls, [])
        self.assertEqual(store.load_agent_run("request-1").status, "completed")

    async def test_local_rag_flow(self) -> None:
        plan = _plan_decision((_task(task_id="local", capability="local_rag"),))
        graph, store, dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(_result(task_id="local", source_id="chunk-1"),),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较 RAG 与 GraphRAG",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(len(dispatcher.calls), 1)
        self.assertIn("[local_rag]", state["final_answer"])
        task = store.load_workspace("conversation-1").task_plan.tasks[0]
        self.assertEqual(task.status, "completed")

    async def test_dynamic_tools_flow(self) -> None:
        plan = _plan_decision((_task(task_id="web", capability="dynamic_tools"),))
        graph, _store, dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(
                _result(
                    status="completed",
                    capability="dynamic_tools",
                    task_id="web",
                    citation_kind="external",
                    source_id="ext-1",
                ),
            ),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="核验最新维护状态",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(len(dispatcher.calls), 1)
        self.assertIn("[dynamic_tools]", state["final_answer"])

    async def test_hybrid_sequence_runs_local_then_dynamic(self) -> None:
        plan = _plan_decision(
            (
                _task(task_id="local", capability="local_rag"),
                _task(
                    task_id="web",
                    capability="dynamic_tools",
                    depends_on=("local",),
                ),
            )
        )
        graph, _store, dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(
                _result(task_id="local", source_id="chunk-1"),
                _result(
                    status="completed",
                    capability="dynamic_tools",
                    task_id="web",
                    citation_kind="external",
                    source_id="ext-1",
                ),
            ),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较论文方法并核验最新状态",
            rag_mode="preferred",
        )
        await self._run(graph, request)
        self.assertEqual(len(dispatcher.calls), 2)
        self.assertEqual(dispatcher.calls[0].task_id, "local")
        self.assertEqual(dispatcher.calls[1].task_id, "web")

    async def test_clarification_stops_execution(self) -> None:
        graph, _store, dispatcher, _planner = self._build(
            interpretation=_interpretation(
                needs_clarification=True,
                clarification_question="你想比较哪两个方案？",
                confidence=0.4,
            ),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较它们",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(dispatcher.calls, [])
        self.assertEqual(state["final_answer"], "你想比较哪两个方案？")

    async def test_replan_path_replans_then_succeeds(self) -> None:
        first_plan = _plan_decision((_task(task_id="local", capability="local_rag"),))
        second_plan = _plan_decision(
            (_task(task_id="local", capability="local_rag"),),
            action="revise",
        )
        graph, _store, dispatcher, planner = self._build(
            plan_decisions=(first_plan, second_plan),
            dispatch_results=(
                _result(status="insufficient_evidence", task_id="local"),
                _result(status="insufficient_evidence", task_id="local"),
                _result(task_id="local", source_id="chunk-1"),
            ),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较 RAG 与 GraphRAG",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(planner.calls, 2)
        self.assertEqual(len(dispatcher.calls), 3)
        self.assertIn("[local_rag]", state["final_answer"])

    async def test_approval_pause_commits_waiting(self) -> None:
        plan = _plan_decision((_task(task_id="save", capability="dynamic_tools"),))
        graph, store, _dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(
                _result(
                    status="waiting_approval",
                    capability="dynamic_tools",
                    task_id="save",
                    citation_kind="none",
                ),
            ),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="保存结论",
            rag_mode="preferred",
        )
        state = await self._run(graph, request)
        self.assertEqual(state["final_answer"], "等待敏感工具审批。")
        cached = store.load_agent_run("request-1")
        self.assertEqual(cached.status, "waiting_approval")
        self.assertIsNotNone(cached.pending_approval)

    async def test_budget_exhaustion_marks_task_failed(self) -> None:
        plan = _plan_decision(
            (
                _task(task_id="local", capability="local_rag"),
                _task(task_id="other", capability="local_rag"),
            )
        )
        graph, store, dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(
                _result(status="insufficient_evidence", task_id="local"),
            ),
            max_child_calls=1,
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较 RAG 与 GraphRAG",
            rag_mode="preferred",
        )
        await self._run(graph, request)
        workspace = store.load_workspace("conversation-1")
        statuses = [task.status for task in workspace.task_plan.tasks]
        self.assertIn("failed", statuses)
        self.assertEqual(len(dispatcher.calls), 1)

    async def test_cached_request_returns_cached_answer(self) -> None:
        plan = _plan_decision((_task(task_id="local", capability="local_rag"),))
        graph, _store, _dispatcher, _planner = self._build(
            plan_decisions=(plan,),
            dispatch_results=(_result(task_id="local", source_id="chunk-1"),),
        )
        request = MainAgentRequest(
            request_id="request-1",
            conversation_id="conversation-1",
            message="比较 RAG 与 GraphRAG",
            rag_mode="preferred",
        )
        first = await self._run(graph, request)
        second = await self._run(graph, request)
        self.assertEqual(first["termination_reason"], "completed")
        self.assertEqual(second["termination_reason"], "cached")
        self.assertEqual(second["final_answer"], first["final_answer"])


if __name__ == "__main__":
    unittest.main()
