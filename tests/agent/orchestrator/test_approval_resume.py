from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.artifacts import DynamicToolArtifact
from paper_research_agent.agent.orchestrator.graph import MainAgentApprovalResumer
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ChildTaskResult,
    ConversationWorkspace,
    GoalState,
    MainAgentRequest,
    MainAgentResult,
    TaskPlan,
)
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
from paper_research_agent.conversation.models import ConversationResolution
from paper_research_agent.conversation.store import InMemoryConversationStore


def _request() -> MainAgentRequest:
    return MainAgentRequest(
        request_id="request-1",
        conversation_id="conversation-1",
        message="保存研究结论",
        rag_mode="preferred",
    )


class _FakeGraph:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, value: object, config: object = None) -> dict[str, object]:
        del value, config
        self.calls += 1
        return {
            "run_id": "r" * 32,
            "base_workspace_version": 0,
            "final_answer": "完成",
            "termination_reason": "completed",
            "child_results": [],
        }


class _ResumeDispatcher:
    def __init__(self, *, outcome: str = "completed") -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, str, bool]] = []
        self.side_effects = 0

    async def resume_dynamic_tools(
        self, request: object, *, approved: bool
    ) -> ChildTaskResult:
        self.calls.append((request.conversation_id, request.task_id, approved))
        if self.outcome == "completed":
            if approved:
                self.side_effects += 1
            return ChildTaskResult(
                child_run_id="d" * 32,
                task_id=request.task_id,
                capability="dynamic_tools",
                status="completed",
                summary="审批恢复完成",
                citation_kind="external",
                artifact=DynamicToolArtifact(text="审批恢复完成"),
            )
        return ChildTaskResult(
            child_run_id="d" * 32,
            task_id=request.task_id,
            capability="dynamic_tools",
            status="failed",
            summary="审批被拒绝" if self.outcome == "denied" else "审批已过期",
            citation_kind="none",
            error_code=(
                "approval_denied" if self.outcome == "denied" else "approval_expired"
            ),
        )


def _utc() -> datetime:
    return datetime(2026, 8, 10, tzinfo=UTC)


def _seed_waiting(
    store: InMemoryConversationStore,
    *,
    request_id: str,
    conversation_id: str,
    task_id: str,
) -> None:
    start = store.begin_agent_run(
        request_id=request_id,
        conversation_id=conversation_id,
        user_question="保存研究结论",
    )
    goal = GoalState(
        goal_id="a" * 32,
        objective="保存研究结论",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )
    task = AgentTask(
        task_id=task_id,
        goal_id=goal.goal_id,
        title="保存结论",
        objective="保存研究结论",
        success_criteria=("完成保存",),
        capability="dynamic_tools",
        status="waiting_approval",
    )
    plan = TaskPlan(
        plan_id="c" * 32,
        goal_id=goal.goal_id,
        revision=1,
        tasks=(task,),
        created_at=_utc(),
        updated_at=_utc(),
    )
    workspace = ConversationWorkspace(
        conversation_id=conversation_id,
        version=start.workspace.version,
        active_goal=goal,
        task_plan=plan,
        updated_at=_utc(),
    )
    pending = {
        "task_id": task_id,
        "tool_name": "save_research_note",
        "arguments": {"content": "validated"},
        "purpose": "保存结论",
        "decision_fingerprint": "e" * 64,
        "approval_request_id": "f" * 32,
        "arguments_sha256": "1" * 64,
        "expires_at_epoch": 2_000_000_000,
    }
    child = ChildTaskResult(
        child_run_id="d" * 32,
        task_id=task_id,
        capability="dynamic_tools",
        status="waiting_approval",
        summary="等待审批",
        pending_approval=pending,
    )
    result = MainAgentResult(
        run_id=start.run_id,
        request_id=request_id,
        conversation_id=conversation_id,
        status="waiting_approval",
        answer="等待敏感工具审批。",
        route_trace=("dynamic_tools",),
        child_results=(child,),
        pending_approval=pending,
        workspace_version=1,
    )
    store.commit_agent_run(
        run_id=start.run_id,
        turn_id=start.turn_id,
        expected_workspace_version=start.workspace.version,
        workspace=workspace,
        route="main_agent",
        status="pending",
        resolution=ConversationResolution(
            original_question="保存研究结论",
            standalone_question="保存研究结论",
            chinese_query="保存研究结论",
            confidence=1,
        ),
        assistant_summary="等待审批",
        source_ids=(),
        result=result,
    )


class ApprovalResumeTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(
        self,
        graph: _FakeGraph,
        resumer: object,
    ) -> MainAgentRuntime:
        return MainAgentRuntime(
            graph=graph,
            repository=InMemoryConversationStore(),
            approval_resumer=resumer,  # type: ignore[arg-type]
        )

    async def test_resume_does_not_rerun_interpret_goal_plan(self) -> None:
        graph = _FakeGraph()

        async def resumer(request_id: str, approved: bool) -> MainAgentResult:
            del request_id, approved
            return MainAgentResult(
                run_id="r" * 32,
                request_id="request-1",
                conversation_id="conversation-1",
                status="completed",
                answer="恢复后完成",
                workspace_version=1,
            )

        runtime = self._runtime(graph, resumer)
        result = await runtime.resume_approval(request_id="request-1", approved=True)
        self.assertEqual(graph.calls, 0)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "恢复后完成")

    async def test_denied_resume_is_a_legal_terminal_state(self) -> None:
        graph = _FakeGraph()

        async def resumer(request_id: str, approved: bool) -> MainAgentResult:
            del approved
            return MainAgentResult(
                run_id="r" * 32,
                request_id=request_id,
                conversation_id="conversation-1",
                status="completed",
                answer="审批被拒绝，未执行写入。",
                workspace_version=1,
            )

        runtime = self._runtime(graph, resumer)
        result = await runtime.resume_approval(request_id="request-1", approved=False)
        self.assertEqual(result.answer, "审批被拒绝，未执行写入。")
        self.assertIsNone(result.pending_approval)

    async def test_expired_resume_is_a_legal_terminal_state(self) -> None:
        graph = _FakeGraph()

        async def resumer(request_id: str, approved: bool) -> MainAgentResult:
            del approved
            return MainAgentResult(
                run_id="r" * 32,
                request_id=request_id,
                conversation_id="conversation-1",
                status="completed",
                answer="审批已过期。",
                workspace_version=1,
            )

        runtime = self._runtime(graph, resumer)
        result = await runtime.resume_approval(request_id="request-1", approved=True)
        self.assertEqual(result.answer, "审批已过期。")
        self.assertIsNone(result.pending_approval)

    async def test_resume_requires_configured_resumer(self) -> None:
        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=InMemoryConversationStore(),
        )
        with self.assertRaises(RuntimeError):
            await runtime.resume_approval(request_id="request-1", approved=True)

    async def test_builtin_resume_completes_only_matching_request(self) -> None:
        store = InMemoryConversationStore()
        _seed_waiting(
            store,
            request_id="request-a",
            conversation_id="conversation-a",
            task_id="task-a",
        )
        _seed_waiting(
            store,
            request_id="request-b",
            conversation_id="conversation-b",
            task_id="task-b",
        )
        dispatcher = _ResumeDispatcher()
        resumer = MainAgentApprovalResumer(
            repository=store,
            dispatcher=dispatcher,  # type: ignore[arg-type]
        )
        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=store,
            approval_resumer=resumer.resume,
        )

        first = await runtime.resume_approval(request_id="request-a", approved=True)
        second = await runtime.resume_approval(request_id="request-b", approved=True)

        self.assertEqual(first.conversation_id, "conversation-a")
        self.assertEqual(second.conversation_id, "conversation-b")
        self.assertEqual(
            dispatcher.calls,
            [
                ("conversation-a", "task-a", True),
                ("conversation-b", "task-b", True),
            ],
        )
        self.assertEqual(
            store.load_workspace("conversation-a").task_plan.tasks[0].status,
            "completed",
        )

    async def test_old_request_cannot_approve_new_request(self) -> None:
        store = InMemoryConversationStore()
        _seed_waiting(
            store,
            request_id="request-old",
            conversation_id="conversation-1",
            task_id="task-old",
        )
        dispatcher = _ResumeDispatcher()
        resumer = MainAgentApprovalResumer(
            repository=store,
            dispatcher=dispatcher,  # type: ignore[arg-type]
        )
        runtime = MainAgentRuntime(
            graph=_FakeGraph(), repository=store, approval_resumer=resumer.resume
        )
        await runtime.resume_approval(request_id="request-old", approved=False)
        _seed_waiting(
            store,
            request_id="request-new",
            conversation_id="conversation-1",
            task_id="task-new",
        )

        with self.assertRaises(RuntimeError):
            await runtime.resume_approval(request_id="request-old", approved=True)

        self.assertEqual(len(dispatcher.calls), 1)
        self.assertEqual(
            store.load_agent_run("request-new").status,
            "waiting_approval",
        )

    async def test_duplicate_approval_executes_side_effect_once(self) -> None:
        store = InMemoryConversationStore()
        _seed_waiting(
            store,
            request_id="request-once",
            conversation_id="conversation-1",
            task_id="task-once",
        )
        dispatcher = _ResumeDispatcher()
        resumer = MainAgentApprovalResumer(
            repository=store,
            dispatcher=dispatcher,  # type: ignore[arg-type]
        )
        runtime = MainAgentRuntime(
            graph=_FakeGraph(), repository=store, approval_resumer=resumer.resume
        )

        outcomes = await asyncio.gather(
            runtime.resume_approval(request_id="request-once", approved=True),
            runtime.resume_approval(request_id="request-once", approved=True),
            return_exceptions=True,
        )

        self.assertEqual(dispatcher.side_effects, 1)
        self.assertEqual(sum(isinstance(item, MainAgentResult) for item in outcomes), 1)

    async def test_denied_and_expired_resume_fail_task_without_side_effect(self) -> None:
        for label, outcome in (("denied", "denied"), ("expired", "expired")):
            with self.subTest(outcome=label):
                store = InMemoryConversationStore()
                _seed_waiting(
                    store,
                    request_id=f"request-{label}",
                    conversation_id=f"conversation-{label}",
                    task_id=f"task-{label}",
                )
                dispatcher = _ResumeDispatcher(outcome=outcome)
                resumer = MainAgentApprovalResumer(
                    repository=store,
                    dispatcher=dispatcher,  # type: ignore[arg-type]
                )
                runtime = MainAgentRuntime(
                    graph=_FakeGraph(),
                    repository=store,
                    approval_resumer=resumer.resume,
                )

                result = await runtime.resume_approval(
                    request_id=f"request-{label}", approved=True
                )

                self.assertEqual(result.status, "completed")
                self.assertIsNone(result.pending_approval)
                self.assertEqual(dispatcher.side_effects, 0)
                task = store.load_workspace(
                    f"conversation-{label}"
                ).task_plan.tasks[0]
                self.assertEqual(task.status, "failed")


if __name__ == "__main__":
    unittest.main()
