from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from paper_research_agent.agent.dynamic.models import (
    DynamicResearchResult,
    PendingApproval,
)
from paper_research_agent.agent.dynamic.runtime import DynamicResearchRuntime
from paper_research_agent.agent.orchestrator.children import ChildGraphDispatcher
from paper_research_agent.agent.orchestrator.models import (
    ChildTaskRequest,
    ContextMessage,
    RecalledContext,
)


def _request(**overrides: object) -> ChildTaskRequest:
    values: dict[str, object] = {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "goal_id": "a" * 32,
        "task_id": "task-1",
        "objective": "比较 RAG 与 GraphRAG 的评测指标",
        "success_criteria": ("找到至少两篇论文证据",),
        "capability": "local_rag",
        "current_message": "比较 RAG 与 GraphRAG",
        "conversation_summary": "用户在做模型测评",
        "constraints": ("只依据本地论文",),
        "rag_mode": "preferred",
    }
    values.update(overrides)
    return ChildTaskRequest(**values)


class _FakeLocalResult:
    def __init__(self, *, sufficient: bool = True) -> None:
        self.run_id = "r" * 32
        self.evidence_sufficient = sufficient
        self.observations = (SimpleNamespace(objective="检索 RAG 评测指标"),)
        self.evidence = (SimpleNamespace(chunk_id="chunk-1"),) if sufficient else ()
        self.question = "检索 RAG 评测指标"


class _FakeLocalExecutor:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, str, bool]] = []

    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        planning_required: bool = False,
    ) -> object:
        self.calls.append((question, thread_id, planning_required))
        return self.result


class _FakeDynamicExecutor:
    def __init__(self, result: DynamicResearchResult) -> None:
        self.result = result
        self.calls: list[
            tuple[
                str,
                str,
                tuple[dict[str, object], ...],
                dict[str, object] | None,
            ]
        ] = []

    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        memory_context: tuple[dict[str, object], ...] = (),
        child_context: dict[str, object] | None = None,
    ) -> DynamicResearchResult:
        self.calls.append((question, thread_id, memory_context, child_context))
        return self.result


class _FakeGraph:
    def __init__(self) -> None:
        self.inputs: list[dict[str, object]] = []

    async def ainvoke(self, value: dict[str, object], config: object = None, **kwargs: object) -> object:
        del config, kwargs
        self.inputs.append(value)
        return {
            "run_id": "a" * 32,
            "observations": [],
            "pending_approval": None,
            "final_summary": "已完成核验",
            "termination_reason": "router_finished",
            "next_action": "finish",
        }


def _pending() -> PendingApproval:
    return PendingApproval(
        tool_name="save_research_note",
        arguments={"title": "结论"},
        purpose="保存结论",
        decision_fingerprint="d" * 64,
        approval_request_id="a" * 32,
        arguments_sha256="e" * 64,
        expires_at_epoch=123.0,
    )


class ChildGraphDispatcherTests(unittest.TestCase):
    def test_local_uses_task_objective_as_question(self) -> None:
        fake = _FakeLocalExecutor(_FakeLocalResult(sufficient=True))
        dispatcher = ChildGraphDispatcher(local_rag=fake)
        result = asyncio.run(dispatcher.dispatch(_request(capability="local_rag")))
        self.assertEqual(fake.calls[0][0], "比较 RAG 与 GraphRAG 的评测指标")
        self.assertTrue(fake.calls[0][2])
        self.assertIn("research::", fake.calls[0][1])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.citation_kind, "local_paper")
        self.assertEqual(result.source_ids, ("chunk-1",))

    def test_local_insufficient_evidence_is_not_success(self) -> None:
        fake = _FakeLocalExecutor(_FakeLocalResult(sufficient=False))
        dispatcher = ChildGraphDispatcher(local_rag=fake)
        result = asyncio.run(dispatcher.dispatch(_request(capability="local_rag")))
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citation_kind, "local_paper")
        self.assertEqual(result.source_ids, ())

    def test_single_paper_local_question_does_not_force_comparison_plan(self) -> None:
        fake = _FakeLocalExecutor(_FakeLocalResult(sufficient=True))
        dispatcher = ChildGraphDispatcher(local_rag=fake)

        asyncio.run(
            dispatcher.dispatch(
                _request(
                    capability="local_rag",
                    objective="总结 C001 的主要方法",
                )
            )
        )

        self.assertFalse(fake.calls[0][2])

    def test_dynamic_receives_goal_criteria_and_constraints(self) -> None:
        result = DynamicResearchResult(
            run_id="a" * 32,
            thread_id="t",
            status="completed",
            final_summary="已核验 2026 年维护状态",
        )
        fake = _FakeDynamicExecutor(result)
        dispatcher = ChildGraphDispatcher(dynamic_tools=fake)
        request = _request(capability="dynamic_tools", task_id="task-2")
        child = asyncio.run(dispatcher.dispatch(request))
        child_context = fake.calls[0][3]
        self.assertIsNotNone(child_context)
        self.assertEqual(child_context["goal_id"], "a" * 32)
        self.assertEqual(child_context["constraints"], ["只依据本地论文"])
        self.assertEqual(child_context["success_criteria"], ["找到至少两篇论文证据"])
        self.assertIn("dynamic::", fake.calls[0][1])
        self.assertEqual(child.status, "completed")
        self.assertEqual(child.citation_kind, "external")

    def test_dynamic_memory_context_projected_from_selected_context(self) -> None:
        request = _request(
            capability="dynamic_tools",
            task_id="task-m",
            selected_context=(
                RecalledContext(
                    source_id="m" * 32,
                    kind="long_term_memory",
                    content="用户偏好用中文回答",
                    relevance=0.6,
                    trust="research_context",
                ),
            ),
        )
        fake = _FakeDynamicExecutor(
            DynamicResearchResult(
                run_id="a" * 32,
                thread_id="t",
                status="completed",
                final_summary="完成",
            )
        )
        dispatcher = ChildGraphDispatcher(dynamic_tools=fake)
        asyncio.run(dispatcher.dispatch(request))
        self.assertEqual(fake.calls[0][2][0]["memory_id"], "m" * 32)
        self.assertEqual(fake.calls[0][2][0]["content"], "用户偏好用中文回答")

    def test_dynamic_without_memory_falls_back_to_internal_recall(self) -> None:
        fake = _FakeDynamicExecutor(
            DynamicResearchResult(
                run_id="a" * 32,
                thread_id="t",
                status="completed",
                final_summary="完成",
            )
        )
        dispatcher = ChildGraphDispatcher(dynamic_tools=fake)
        asyncio.run(dispatcher.dispatch(_request(capability="dynamic_tools", task_id="task-x")))
        self.assertEqual(fake.calls[0][2], ())

    def test_runtime_passes_memory_supplied_flag_when_provided(self) -> None:
        graph = _FakeGraph()
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)
        result = asyncio.run(
            runtime.run(
                "核验最新维护状态",
                thread_id="thread-1",
                memory_context=({"memory_id": "m1", "content": "偏好"},),
                child_context={"goal_id": "a" * 32},
            )
        )
        self.assertTrue(graph.inputs[0]["memory_supplied"])
        self.assertEqual(graph.inputs[0]["memory_context"], [{"memory_id": "m1", "content": "偏好"}])
        self.assertEqual(graph.inputs[0]["child_context"], {"goal_id": "a" * 32})
        self.assertEqual(result.status, "completed")

    def test_runtime_without_memory_falls_back_to_internal_recall(self) -> None:
        graph = _FakeGraph()
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)
        asyncio.run(runtime.run("核验最新维护状态", thread_id="thread-1"))
        self.assertFalse(graph.inputs[0]["memory_supplied"])
        self.assertEqual(graph.inputs[0]["memory_context"], [])

    def test_dynamic_approval_is_projected_as_waiting(self) -> None:
        result = DynamicResearchResult(
            run_id="a" * 32,
            thread_id="t",
            status="approval_required",
            pending_approval=_pending(),
        )
        fake = _FakeDynamicExecutor(result)
        dispatcher = ChildGraphDispatcher(dynamic_tools=fake)
        child = asyncio.run(
            dispatcher.dispatch(_request(capability="dynamic_tools", task_id="task-3"))
        )
        self.assertEqual(child.status, "waiting_approval")
        self.assertIsNotNone(child.pending_approval)
        self.assertEqual(child.citation_kind, "none")

    def test_local_and_external_citations_never_confuse(self) -> None:
        local_fake = _FakeLocalExecutor(_FakeLocalResult(sufficient=True))
        dynamic_fake = _FakeDynamicExecutor(
            DynamicResearchResult(
                run_id="a" * 32,
                thread_id="t",
                status="completed",
                final_summary="外部信息",
            )
        )
        dispatcher = ChildGraphDispatcher(local_rag=local_fake, dynamic_tools=dynamic_fake)
        local = asyncio.run(dispatcher.dispatch(_request(capability="local_rag")))
        dynamic = asyncio.run(
            dispatcher.dispatch(_request(capability="dynamic_tools", task_id="task-4"))
        )
        self.assertEqual(local.citation_kind, "local_paper")
        self.assertEqual(dynamic.citation_kind, "external")

    def test_unavailable_executor_fails_closed(self) -> None:
        dispatcher = ChildGraphDispatcher()
        result = asyncio.run(dispatcher.dispatch(_request(capability="local_rag")))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error_code, "local_rag_unavailable")

    def test_unsupported_capability_raises(self) -> None:
        dispatcher = ChildGraphDispatcher()
        with self.assertRaises(ValueError):
            asyncio.run(
                dispatcher.dispatch(_request(capability="direct_chat"))
            )

    def test_direct_response_request_consumes_orchestrator_context(self) -> None:
        from paper_research_agent.web.chat_runtime import DirectResponseRequest

        request = DirectResponseRequest(
            session_id="session-1",
            current_message="继续",
            recent_messages=(
                ContextMessage(
                    turn_id="t1", sequence=1, role="user", content="旧问题"
                ),
            ),
            active_goal="比较 RAG 与 GraphRAG",
            recalled_context=(
                RecalledContext(
                    source_id="m" * 32,
                    kind="long_term_memory",
                    content="偏好",
                    relevance=0.5,
                    trust="research_context",
                ),
            ),
        )
        self.assertEqual(request.recent_messages[0].content, "旧问题")
        self.assertEqual(request.recalled_context[0].source_id, "m" * 32)


if __name__ == "__main__":
    unittest.main()
