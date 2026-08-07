from __future__ import annotations

import unittest

from paper_research_agent.agent.orchestrator.models import MainAgentRequest, MainAgentResult
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
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


if __name__ == "__main__":
    unittest.main()
