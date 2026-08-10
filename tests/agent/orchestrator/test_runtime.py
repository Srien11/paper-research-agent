from __future__ import annotations

import asyncio
import unittest

from paper_research_agent.agent.orchestrator.models import MainAgentRequest, MainAgentResult
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
from paper_research_agent.conversation.store import InMemoryConversationStore


def _request(
    conversation_id: str = "conversation-1", request_id: str = "request-1"
) -> MainAgentRequest:
    return MainAgentRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        message="比较 RAG 与 GraphRAG",
        rag_mode="preferred",
    )


class _FakeGraph:
    def __init__(
        self,
        delay: float = 0,
        *,
        termination_reason: str = "completed",
        base_workspace_version: int = 0,
    ) -> None:
        self.delay = delay
        self.termination_reason = termination_reason
        self.base_workspace_version = base_workspace_version
        self.calls = 0

    async def ainvoke(self, value: object, config: object = None) -> dict[str, object]:
        del value, config
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return {
            "run_id": "r" * 32,
            "base_workspace_version": self.base_workspace_version,
            "final_answer": "完成",
            "termination_reason": self.termination_reason,
            "child_results": [],
        }


class MainAgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_returns_result_from_graph(self) -> None:
        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=InMemoryConversationStore(),
            timeout_seconds=5,
        )
        result = await runtime.run(_request())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "完成")

    async def test_failed_graph_returns_failed_without_incrementing_workspace(self) -> None:
        runtime = MainAgentRuntime(
            graph=_FakeGraph(
                termination_reason="failed",
                base_workspace_version=3,
            ),
            repository=InMemoryConversationStore(),
            timeout_seconds=5,
        )

        result = await runtime.run(_request())

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.workspace_version, 3)

    async def test_run_times_out_when_graph_is_slow(self) -> None:
        runtime = MainAgentRuntime(
            graph=_FakeGraph(delay=5),
            repository=InMemoryConversationStore(),
            timeout_seconds=0.1,
        )
        with self.assertRaises(TimeoutError):
            await runtime.run(_request())

    async def test_different_conversations_run_in_parallel(self) -> None:
        graph = _FakeGraph(delay=0.2)
        runtime = MainAgentRuntime(
            graph=graph,
            repository=InMemoryConversationStore(),
            timeout_seconds=5,
        )
        started = asyncio.get_running_loop().time()
        first = asyncio.create_task(runtime.run(_request(conversation_id="a")))
        second = asyncio.create_task(runtime.run(_request(conversation_id="b")))
        results = await asyncio.gather(first, second)
        elapsed = asyncio.get_running_loop().time() - started
        self.assertTrue(all(item.status == "completed" for item in results))
        self.assertLess(elapsed, 0.4)

    async def test_same_conversation_is_serialized(self) -> None:
        graph = _FakeGraph(delay=0.1)
        runtime = MainAgentRuntime(
            graph=graph,
            repository=InMemoryConversationStore(),
            timeout_seconds=5,
        )
        started = asyncio.get_running_loop().time()
        first = asyncio.create_task(runtime.run(_request(conversation_id="a")))
        second = asyncio.create_task(runtime.run(_request(conversation_id="a")))
        await asyncio.gather(first, second)
        elapsed = asyncio.get_running_loop().time() - started
        self.assertGreaterEqual(elapsed, 0.19)

    async def test_resume_approval_delegates_to_resumer(self) -> None:
        calls: list[tuple[str, bool]] = []

        async def resumer(request_id: str, approved: bool) -> MainAgentResult:
            calls.append((request_id, approved))
            return MainAgentResult(
                run_id="r" * 32,
                request_id=request_id,
                conversation_id="conversation-1",
                status="completed",
                answer="已批准",
                workspace_version=1,
            )

        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=InMemoryConversationStore(),
            approval_resumer=resumer,
        )
        result = await runtime.resume_approval(request_id="request-1", approved=True)
        self.assertEqual(calls, [("request-1", True)])
        self.assertEqual(result.answer, "已批准")

    async def test_clear_removes_lock_and_calls_callback(self) -> None:
        cleared: list[str] = []

        async def clear(conversation_id: str) -> None:
            cleared.append(conversation_id)

        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=InMemoryConversationStore(),
            clear=clear,
        )
        await runtime.run(_request(conversation_id="a"))
        await runtime.clear("a")
        self.assertEqual(cleared, ["a"])
        self.assertNotIn("a", runtime._locks)

    async def test_closed_runtime_rejects_work(self) -> None:
        closed = False

        async def close() -> None:
            nonlocal closed
            closed = True

        runtime = MainAgentRuntime(
            graph=_FakeGraph(),
            repository=InMemoryConversationStore(),
            close=close,
        )
        await runtime.aclose()
        self.assertTrue(closed)
        with self.assertRaises(RuntimeError):
            await runtime.run(_request())
        with self.assertRaises(RuntimeError):
            await runtime.resume_approval(request_id="request-1", approved=True)


if __name__ == "__main__":
    unittest.main()
