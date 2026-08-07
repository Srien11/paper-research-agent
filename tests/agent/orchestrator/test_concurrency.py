from __future__ import annotations

import asyncio
import unittest

from paper_research_agent.agent.orchestrator.graph import build_main_agent_graph
from paper_research_agent.agent.orchestrator.models import MainAgentRequest
from paper_research_agent.conversation.store import InMemoryConversationStore
from tests.agent.orchestrator.test_graph import (
    FakeDispatcher,
    FakeHydrator,
    FakeInterpreter,
    FakePlanner,
    FakeReconciler,
    _goal_decision,
    _interpretation,
    _plan_decision,
    _result,
    _task,
)


def _request(
    request_id: str = "request-1", conversation_id: str = "conversation-1"
) -> MainAgentRequest:
    return MainAgentRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        message="比较 RAG 与 GraphRAG",
        rag_mode="preferred",
    )


def _build_graph(
    store: InMemoryConversationStore,
    plan_decisions: tuple[object, ...],
    dispatch_results: tuple[object, ...],
) -> tuple[object, FakeDispatcher]:
    planner = FakePlanner(*plan_decisions)
    dispatcher = FakeDispatcher(*dispatch_results)
    graph = build_main_agent_graph(
        repository=store,
        hydrator=FakeHydrator(),
        interpreter=FakeInterpreter(_interpretation()),
        goal_reconciler=FakeReconciler(_goal_decision()),
        task_planner=planner,
        dispatcher=dispatcher,
    )
    return graph, dispatcher


class MainAgentConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_request_concurrent_runs_single_effect(self) -> None:
        store = InMemoryConversationStore()
        graph, dispatcher = _build_graph(
            store,
            (_plan_decision((_task(),)),),
            (_result(source_id="chunk-1"),),
        )
        request = _request()
        results = await asyncio.gather(
            graph.ainvoke({"request": request.model_dump(mode="json")}),
            graph.ainvoke({"request": request.model_dump(mode="json")}),
        )
        reasons = {item["termination_reason"] for item in results}
        self.assertTrue({"completed", "cached"} & reasons or "running_reused" in reasons)
        self.assertEqual(len(dispatcher.calls), 1)
        self.assertEqual(len(store.history("conversation-1")), 1)

    async def test_same_request_retry_does_not_duplicate_side_effect(self) -> None:
        store = InMemoryConversationStore()
        graph, dispatcher = _build_graph(
            store,
            (_plan_decision((_task(),)),),
            (_result(source_id="chunk-1"),),
        )
        request = _request()
        first = await graph.ainvoke({"request": request.model_dump(mode="json")})
        second = await graph.ainvoke({"request": request.model_dump(mode="json")})
        self.assertEqual(first["termination_reason"], "completed")
        self.assertEqual(second["termination_reason"], "cached")
        self.assertEqual(len(dispatcher.calls), 1)
        self.assertEqual(len(store.history("conversation-1")), 1)

    async def test_two_requests_same_conversation_advance_workspace_version(self) -> None:
        store = InMemoryConversationStore()
        graph, _dispatcher = _build_graph(
            store,
            (_plan_decision((_task(),)), _plan_decision((_task(),))),
            (_result(source_id="chunk-1"), _result(source_id="chunk-2")),
        )
        first_request = _request(request_id="request-1")
        second_request = _request(request_id="request-2")
        await graph.ainvoke({"request": first_request.model_dump(mode="json")})
        await graph.ainvoke({"request": second_request.model_dump(mode="json")})
        workspace = store.load_workspace("conversation-1")
        self.assertEqual(workspace.version, 2)
        self.assertIsNotNone(store.load_agent_run("request-1"))
        self.assertIsNotNone(store.load_agent_run("request-2"))

    async def test_different_conversations_run_in_parallel(self) -> None:
        store = InMemoryConversationStore()
        graph, _dispatcher = _build_graph(
            store,
            (_plan_decision((_task(),)), _plan_decision((_task(),))),
            (_result(source_id="chunk-1"), _result(source_id="chunk-2")),
        )
        first = _request(request_id="request-1", conversation_id="conversation-a")
        second = _request(request_id="request-2", conversation_id="conversation-b")
        started = asyncio.get_running_loop().time()
        await asyncio.gather(
            graph.ainvoke({"request": first.model_dump(mode="json")}),
            graph.ainvoke({"request": second.model_dump(mode="json")}),
        )
        elapsed = asyncio.get_running_loop().time() - started
        self.assertEqual(store.load_workspace("conversation-a").version, 1)
        self.assertEqual(store.load_workspace("conversation-b").version, 1)
        self.assertLess(elapsed, 5.0)

    async def test_cancelled_run_does_not_leave_conversation_blocked(self) -> None:
        store = InMemoryConversationStore()
        graph, dispatcher = _build_graph(
            store,
            (_plan_decision((_task(),)),),
            (_result(source_id="chunk-1"),),
        )
        request = _request(request_id="request-1")

        async def cancelled_invoke() -> object:
            task = asyncio.create_task(
                graph.ainvoke({"request": request.model_dump(mode="json")})
            )
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return None
            raise AssertionError("expected cancellation")

        await cancelled_invoke()
        result = await graph.ainvoke({"request": request.model_dump(mode="json")})
        self.assertIn(result.get("termination_reason"), {"cached", "running_reused", "completed"})
        self.assertLessEqual(len(dispatcher.calls), 1)


if __name__ == "__main__":
    unittest.main()
