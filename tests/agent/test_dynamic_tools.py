from __future__ import annotations

import time
import unittest
from collections import deque
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from paper_research_agent.agent.dynamic.graph import build_dynamic_tool_graph
from paper_research_agent.agent.dynamic.memory import MemoryProposal
from paper_research_agent.agent.dynamic.models import ToolDecision, validate_tool_decision
from paper_research_agent.agent.dynamic.runtime import DynamicResearchRuntime
from paper_research_agent.agent.observability import AgentEvent
from paper_research_agent.agent.tooling.catalog import ToolSpec
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.registry import (
    RegisteredTool,
    ToolRegistrySnapshot,
    builtin_registry_snapshot,
)


class ValidationProvider:
    provider_id = "builtin"

    async def execute(self, *args: Any, **kwargs: Any) -> ToolExecutionResult:
        del args, kwargs
        raise AssertionError("validation provider must not execute")


class McpValidationProvider(ValidationProvider):
    provider_id = "zotero"


class SequenceRouter:
    def __init__(self, *decisions: ToolDecision):
        self.decisions = deque(decisions)

    async def decide(
        self,
        question: str,
        observations: tuple[Any, ...],
        memory_context: tuple[dict[str, object], ...],
        *,
        remaining_steps: int,
        child_context: dict[str, object] | None = None,
    ) -> ToolDecision:
        del question, observations, memory_context, remaining_steps, child_context
        return self.decisions.popleft()


class FakeToolkit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.approved_ids: list[str] = []
        self.memory_items: tuple[dict[str, Any], ...] = ()
        self.memory_recalls: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> ToolExecutionResult:
        del run_id
        if tool_name == "manage_long_term_memory" and arguments.get("action") in {
            "search",
            "list",
        }:
            self.memory_recalls.append(arguments)
            return ToolExecutionResult(
                tool_name=tool_name,
                status="ok" if self.memory_items else "not_found",
                trust="research_context",
                items=self.memory_items,
            )
        self.calls.append((tool_name, arguments))
        if tool_name == "save_research_note" and "approval_token" not in arguments:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="approval_required",
                trust="side_effect",
                summary={
                    "approval_request_id": "a" * 32,
                    "arguments_sha256": "b" * 64,
                    "expires_at_epoch": time.time() + 60,
                },
            )
        if tool_name == "manage_long_term_memory" and "approval_token" not in arguments:
            return ToolExecutionResult(
                tool_name=tool_name,
                status="approval_required",
                trust="side_effect",
                summary={
                    "approval_request_id": "a" * 32,
                    "arguments_sha256": "b" * 64,
                    "expires_at_epoch": time.time() + 60,
                },
            )
        if tool_name == "manage_long_term_memory":
            return ToolExecutionResult(
                tool_name=tool_name,
                trust="side_effect",
                items=({"memory_id": "e" * 32, "action": arguments["action"]},),
            )
        if tool_name == "save_research_note":
            return ToolExecutionResult(
                tool_name=tool_name,
                trust="side_effect",
                items=({"note_id": "c" * 32},),
            )
        return ToolExecutionResult(
            tool_name=tool_name,
            trust="computed_result",
            items=({"value": 42},),
        )

    def approve(self, request_id: str) -> str:
        self.approved_ids.append(request_id)
        return "d" * 64


class EventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def write(self, event: AgentEvent) -> bool:
        self.events.append(event)
        return True


class FakeMemoryProposer:
    def __init__(self, proposal: MemoryProposal):
        self.proposal = proposal
        self.calls: list[tuple[str, tuple[dict[str, Any], ...]]] = []

    async def propose(
        self,
        question: str,
        memories: tuple[dict[str, Any], ...],
        observations: tuple[Any, ...],
    ) -> MemoryProposal:
        del observations
        self.calls.append((question, memories))
        return self.proposal


class DynamicToolModelTests(unittest.TestCase):
    def test_rejects_unknown_tool_and_invalid_arguments(self) -> None:
        registry = builtin_registry_snapshot(ValidationProvider())
        unknown = ToolDecision(
            action="call_tool",
            tool_name="run_shell",
            arguments={},
            purpose="Unsafe",
        )
        with self.assertRaises(PermissionError):
            validate_tool_decision(unknown, registry)
        invalid = ToolDecision(
            action="call_tool",
            tool_name="calculate",
            arguments={"expression": ""},
            purpose="Calculate",
        )
        with self.assertRaises(ValidationError):
            validate_tool_decision(invalid, registry)
        with self.assertRaises(ValidationError):
            ToolDecision(
                action="call_tool",
                tool_name="save_research_note",
                arguments={
                    "title": "Finding",
                    "content": "Text",
                    "approval_token": "a" * 64,
                },
                purpose="Bypass approval",
            )


class DynamicToolGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_mcp_tool_is_authorized_by_captured_snapshot(self) -> None:
        toolkit = FakeToolkit()
        provider = McpValidationProvider()
        tool = RegisteredTool(
            public_name="zotero__search_items",
            provider_id="zotero",
            provider_kind="mcp",
            remote_name="search_items",
            spec=ToolSpec(
                name="zotero__search_items",
                risk="local_read",
                trust="research_context",
                timeout_seconds=5,
                max_result_items=20,
                description="在本机 Zotero 文献库中搜索条目。",
            ),
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        snapshot = ToolRegistrySnapshot({tool.public_name: tool}, {"zotero": provider})
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name=tool.public_name,
                    arguments={"query": "agent"},
                    purpose="Search local library",
                ),
                ToolDecision(action="finish", purpose="Done", final_summary="完成。"),
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            registry=snapshot,
            max_steps=2,
        )
        result = await DynamicResearchRuntime(graph=graph, max_steps=2).run(
            "查我的 Zotero 文献库", thread_id="mcp-snapshot"
        )
        self.assertEqual(result.status, "completed")
        self.assertIn((tool.public_name, {"query": "agent"}), toolkit.calls)

    async def test_greeting_cannot_trigger_sensitive_note_write(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name="save_research_note",
                    arguments={"title": "Greeting", "content": "你好"},
                    purpose="Incorrectly save a greeting",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=2,
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        result = await runtime.run("你好", thread_id="greeting")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "router_finished")
        self.assertEqual(toolkit.calls, [])
        self.assertIn("没有调用敏感工具", result.final_summary or "")

    async def test_supplied_memory_skips_internal_recall(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="finish",
                    purpose="Return answer",
                    final_summary="收到已上移的记忆。",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=2,
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        result = await runtime.run(
            "继续",
            thread_id="memory-supplied",
            memory_context=(
                {"memory_id": "m" * 32, "content": "偏好", "kind": "preference"},
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(toolkit.memory_recalls, [])

    async def test_without_supplied_memory_keeps_internal_recall(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="finish",
                    purpose="Return answer",
                    final_summary="默认内部召回。",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=2,
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        result = await runtime.run("继续", thread_id="memory-fallback")

        self.assertEqual(result.status, "completed")
        self.assertTrue(toolkit.memory_recalls)

    async def test_routes_tool_then_finishes_with_trust_label(self) -> None:
        toolkit = FakeToolkit()
        event_sink = EventSink()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                    purpose="Compute the value",
                ),
                ToolDecision(
                    action="finish",
                    purpose="Return answer",
                    final_summary="The computed value is 42.",
                ),
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=3,
            checkpointer=MemorySaver(),
            event_sink=event_sink,
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=3)

        result = await runtime.run("What is 6 * 7?", thread_id="calculation")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.termination_reason, "router_finished")
        self.assertEqual(result.observations[0].result.trust, "computed_result")
        self.assertEqual(len(toolkit.calls), 1)
        completed_nodes = {
            event.name for event in event_sink.events if event.event_type == "node_completed"
        }
        self.assertEqual(
            completed_nodes,
            {
                "dynamic_recall_memory",
                "dynamic_route",
                "dynamic_execute",
                "dynamic_finalize",
            },
        )
        self.assertEqual(toolkit.memory_recalls[0]["action"], "search")

    async def test_stops_identical_repeated_tool_call(self) -> None:
        decision = ToolDecision(
            action="call_tool",
            tool_name="calculate",
            arguments={"expression": "1 + 1"},
            purpose="Compute",
        )
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(decision, decision),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=3,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=3)

        result = await runtime.run("Compute twice", thread_id="repeat")

        self.assertEqual(result.termination_reason, "repeated_tool_call")
        self.assertEqual(len(toolkit.calls), 1)

    async def test_write_pauses_then_executes_only_after_approval(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name="save_research_note",
                    arguments={
                        "title": "Finding",
                        "content": "Bounded note",
                        "source_chunk_ids": [],
                    },
                    purpose="Save the confirmed finding",
                ),
                ToolDecision(
                    action="finish",
                    purpose="Confirm save",
                    final_summary="The note was saved.",
                ),
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=3,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=3)

        paused = await runtime.run("Save my note", thread_id="approval")

        self.assertEqual(paused.status, "approval_required")
        self.assertEqual(len(toolkit.calls), 1)
        self.assertNotIn("approval_token", toolkit.calls[0][1])

        completed = await runtime.resume(thread_id="approval", approved=True)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.termination_reason, "router_finished")
        self.assertEqual(toolkit.approved_ids, ["a" * 32])
        self.assertEqual(toolkit.calls[1][1]["approval_token"], "d" * 64)

    async def test_denial_finishes_without_side_effect(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name="save_research_note",
                    arguments={"title": "Finding", "content": "No write"},
                    purpose="Save note",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=2,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)
        await runtime.run("Do not save yet", thread_id="denial")

        completed = await runtime.resume(thread_id="denial", approved=False)

        self.assertEqual(completed.termination_reason, "approval_denied")
        self.assertEqual(completed.observations[-1].result.status, "denied")
        self.assertEqual(len(toolkit.calls), 1)
        self.assertEqual(toolkit.approved_ids, [])

    async def test_explicit_memory_proposal_pauses_and_resumes_through_same_approval(self) -> None:
        toolkit = FakeToolkit()
        toolkit.memory_items = (
            {
                "memory_id": "f" * 32,
                "kind": "preference",
                "content": "Old preference",
                "version": 1,
            },
        )
        proposer = FakeMemoryProposer(
            MemoryProposal(
                action="add",
                kind="preference",
                content="回答时优先使用中文",
                rationale="The user explicitly asked to remember this preference.",
            )
        )
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="finish",
                    purpose="Answer",
                    final_summary="我会优先使用中文。",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            memory_proposer=proposer,
            max_steps=2,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        paused = await runtime.run("请记住我偏好中文回答", thread_id="memory-add")

        self.assertEqual(paused.status, "approval_required")
        self.assertEqual(toolkit.memory_recalls[0]["action"], "list")
        self.assertEqual(proposer.calls[0][1][0]["memory_id"], "f" * 32)
        self.assertEqual(toolkit.calls[0][0], "manage_long_term_memory")
        self.assertNotIn("approval_token", toolkit.calls[0][1])

        completed = await runtime.resume(thread_id="memory-add", approved=True)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.final_summary, "我会优先使用中文。")
        self.assertEqual(completed.observations[-1].result.trust, "side_effect")
        self.assertEqual(toolkit.calls[1][1]["approval_token"], "d" * 64)

    async def test_none_memory_proposal_finishes_without_write(self) -> None:
        toolkit = FakeToolkit()
        proposer = FakeMemoryProposer(
            MemoryProposal(action="none", rationale="No explicit memory request.")
        )
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="finish",
                    purpose="Answer",
                    final_summary="Ordinary answer.",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            memory_proposer=proposer,
            max_steps=2,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        result = await runtime.run("Explain RAG", thread_id="no-memory")

        self.assertEqual(result.status, "completed")
        self.assertEqual(toolkit.calls, [])

    async def test_router_cannot_bypass_memory_proposal_for_mutation(self) -> None:
        toolkit = FakeToolkit()
        graph = build_dynamic_tool_graph(
            router=SequenceRouter(
                ToolDecision(
                    action="call_tool",
                    tool_name="manage_long_term_memory",
                    arguments={
                        "action": "add",
                        "kind": "preference",
                        "content": "Skip the proposal gate",
                    },
                    purpose="Mutate memory directly",
                )
            ),
            toolkit=toolkit,  # type: ignore[arg-type]
            max_steps=2,
            checkpointer=MemorySaver(),
        )
        runtime = DynamicResearchRuntime(graph=graph, max_steps=2)

        with self.assertRaisesRegex(PermissionError, "cannot directly mutate"):
            await runtime.run("Remember this", thread_id="memory-bypass")

        self.assertEqual(toolkit.calls, [])


if __name__ == "__main__":
    unittest.main()
