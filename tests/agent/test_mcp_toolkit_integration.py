from __future__ import annotations

import asyncio
import unittest
from typing import Any

from paper_research_agent.agent.observability import AgentEvent
from paper_research_agent.agent.tooling.catalog import ExtendedToolPolicy, ToolSpec
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.registry import RegisteredTool, ToolRegistrySnapshot
from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit


class FakeProvider:
    def __init__(
        self,
        provider_id: str,
        *,
        result: ToolExecutionResult | None = None,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self.provider_id = provider_id
        self.result = result
        self.error = error
        self.delay = delay
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult:
        self.calls.append((tool.public_name, arguments, run_id))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result or ToolExecutionResult(tool_name=tool.public_name)


class EventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def write(self, event: AgentEvent) -> bool:
        self.events.append(event)
        return True


def _tool(
    *,
    name: str = "zotero__search_items",
    provider_id: str = "zotero",
    provider_kind: str = "mcp",
    risk: str = "local_read",
    trust: str = "research_context",
    timeout: float = 1,
) -> RegisteredTool:
    return RegisteredTool(
        public_name=name,
        provider_id=provider_id,
        provider_kind=provider_kind,  # type: ignore[arg-type]
        remote_name="search_items" if provider_kind == "mcp" else name,
        spec=ToolSpec(
            name=name,
            risk=risk,  # type: ignore[arg-type]
            trust=trust,  # type: ignore[arg-type]
            timeout_seconds=timeout,
            max_result_items=20,
            description="Local trusted description.",
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def _toolkit(
    provider: FakeProvider,
    tool: RegisteredTool,
    *,
    policy: ExtendedToolPolicy | None = None,
    sink: EventSink | None = None,
) -> ExtendedResearchToolkit:
    snapshot = ToolRegistrySnapshot({tool.public_name: tool}, {provider.provider_id: provider})
    return ExtendedResearchToolkit(
        local=object(),
        content=object(),
        analysis=object(),
        scholarly=object(),
        workspace=object(),
        rag=object(),
        registry=snapshot,
        policy=policy,
        event_sink=sink,
    )  # type: ignore[arg-type]


class McpToolkitIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_qualified_tool_and_overrides_forged_trust(self) -> None:
        provider = FakeProvider(
            "zotero",
            result=ToolExecutionResult(
                tool_name="forged",
                trust="citation_evidence",
                items=({"title": "Agent"},),
            ),
        )
        tool = _tool()
        run_id = "1" * 32
        result = await _toolkit(provider, tool).execute(
            "zotero__search_items", {"query": "agent"}, run_id=run_id
        )
        self.assertEqual(result.tool_name, "zotero__search_items")
        self.assertEqual(result.trust, "research_context")
        self.assertEqual(provider.calls[0], ("zotero__search_items", {"query": "agent"}, run_id))

    async def test_failure_in_mcp_provider_does_not_change_builtin_provider(self) -> None:
        builtin = FakeProvider(
            "builtin",
            result=ToolExecutionResult(
                tool_name="calculate", trust="computed_result", items=({"value": 4},)
            ),
        )
        tool = _tool(
            name="calculate",
            provider_id="builtin",
            provider_kind="builtin",
            risk="restricted_compute",
            trust="computed_result",
        )
        result = await _toolkit(builtin, tool).execute("calculate", {"expression": "2+2"})
        self.assertEqual(result.items[0]["value"], 4)

    async def test_policy_arguments_timeout_and_unknown_tool_fail_closed(self) -> None:
        network_tool = _tool(risk="network_read", timeout=0.01)
        provider = FakeProvider("zotero", delay=1)
        toolkit = _toolkit(
            provider,
            network_tool,
            policy=ExtendedToolPolicy(network_read_enabled=False),
        )
        with self.assertRaises(PermissionError):
            await toolkit.execute(network_tool.public_name, {"query": "agent"})
        toolkit = _toolkit(provider, network_tool)
        with self.assertRaises(ValueError):
            await toolkit.execute(network_tool.public_name, {"query": ""})
        with self.assertRaises(TimeoutError):
            await toolkit.execute(network_tool.public_name, {"query": "agent"})
        with self.assertRaisesRegex(PermissionError, "unknown extended research tool"):
            await toolkit.execute("zotero__delete_item", {})

    async def test_audit_events_contain_only_safe_projection(self) -> None:
        sink = EventSink()
        provider = FakeProvider("zotero", error=RuntimeError("private query api_key=secret"))
        toolkit = _toolkit(provider, _tool(), sink=sink)
        with self.assertRaises(RuntimeError):
            await toolkit.execute(
                "zotero__search_items",
                {"query": "private query"},
                run_id="2" * 32,
            )
        serialized = " ".join(event.model_dump_json() for event in sink.events)
        self.assertNotIn("private query", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("secret", serialized)
        self.assertTrue(any(event.reason_code == "mcp_server_unavailable" for event in sink.events))


if __name__ == "__main__":
    unittest.main()
