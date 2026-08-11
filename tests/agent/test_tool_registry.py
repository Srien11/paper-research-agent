from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from typing import Any

from paper_research_agent.agent.tooling.catalog import (
    EXTENDED_TOOL_NAMES,
    EXTENDED_TOOL_SPECS,
    ToolSpec,
)
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.registry import (
    RegisteredTool,
    ToolRegistrySnapshot,
    builtin_registry_snapshot,
)


class FakeProvider:
    def __init__(self, provider_id: str = "builtin") -> None:
        self.provider_id = provider_id

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(tool_name=tool.public_name)


def _mcp_tool(**updates: Any) -> RegisteredTool:
    values: dict[str, Any] = {
        "public_name": "zotero__search_items",
        "provider_id": "zotero",
        "provider_kind": "mcp",
        "remote_name": "search_items",
        "spec": ToolSpec(
            name="zotero__search_items",
            risk="local_read",
            trust="research_context",
            timeout_seconds=5,
            approval_required=False,
            max_result_items=20,
            description="在本机 Zotero 文献库中搜索条目。",
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    values.update(updates)
    return RegisteredTool(**values)


class ToolRegistryTests(unittest.TestCase):
    def test_registry_keeps_exact_builtin_catalog(self) -> None:
        snapshot = builtin_registry_snapshot(FakeProvider())
        self.assertEqual(snapshot.names, EXTENDED_TOOL_NAMES)
        self.assertEqual(len(EXTENDED_TOOL_SPECS), 18)

    def test_registry_adds_qualified_mcp_without_changing_builtin_count(self) -> None:
        builtin = builtin_registry_snapshot(FakeProvider())
        tools = {tool.public_name: tool for tool in builtin.list_tools()}
        tools["zotero__search_items"] = _mcp_tool()
        snapshot = ToolRegistrySnapshot(
            tools,
            {"builtin": FakeProvider(), "zotero": FakeProvider("zotero")},
        )
        self.assertIn("zotero__search_items", snapshot.names)
        self.assertEqual(len(EXTENDED_TOOL_SPECS), 18)

    def test_registry_rejects_unsafe_mcp_contracts(self) -> None:
        unsafe_specs = (
            ToolSpec(
                name="zotero__search_items",
                risk="local_read",
                trust="citation_evidence",
                timeout_seconds=5,
                max_result_items=20,
                description="unsafe",
            ),
            ToolSpec(
                name="zotero__search_items",
                risk="write",
                trust="research_context",
                timeout_seconds=5,
                approval_required=True,
                max_result_items=20,
                description="unsafe",
            ),
        )
        for spec in unsafe_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                _mcp_tool(spec=spec)

    def test_registry_rejects_wrong_namespace_and_missing_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "server namespace"):
            _mcp_tool(
                public_name="other__search_items",
                spec=_mcp_tool().spec.model_copy(update={"name": "other__search_items"}),
            )
        with self.assertRaisesRegex(ValueError, "provider"):
            ToolRegistrySnapshot(
                {"zotero__search_items": _mcp_tool()},
                {"builtin": FakeProvider()},
            )

    def test_registry_is_deeply_immutable(self) -> None:
        tool = _mcp_tool()
        with self.assertRaises(TypeError):
            tool.input_schema["properties"]["query"]["type"] = "integer"
        with self.assertRaises(FrozenInstanceError):
            tool.public_name = "zotero__get_item"
        snapshot = ToolRegistrySnapshot(
            {tool.public_name: tool},
            {"zotero": FakeProvider("zotero")},
        )
        with self.assertRaises(TypeError):
            snapshot.tools[tool.public_name] = tool

    def test_list_is_stably_sorted_and_unknown_tool_fails_closed(self) -> None:
        first = _mcp_tool()
        second = _mcp_tool(
            public_name="zotero__get_item",
            remote_name="get_item",
            spec=_mcp_tool().spec.model_copy(update={"name": "zotero__get_item"}),
        )
        snapshot = ToolRegistrySnapshot(
            {first.public_name: first, second.public_name: second},
            {"zotero": FakeProvider("zotero")},
        )
        self.assertEqual(
            [tool.public_name for tool in snapshot.list_tools()],
            ["zotero__get_item", "zotero__search_items"],
        )
        with self.assertRaisesRegex(PermissionError, "not registered"):
            snapshot.resolve("zotero__delete_item")


if __name__ == "__main__":
    unittest.main()
