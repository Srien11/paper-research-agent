from __future__ import annotations

import unittest

from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.langchain import build_extended_langchain_tools


class FakeToolkit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(self, name: str, arguments: dict[str, object]) -> ToolExecutionResult:
        self.calls.append((name, arguments))
        return ToolExecutionResult(tool_name=name, items=({"ok": True},))


class LangChainToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_exact_nineteen_tools_with_strict_schemas(self) -> None:
        toolkit = FakeToolkit()
        tools = build_extended_langchain_tools(toolkit)  # type: ignore[arg-type]
        self.assertEqual(len(tools), 19)
        self.assertEqual(len({tool.name for tool in tools}), 19)
        calculate = next(tool for tool in tools if tool.name == "calculate")
        result = await calculate.ainvoke({"expression": "2 + 2"})
        self.assertEqual(result["tool_name"], "calculate")
        self.assertEqual(toolkit.calls, [("calculate", {"expression": "2 + 2"})])


if __name__ == "__main__":
    unittest.main()
