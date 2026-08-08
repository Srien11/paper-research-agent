from __future__ import annotations

import unittest

from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.langchain import build_extended_langchain_tools
from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit


class FakeToolkit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute(self, name: str, arguments: dict[str, object]) -> ToolExecutionResult:
        self.calls.append((name, arguments))
        return ToolExecutionResult(tool_name=name, items=({"ok": True},))


class LangChainToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_exact_eighteen_tools_with_strict_schemas(self) -> None:
        toolkit = FakeToolkit()
        tools = build_extended_langchain_tools(toolkit)  # type: ignore[arg-type]
        self.assertEqual(len(tools), 18)
        self.assertEqual(len({tool.name for tool in tools}), 18)
        self.assertNotIn("compare_papers", {tool.name for tool in tools})
        calculate = next(tool for tool in tools if tool.name == "calculate")
        result = await calculate.ainvoke({"expression": "2 + 2"})
        self.assertEqual(result["tool_name"], "calculate")
        self.assertEqual(toolkit.calls, [("calculate", {"expression": "2 + 2"})])

    async def test_removed_compare_papers_is_rejected_as_unknown(self) -> None:
        toolkit = ExtendedResearchToolkit(
            local=object(),
            content=object(),
            analysis=object(),
            scholarly=object(),
            workspace=object(),
            rag=object(),
        )  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            PermissionError,
            "unknown extended research tool: compare_papers",
        ):
            await toolkit.execute(
                "compare_papers",
                {"corpus_ids": ["C001", "T001"], "dimensions": ["accuracy"]},
            )


if __name__ == "__main__":
    unittest.main()
