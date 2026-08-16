from __future__ import annotations

import asyncio
import unittest
from typing import Any

from paper_research_agent.agent.orchestrator.memory import (
    ToolkitLongTermMemoryProvider,
)
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult


class _FakeExecutor:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> object:
        self.calls.append((tool_name, arguments, run_id))
        return self.result


class ToolkitLongTermMemoryProviderTests(unittest.TestCase):
    def test_searches_existing_tool_with_global_scope(self) -> None:
        executor = _FakeExecutor(
            ToolExecutionResult(
                tool_name="manage_long_term_memory",
                items=(
                    {
                        "memory_id": "a" * 32,
                        "kind": "preference",
                        "content": "用户偏好中文回答",
                        "relevance": 0.8,
                    },
                ),
            )
        )
        provider = ToolkitLongTermMemoryProvider(executor, scope_id="global")

        result = asyncio.run(provider.search("继续之前的研究", limit=5))

        self.assertEqual(result[0]["memory_id"], "a" * 32)
        self.assertEqual(
            executor.calls,
            [
                (
                    "manage_long_term_memory",
                    {
                        "action": "search",
                        "scope_id": "global",
                        "query": "继续之前的研究",
                        "limit": 5,
                    },
                    None,
                )
            ],
        )

    def test_truncates_query_to_tool_contract(self) -> None:
        executor = _FakeExecutor(
            ToolExecutionResult(tool_name="manage_long_term_memory", status="not_found")
        )
        provider = ToolkitLongTermMemoryProvider(executor)

        result = asyncio.run(provider.search("问" * 600, limit=3))

        self.assertEqual(result, ())
        self.assertEqual(len(executor.calls[0][1]["query"]), 500)
        self.assertEqual(executor.calls[0][1]["limit"], 3)

    def test_rejects_non_read_status(self) -> None:
        executor = _FakeExecutor(
            ToolExecutionResult(
                tool_name="manage_long_term_memory",
                status="approval_required",
            )
        )
        provider = ToolkitLongTermMemoryProvider(executor)

        with self.assertRaisesRegex(RuntimeError, "invalid status"):
            asyncio.run(provider.search("继续", limit=5))


if __name__ == "__main__":
    unittest.main()
