"""Read-only long-term-memory adapter for main-Agent context hydration."""

from __future__ import annotations

from typing import Any, Protocol

from paper_research_agent.agent.tooling.contracts import ToolExecutionResult


class LongTermMemoryToolExecutor(Protocol):
    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> object: ...


class ToolkitLongTermMemoryProvider:
    """Expose only bounded search from the approval-capable extended toolkit."""

    def __init__(
        self,
        executor: LongTermMemoryToolExecutor,
        *,
        scope_id: str = "global",
    ) -> None:
        self._executor = executor
        self._scope_id = scope_id

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> tuple[dict[str, object], ...]:
        result = ToolExecutionResult.model_validate(
            await self._executor.execute_tool(
                "manage_long_term_memory",
                {
                    "action": "search",
                    "scope_id": self._scope_id,
                    "query": query[:500],
                    "limit": limit,
                },
            )
        )
        if result.status not in {"ok", "not_found"}:
            raise RuntimeError("long-term memory search returned an invalid status")
        return tuple(dict(item) for item in result.items)
