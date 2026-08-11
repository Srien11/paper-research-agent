from __future__ import annotations

import asyncio
import sys
import unittest
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from paper_research_agent.agent.mcp.client import McpClientManager
from paper_research_agent.agent.mcp.config import McpStdioServerConfig


@dataclass(frozen=True)
class RemoteTool:
    name: str
    description: str = "remote description"
    inputSchema: dict[str, Any] | None = None


class FakeSession:
    def __init__(
        self,
        *,
        tools: tuple[RemoteTool, ...] = (RemoteTool("search_items"),),
        initialize_error: Exception | None = None,
        call_error: Exception | None = None,
        block_initialize: bool = False,
        block_call: bool = False,
    ) -> None:
        self.tools = tools
        self.initialize_error = initialize_error
        self.call_error = call_error
        self.block_initialize = block_initialize
        self.block_call = block_call
        self.initialize_calls = 0
        self.list_calls = 0
        self.call_calls = 0
        self.cancelled = False
        self.closed = False

    async def initialize(self) -> object:
        self.initialize_calls += 1
        if self.initialize_error:
            raise self.initialize_error
        if self.block_initialize:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return object()

    async def list_tools(self) -> tuple[RemoteTool, ...]:
        self.list_calls += 1
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.call_calls += 1
        if self.call_error:
            raise self.call_error
        if self.block_call:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return {"name": name, "arguments": arguments}


class FakeFactory:
    def __init__(self, sessions: dict[str, FakeSession]) -> None:
        self.sessions = sessions
        self.open_calls = 0

    @asynccontextmanager
    async def __call__(self, config: McpStdioServerConfig) -> AsyncIterator[FakeSession]:
        self.open_calls += 1
        session = self.sessions[config.server_id]
        try:
            yield session
        finally:
            session.closed = True


def _server(
    server_id: str = "zotero",
    *,
    startup_timeout_seconds: float = 1,
) -> McpStdioServerConfig:
    return McpStdioServerConfig.model_validate(
        {
            "server_id": server_id,
            "enabled": True,
            "command": sys.executable,
            "startup_timeout_seconds": startup_timeout_seconds,
            "tools": [
                {
                    "remote_name": "search_items",
                    "public_name": f"{server_id}__search_items",
                    "description": "local configured description",
                    "risk": "local_read",
                    "timeout_seconds": 1,
                    "max_result_items": 20,
                }
            ],
        }
    )


class McpClientManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_tools_once_and_closes_session(self) -> None:
        session = FakeSession()
        factory = FakeFactory({"zotero": session})
        manager = McpClientManager((_server(),), session_factory=factory)
        await asyncio.gather(manager.start(), manager.start())
        self.assertEqual(session.initialize_calls, 1)
        self.assertEqual(session.list_calls, 1)
        self.assertEqual(factory.open_calls, 1)
        self.assertEqual(manager.status("zotero").state, "ready")
        await manager.aclose()
        await manager.aclose()
        self.assertTrue(session.closed)
        self.assertEqual(manager.status("zotero").state, "closed")

    async def test_call_requires_ready_known_server_and_discovered_tool(self) -> None:
        manager = McpClientManager((_server(),), session_factory=FakeFactory({"zotero": FakeSession()}))
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            await manager.call_tool("zotero", "search_items", {}, timeout_seconds=1)
        await manager.start()
        with self.assertRaisesRegex(PermissionError, "not discovered"):
            await manager.call_tool("zotero", "delete_item", {}, timeout_seconds=1)
        with self.assertRaisesRegex(PermissionError, "unknown MCP server"):
            await manager.call_tool("other", "search_items", {}, timeout_seconds=1)
        await manager.aclose()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await manager.call_tool("zotero", "search_items", {}, timeout_seconds=1)

    async def test_startup_timeout_cancels_and_degrades_without_raising(self) -> None:
        session = FakeSession(block_initialize=True)
        manager = McpClientManager(
            (_server(startup_timeout_seconds=0.01),),
            session_factory=FakeFactory({"zotero": session}),
        )
        await manager.start()
        self.assertTrue(session.cancelled)
        self.assertEqual(manager.status("zotero").state, "degraded")
        self.assertEqual(manager.status("zotero").reason_code, "mcp_startup_timeout")
        await manager.aclose()

    async def test_call_timeout_cancels_and_does_not_reconnect(self) -> None:
        session = FakeSession(block_call=True)
        factory = FakeFactory({"zotero": session})
        manager = McpClientManager((_server(),), session_factory=factory)
        await manager.start()
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            await manager.call_tool("zotero", "search_items", {}, timeout_seconds=0.01)
        self.assertTrue(session.cancelled)
        self.assertEqual(manager.status("zotero").state, "degraded")
        await manager.start()
        self.assertEqual(factory.open_calls, 1)
        await manager.aclose()

    async def test_one_server_failure_does_not_close_another(self) -> None:
        sessions = {
            "zotero": FakeSession(initialize_error=RuntimeError("stderr secret")),
            "github": FakeSession(),
        }
        manager = McpClientManager(
            (_server("zotero"), _server("github")),
            session_factory=FakeFactory(sessions),
        )
        await manager.start()
        self.assertEqual(manager.status("zotero").state, "degraded")
        self.assertEqual(manager.status("github").state, "ready")
        result = await manager.call_tool(
            "github", "search_items", {"token": "must-not-leak"}, timeout_seconds=1
        )
        self.assertEqual(result["name"], "search_items")
        await manager.aclose()

    async def test_exception_text_never_contains_arguments_or_server_error(self) -> None:
        session = FakeSession(call_error=RuntimeError("stderr api_key=secret"))
        manager = McpClientManager((_server(),), session_factory=FakeFactory({"zotero": session}))
        await manager.start()
        with self.assertRaises(RuntimeError) as raised:
            await manager.call_tool(
                "zotero",
                "search_items",
                {"query": "private query", "api_key": "secret"},
                timeout_seconds=1,
            )
        message = str(raised.exception)
        self.assertNotIn("private query", message)
        self.assertNotIn("secret", message)
        self.assertNotIn("stderr", message)
        await manager.aclose()


if __name__ == "__main__":
    unittest.main()
