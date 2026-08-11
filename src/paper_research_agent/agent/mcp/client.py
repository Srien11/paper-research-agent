"""Lifecycle-safe stdio client gateway built on the official MCP SDK."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, Protocol, TextIO, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from paper_research_agent.agent.mcp.config import McpStdioServerConfig

McpServerState = Literal["stopped", "starting", "ready", "degraded", "closed"]


class McpSession(Protocol):
    def initialize(self) -> Awaitable[Any]: ...

    def list_tools(self) -> Awaitable[Any]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Awaitable[Any]: ...


McpSessionFactory = Callable[
    [McpStdioServerConfig], AbstractAsyncContextManager[McpSession]
]


@dataclass(frozen=True)
class McpServerStatus:
    server_id: str
    state: McpServerState
    reason_code: str | None
    tool_count: int


@dataclass
class _ServerRuntime:
    config: McpStdioServerConfig
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: McpServerState = "stopped"
    reason_code: str | None = None
    tools: tuple[Any, ...] = ()
    tool_names: frozenset[str] = frozenset()
    session: McpSession | None = None
    stack: AsyncExitStack | None = None


_BASE_ENVIRONMENT = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")


def _subprocess_environment(config: McpStdioServerConfig) -> dict[str, str]:
    names = (*_BASE_ENVIRONMENT, *config.inherit_env)
    return {name: os.environ[name] for name in names if name in os.environ}


def _open_error_log() -> TextIO:
    return open(os.devnull, "w", encoding="utf-8")


@asynccontextmanager
async def stdio_session_factory(config: McpStdioServerConfig) -> AsyncIterator[McpSession]:
    stack = AsyncExitStack()
    try:
        error_log = stack.enter_context(_open_error_log())
        parameters = StdioServerParameters(
            command=config.command,
            args=list(config.args),
            env=_subprocess_environment(config),
        )
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(parameters, errlog=error_log)
        )
        session = await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=30),
            )
        )
        yield cast(McpSession, session)
    finally:
        await stack.aclose()


class McpClientManager:
    def __init__(
        self,
        servers: tuple[McpStdioServerConfig, ...],
        *,
        session_factory: McpSessionFactory = stdio_session_factory,
    ) -> None:
        self._servers = {server.server_id: _ServerRuntime(server) for server in servers}
        if len(self._servers) != len(servers):
            raise ValueError("MCP server IDs must be unique")
        self._session_factory = session_factory
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closed = False

    async def start(self) -> None:
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("MCP client manager is closed")
            if self._started:
                return
            self._started = True
            for runtime in self._servers.values():
                if runtime.config.enabled:
                    await self._start_server(runtime)

    async def _start_server(self, runtime: _ServerRuntime) -> None:
        async with runtime.lock:
            runtime.state = "starting"
            stack = AsyncExitStack()
            runtime.stack = stack
            try:
                async with asyncio.timeout(runtime.config.startup_timeout_seconds):
                    session = await stack.enter_async_context(
                        self._session_factory(runtime.config)
                    )
                    await session.initialize()
                    listed = await session.list_tools()
                tools = tuple(getattr(listed, "tools", listed))
                names = frozenset(
                    name
                    for tool in tools
                    if isinstance((name := getattr(tool, "name", None)), str)
                )
                runtime.session = session
                runtime.tools = tools
                runtime.tool_names = names
                runtime.reason_code = None
                runtime.state = "ready"
            except TimeoutError:
                runtime.state = "degraded"
                runtime.reason_code = "mcp_startup_timeout"
                await stack.aclose()
            except Exception:  # noqa: BLE001 - sanitize all untrusted server failures
                runtime.state = "degraded"
                runtime.reason_code = "mcp_server_unavailable"
                await stack.aclose()

    def status(self, server_id: str) -> McpServerStatus:
        runtime = self._servers.get(server_id)
        if runtime is None:
            raise PermissionError(f"unknown MCP server: {server_id}")
        return McpServerStatus(
            server_id=server_id,
            state=runtime.state,
            reason_code=runtime.reason_code,
            tool_count=len(runtime.tools),
        )

    def tools_for(self, server_id: str) -> tuple[Any, ...]:
        runtime = self._servers.get(server_id)
        if runtime is None:
            raise PermissionError(f"unknown MCP server: {server_id}")
        if runtime.state != "ready":
            return ()
        return runtime.tools

    def degrade(self, server_id: str, reason_code: str) -> None:
        runtime = self._servers.get(server_id)
        if runtime is None:
            raise PermissionError(f"unknown MCP server: {server_id}")
        if runtime.state != "closed":
            runtime.state = "degraded"
            runtime.reason_code = reason_code

    async def call_tool(
        self,
        server_id: str,
        remote_name: str,
        arguments: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> Any:
        runtime = self._servers.get(server_id)
        if runtime is None:
            raise PermissionError(f"unknown MCP server: {server_id}")
        if runtime.state == "closed":
            raise RuntimeError(f"MCP server is closed: {server_id}")
        if runtime.state != "ready" or runtime.session is None:
            raise RuntimeError(f"MCP server is not ready: {server_id}")
        if remote_name not in runtime.tool_names:
            raise PermissionError(f"MCP tool was not discovered for server: {server_id}")
        try:
            async with asyncio.timeout(timeout_seconds):
                return await runtime.session.call_tool(remote_name, arguments)
        except TimeoutError:
            runtime.state = "degraded"
            runtime.reason_code = "mcp_call_timeout"
            raise TimeoutError(f"MCP tool call timed out: {server_id}") from None
        except Exception:  # noqa: BLE001 - sanitize all untrusted server failures
            runtime.state = "degraded"
            runtime.reason_code = "mcp_server_unavailable"
            raise RuntimeError(f"MCP tool call failed: {server_id}") from None

    async def aclose(self) -> None:
        async with self._start_lock:
            if self._closed:
                return
            self._closed = True
            for runtime in reversed(tuple(self._servers.values())):
                async with runtime.lock:
                    stack = runtime.stack
                    runtime.stack = None
                    runtime.session = None
                    runtime.state = "closed"
                    if stack is not None:
                        try:
                            await stack.aclose()
                        except Exception:  # noqa: BLE001 - continue releasing other servers
                            runtime.reason_code = "mcp_shutdown_failed"
