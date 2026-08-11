from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import unittest
from pathlib import Path

from paper_research_agent.agent.mcp.client import McpClientManager
from paper_research_agent.agent.mcp.config import McpStdioServerConfig
from paper_research_agent.agent.mcp.provider import McpToolProvider

FIXTURE = Path(__file__).resolve().parents[1] / "mcp_servers" / "fake_stdio_server.py"


def _server() -> McpStdioServerConfig:
    tools = []
    for name, limit, output_bytes in (
        ("search_items", 20, 4096),
        ("get_item", 1, 4096),
        ("oversized_result", 3, 1024),
        ("server_error", 1, 4096),
    ):
        tools.append(
            {
                "remote_name": name,
                "public_name": f"fake__{name}",
                "description": f"本地协议测试工具 {name}。",
                "risk": "local_read",
                "timeout_seconds": 3,
                "max_result_items": limit,
                "max_output_bytes": output_bytes,
            }
        )
    return McpStdioServerConfig.model_validate(
        {
            "server_id": "fake",
            "enabled": True,
            "command": sys.executable,
            "args": [str(FIXTURE)],
            "startup_timeout_seconds": 5,
            "tools": tools,
        }
    )


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(process)
        return exit_code.value == 259
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


async def _wait_for_exit(pid: int) -> bool:
    for _ in range(50):
        if not _pid_exists(pid):
            return True
        await asyncio.sleep(0.02)
    return False


class McpEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_initialize_discovery_calls_and_normalization(self) -> None:
        self.assertTrue(FIXTURE.is_file())
        manager = McpClientManager((_server(),))
        await manager.start()
        self.assertEqual(manager.status("fake").state, "ready")
        self.assertEqual(manager.status("fake").tool_count, 4)
        provider = McpToolProvider(_server(), manager)
        tools = {tool.remote_name: tool for tool in provider.discover()}
        self.assertEqual(set(tools), {"search_items", "get_item", "oversized_result", "server_error"})

        search = await provider.execute(
            tools["search_items"], {"query": "agent", "limit": 2}, run_id="1" * 32
        )
        self.assertEqual(search.status, "ok")
        self.assertEqual(search.items[0]["title"], "agent")
        self.assertEqual(search.trust, "research_context")

        oversized = await provider.execute(
            tools["oversized_result"], {}, run_id="2" * 32
        )
        self.assertTrue(oversized.summary["truncated"])
        self.assertLessEqual(len(oversized.model_dump_json().encode("utf-8")), 1024)

        server_error = await provider.execute(tools["server_error"], {}, run_id="3" * 32)
        self.assertEqual(server_error.status, "insufficient")
        self.assertNotIn("private", server_error.model_dump_json())
        await manager.aclose()
        self.assertEqual(manager.status("fake").state, "closed")

    async def test_real_stdio_crash_degrades_and_shutdown_reaps_process(self) -> None:
        manager = McpClientManager((_server(),))
        await manager.start()
        provider = McpToolProvider(_server(), manager)
        tools = {tool.remote_name: tool for tool in provider.discover()}
        item = await provider.execute(tools["get_item"], {"item_key": "ABCD1234"}, run_id="4" * 32)
        pid = int(item.items[0]["server_pid"])
        self.assertTrue(_pid_exists(pid))
        await manager.aclose()
        self.assertTrue(await _wait_for_exit(pid))

        crashing = McpClientManager((_server(),))
        await crashing.start()
        provider = McpToolProvider(_server(), crashing)
        search_tool = next(tool for tool in provider.discover() if tool.remote_name == "search_items")
        with self.assertRaises(RuntimeError):
            await provider.execute(
                search_tool,
                {"query": "__crash__", "limit": 1},
                run_id="5" * 32,
            )
        self.assertEqual(crashing.status("fake").state, "degraded")
        await crashing.aclose()


if __name__ == "__main__":
    unittest.main()
