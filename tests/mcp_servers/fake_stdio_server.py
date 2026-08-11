"""Network-free real stdio fixture used only by MCP end-to-end tests."""

from __future__ import annotations

import os
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

server: FastMCP[None] = FastMCP("paper-research-mcp-fixture", log_level="ERROR")


@server.tool(description="Search deterministic fake items.")
async def search_items(
    query: Annotated[str, Field(min_length=1, max_length=500)],
    limit: Annotated[int, Field(ge=1, le=20)] = 10,
) -> dict[str, Any]:
    if query == "__crash__":
        os._exit(17)
    return {
        "items": [
            {"item_key": f"ITEM{index:04d}", "title": query}
            for index in range(limit)
        ]
    }


@server.tool(description="Get one deterministic fake item.")
async def get_item(
    item_key: Annotated[str, Field(pattern=r"^[A-Z0-9]{8}$")],
) -> dict[str, Any]:
    return {"items": [{"item_key": item_key, "server_pid": os.getpid()}]}


@server.tool(description="Return a deliberately oversized structured result.")
async def oversized_result() -> dict[str, Any]:
    return {
        "items": [
            {"index": index, "value": "x" * 50_000}
            for index in range(100)
        ]
    }


@server.tool(description="Raise a deterministic private server error.")
async def server_error() -> dict[str, Any]:
    raise RuntimeError("private fixture stack api_key=secret")


if __name__ == "__main__":
    server.run(transport="stdio")
