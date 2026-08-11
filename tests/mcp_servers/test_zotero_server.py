from __future__ import annotations

import unittest
from typing import Any

from paper_research_agent.mcp_servers.zotero import build_zotero_server


class FakeService:
    async def search_items(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [{"item_key": "ABCD1234", "title": query, "limit": limit}]

    async def get_item(self, *, item_key: str) -> dict[str, Any]:
        return {"item_key": item_key}

    async def list_collections(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return [{"collection_key": "COLL1234", "limit": limit}]

    async def get_annotations(
        self, *, item_key: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        return [{"item_key": item_key, "limit": limit}]

    async def get_attachment_metadata(self, *, item_key: str) -> dict[str, Any]:
        return {"item_key": item_key, "content_type": "application/pdf"}

    async def get_fulltext(self, *, item_key: str) -> dict[str, Any]:
        return {"item_key": item_key, "content": "bounded"}


class ZoteroServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_exposes_exactly_six_read_tools_with_strict_schemas(self) -> None:
        server = build_zotero_server(FakeService())  # type: ignore[arg-type]
        tools = await server.list_tools()
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "search_items",
                "get_item",
                "list_collections",
                "get_annotations",
                "get_attachment_metadata",
                "get_fulltext",
            },
        )
        self.assertFalse(any(name.startswith(("create", "update", "delete")) for name in {tool.name for tool in tools}))
        search = next(tool for tool in tools if tool.name == "search_items")
        self.assertEqual(search.inputSchema["properties"]["limit"]["maximum"], 20)

    async def test_calls_service_and_returns_structured_result(self) -> None:
        server = build_zotero_server(FakeService())  # type: ignore[arg-type]
        result = await server.call_tool("search_items", {"query": "agent", "limit": 5})
        self.assertIsInstance(result, tuple)
        _content, structured = result
        self.assertEqual(structured["items"][0]["item_key"], "ABCD1234")


if __name__ == "__main__":
    unittest.main()
