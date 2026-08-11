"""Six-tool stdio-only MCP surface for the local Zotero library."""

from __future__ import annotations

from typing import Annotated, Any, Protocol

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from paper_research_agent.mcp_servers.zotero_service import ZoteroLocalService

ItemKey = Annotated[str, Field(pattern=r"^[A-Z0-9]{8}$")]
Limit = Annotated[int, Field(ge=1, le=20)]
Query = Annotated[str, Field(min_length=1, max_length=500)]


class ZoteroReadService(Protocol):
    async def search_items(self, *, query: str, limit: int = 10) -> list[dict[str, Any]]: ...

    async def get_item(self, *, item_key: str) -> dict[str, Any]: ...

    async def list_collections(self, *, limit: int = 20) -> list[dict[str, Any]]: ...

    async def get_annotations(
        self, *, item_key: str, limit: int = 20
    ) -> list[dict[str, Any]]: ...

    async def get_attachment_metadata(self, *, item_key: str) -> dict[str, Any]: ...

    async def get_fulltext(self, *, item_key: str) -> dict[str, Any]: ...


def build_zotero_server(service: ZoteroReadService) -> FastMCP[None]:
    server: FastMCP[None] = FastMCP("paper-research-zotero", log_level="ERROR")

    @server.tool(description="按题名、作者、关键词或本地索引内容搜索 Zotero 条目。")
    async def search_items(query: Query, limit: Limit = 10) -> dict[str, Any]:
        return {"items": await service.search_items(query=query, limit=limit)}

    @server.tool(description="读取单个 Zotero 条目的有限书目元数据。")
    async def get_item(item_key: ItemKey) -> dict[str, Any]:
        return {"items": [await service.get_item(item_key=item_key)]}

    @server.tool(description="列出有限数量的本地 Zotero 收藏夹。")
    async def list_collections(limit: Limit = 20) -> dict[str, Any]:
        return {"items": await service.list_collections(limit=limit)}

    @server.tool(description="读取指定条目下有限数量的 PDF 批注。")
    async def get_annotations(item_key: ItemKey, limit: Limit = 20) -> dict[str, Any]:
        return {"items": await service.get_annotations(item_key=item_key, limit=limit)}

    @server.tool(description="读取附件类型、父条目和页数等有限元数据。")
    async def get_attachment_metadata(item_key: ItemKey) -> dict[str, Any]:
        return {"items": [await service.get_attachment_metadata(item_key=item_key)]}

    @server.tool(description="读取最多 20000 字符的本地索引全文片段。")
    async def get_fulltext(item_key: ItemKey) -> dict[str, Any]:
        return {"items": [await service.get_fulltext(item_key=item_key)]}

    return server


def main() -> None:
    build_zotero_server(ZoteroLocalService()).run(transport="stdio")


if __name__ == "__main__":
    main()
