"""公开书目联网验收；仅输出状态、数量和耗时，不保存 Provider 响应。"""

from __future__ import annotations

import asyncio
import json
import time

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)
from paper_research_agent.agent.tooling.public_scholarly import ArxivProvider, CrossrefProvider
from paper_research_agent.agent.tooling.scholarly_providers import (
    ScholarlyOperation,
    ScholarlyProviderRegistry,
    ScholarlyProviderRequest,
)


async def main() -> None:
    async with httpx.AsyncClient(
        timeout=10,
        trust_env=False,
        follow_redirects=False,
        headers={"User-Agent": "paper-research-agent/0.1"},
    ) as client:
        registry = ScholarlyProviderRegistry((CrossrefProvider(client), ArxivProvider(client)))
        cases: tuple[tuple[str, ScholarlyOperation, ScholarlyProviderRequest], ...] = (
            (
                "crossref_search",
                "search",
                ScholarlySearchInput(query="Deep learning", source="crossref", limit=3),
            ),
            (
                "crossref_doi",
                "resolve",
                IdentifierInput(identifier="10.1038/nature14539", source="crossref"),
            ),
            (
                "crossref_status",
                "status",
                IdentifierInput(identifier="10.1038/nature14539", source="crossref"),
            ),
            (
                "crossref_references",
                "citations",
                CitationGraphInput(identifier="10.1038/nature14539", source="crossref", limit=3),
            ),
            ("arxiv_id", "resolve", IdentifierInput(identifier="1706.03762", source="arxiv")),
            (
                "arxiv_search",
                "search",
                ScholarlySearchInput(query="attention transformer", source="arxiv", limit=3),
            ),
            ("arxiv_cached", "resolve", IdentifierInput(identifier="1706.03762", source="arxiv")),
        )
        for case_id, operation, request in cases:
            started = time.perf_counter()
            result = await registry.execute(operation, request)
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "provider": result.provider_id,
                        "status": result.status,
                        "count": len(result.items),
                        "cache_hit": result.summary.get("cache_hit", False),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "reason_code": result.summary.get("reason_code"),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
