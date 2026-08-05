from __future__ import annotations

import unittest

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)
from paper_research_agent.agent.tooling.scholarly import ScholarlyResearchTools


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/paper/search"):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "A Paper",
                        "year": 2026,
                        "authors": [{"name": "Author"}],
                        "externalIds": {"DOI": "10.1/test"},
                    }
                ]
            },
        )
    if request.url.path.endswith("/references"):
        return httpx.Response(
            200, json={"data": [{"citedPaper": {"paperId": "ref-1", "title": "Reference"}}]}
        )
    if request.url.path.endswith("/citations"):
        return httpx.Response(
            200, json={"data": [{"citingPaper": {"paperId": "cite-1", "title": "Citation"}}]}
        )
    if "/works/" in request.url.path:
        return httpx.Response(
            200, json={"message": {"DOI": "10.1/test", "type": "journal-article", "update-to": []}}
        )
    return httpx.Response(
        200, json={"paperId": "paper-1", "title": "A Paper", "externalIds": {"DOI": "10.1/test"}}
    )


class ScholarlyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        self.tools = ScholarlyResearchTools(self.client)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_search_resolve_graph_and_status(self) -> None:
        search = await self.tools.search_scholarly_sources(
            ScholarlySearchInput(query="RAG", limit=5)
        )
        self.assertEqual(search.items[0]["paper_id"], "paper-1")
        resolved = await self.tools.resolve_paper_identifier(
            IdentifierInput(identifier="10.1/test")
        )
        self.assertEqual(resolved.items[0]["title"], "A Paper")
        graph = await self.tools.get_citation_graph(CitationGraphInput(identifier="paper-1"))
        self.assertEqual({item["direction"] for item in graph.items}, {"references", "citations"})
        status = await self.tools.check_paper_status(IdentifierInput(identifier="10.1/test"))
        self.assertFalse(status.items[0]["has_update"])


if __name__ == "__main__":
    unittest.main()
