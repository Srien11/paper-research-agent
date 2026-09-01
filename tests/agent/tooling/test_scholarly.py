from __future__ import annotations

import unittest
from typing import Literal, cast

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)
from paper_research_agent.agent.tooling.scholarly import (
    ScholarlyResearchTools,
    SemanticScholarCrossrefProvider,
)
from paper_research_agent.agent.tooling.scholarly_providers import (
    SCHOLARLY_OPERATIONS,
    OfflineScholarlyProvider,
    ScholarlyOperation,
    ScholarlyProvider,
    ScholarlyProviderRateLimited,
    ScholarlyProviderRegistry,
    ScholarlyProviderRequest,
    ScholarlyProviderResult,
    ScholarlyProviderTimeout,
)


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


class LiveScholarlyAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        provider = SemanticScholarCrossrefProvider(self.client)
        self.tools = ScholarlyResearchTools(ScholarlyProviderRegistry((provider,)))

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_search_resolve_graph_and_status(self) -> None:
        search = await self.tools.search_scholarly_sources(
            ScholarlySearchInput(query="RAG", limit=5)
        )
        self.assertEqual(search.items[0]["paper_id"], "paper-1")
        self.assertEqual(search.summary["provider"], "semantic_scholar_crossref")
        resolved = await self.tools.resolve_paper_identifier(
            IdentifierInput(identifier="10.1/test")
        )
        self.assertEqual(resolved.items[0]["title"], "A Paper")
        graph = await self.tools.get_citation_graph(CitationGraphInput(identifier="paper-1"))
        self.assertEqual({item["direction"] for item in graph.items}, {"references", "citations"})
        status = await self.tools.check_paper_status(IdentifierInput(identifier="10.1/test"))
        self.assertFalse(status.items[0]["has_update"])

    async def test_rate_limit_is_normalized_without_real_network(self) -> None:
        async def limited(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "2"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
            tools = ScholarlyResearchTools(
                ScholarlyProviderRegistry((SemanticScholarCrossrefProvider(client),))
            )
            result = await tools.search_scholarly_sources(ScholarlySearchInput(query="RAG"))

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "all_providers_unavailable")
        self.assertEqual(
            result.summary["provider_attempts"][0]["reason_code"],
            "provider_rate_limited",
        )
        self.assertEqual(result.summary["provider_attempts"][0]["retry_after_seconds"], 2.0)


class OfflineScholarlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_public_tools_fail_closed_without_http_client(self) -> None:
        tools = ScholarlyResearchTools(
            ScholarlyProviderRegistry((OfflineScholarlyProvider(),))
        )
        results = (
            await tools.search_scholarly_sources(ScholarlySearchInput(query="RAG")),
            await tools.resolve_paper_identifier(IdentifierInput(identifier="10.1/test")),
            await tools.get_citation_graph(CitationGraphInput(identifier="paper-1")),
            await tools.check_paper_status(IdentifierInput(identifier="10.1/test")),
        )

        self.assertEqual(tools.provider_ids, ("offline",))
        self.assertTrue(all(result.status == "insufficient" for result in results))
        self.assertTrue(
            all(result.summary["reason_code"] == "provider_not_configured" for result in results)
        )
        self.assertTrue(all(result.summary["network_accessed"] is False for result in results))


class _FixtureProvider:
    def __init__(
        self,
        provider_id: str,
        outcomes: dict[ScholarlyOperation, ScholarlyProviderResult | Exception],
        *,
        capabilities: frozenset[ScholarlyOperation] = SCHOLARLY_OPERATIONS,
    ) -> None:
        self.provider_id = provider_id
        self.capabilities = capabilities
        self.outcomes = outcomes
        self.calls: list[ScholarlyOperation] = []

    async def execute(
        self,
        operation: ScholarlyOperation,
        request: ScholarlyProviderRequest,
    ) -> ScholarlyProviderResult:
        del request
        self.calls.append(operation)
        outcome = self.outcomes[operation]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _result(
    provider_id: str,
    status: Literal["ok", "not_found", "insufficient"],
) -> ScholarlyProviderResult:
    return ScholarlyProviderResult(
        provider_id=provider_id,
        status=status,
        items=({"title": "Fixture Paper"},) if status == "ok" else (),
    )


class ScholarlyProviderRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordered_fallback_handles_rate_limit_not_found_and_success(self) -> None:
        first = _FixtureProvider("first", {"search": ScholarlyProviderRateLimited()})
        second = _FixtureProvider("second", {"search": _result("second", "not_found")})
        third = _FixtureProvider("third", {"search": _result("third", "ok")})
        registry = ScholarlyProviderRegistry((first, second, third))

        result = await registry.execute("search", ScholarlySearchInput(query="RAG"))

        self.assertEqual(result.provider_id, "third")
        self.assertEqual(result.status, "ok")
        self.assertEqual(first.calls, ["search"])
        self.assertEqual(second.calls, ["search"])
        self.assertEqual(third.calls, ["search"])
        self.assertEqual(len(result.summary["provider_attempts"]), 3)

    async def test_timeout_returns_bounded_insufficient_result(self) -> None:
        provider = _FixtureProvider("timeout", {"resolve": ScholarlyProviderTimeout()})
        registry = ScholarlyProviderRegistry((provider,))

        result = await registry.execute("resolve", IdentifierInput(identifier="A Paper"))

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "all_providers_unavailable")
        self.assertEqual(
            result.summary["provider_attempts"][0]["reason_code"],
            "provider_timeout",
        )

    async def test_missing_capability_does_not_call_provider(self) -> None:
        provider = _FixtureProvider(
            "searchonly",
            {"search": _result("searchonly", "ok")},
            capabilities=frozenset({"search"}),
        )
        registry = ScholarlyProviderRegistry((provider,))

        result = await registry.execute("status", IdentifierInput(identifier="10.1/test"))

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "provider_capability_not_configured")
        self.assertEqual(provider.calls, [])

    def test_rejects_empty_duplicate_and_unknown_provider_definitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            ScholarlyProviderRegistry(())
        provider = _FixtureProvider("same", {"search": _result("same", "ok")})
        with self.assertRaisesRegex(ValueError, "unique"):
            ScholarlyProviderRegistry((provider, provider))
        invalid = _FixtureProvider(
            "invalid",
            {"search": _result("invalid", "ok")},
            capabilities=cast(frozenset[ScholarlyOperation], frozenset({"unknown"})),
        )
        with self.assertRaisesRegex(ValueError, "unknown capability"):
            ScholarlyProviderRegistry((cast(ScholarlyProvider, invalid),))


if __name__ == "__main__":
    unittest.main()
