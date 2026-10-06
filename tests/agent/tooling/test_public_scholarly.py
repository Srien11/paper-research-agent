from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from paper_research_agent.agent.tooling.contracts import (
    CitationGraphInput,
    IdentifierInput,
    ScholarlySearchInput,
)
from paper_research_agent.agent.tooling.public_scholarly import ArxivProvider, CrossrefProvider
from paper_research_agent.agent.tooling.scholarly import (
    ScholarlyResearchTools,
    SemanticScholarCrossrefProvider,
)
from paper_research_agent.agent.tooling.scholarly_providers import (
    OfflineScholarlyProvider,
    ScholarlyProviderRegistry,
)

WORK = {
    "DOI": "10.1234/test",
    "title": ["Test title"],
    "abstract": "<jats:p>A &amp; B</jats:p>",
    "author": [{"given": "A", "family": "B"}],
    "published": {"date-parts": [[2024, 1, 1]]},
    "reference": [{"DOI": "10.1234/ref"}, {"unstructured": "Reference text"}],
    "update-to": [{"DOI": "10.1234/old", "type": "correction", "extra": "exclude"}],
}
FEED = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1706.03762v1</id><title>Attention Is All You Need</title>
<published>2017-06-12T00:00:00Z</published><summary>Public abstract</summary>
<author><name>Public Author</name></author><arxiv:doi>10.1234/test</arxiv:doi></entry></feed>"""


class PublicScholarlyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        def handler(request):
            self.requests.append(request)
            if request.url.host == "export.arxiv.org":
                return httpx.Response(200, content=FEED)
            return httpx.Response(
                200, json={"message": ({"items": [WORK]} if request.url.path == "/works" else WORK)}
            )

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.crossref = CrossrefProvider(self.client)
        self.arxiv = ArxivProvider(self.client)
        self.tools = ScholarlyResearchTools(ScholarlyProviderRegistry((self.crossref, self.arxiv)))

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_search_filters_normalized_metadata_and_trust(self):
        result = await self.tools.search_scholarly_sources(
            ScholarlySearchInput(
                query="test", source="crossref", limit=1, year_from=2020, year_to=2024
            )
        )
        self.assertEqual(result.trust, "research_context")
        self.assertEqual(result.items[0]["abstract"], "A & B")
        self.assertEqual(result.items[0]["authors"], ("A B",))
        self.assertEqual(result.items[0]["year"], 2024)
        self.assertEqual(self.requests[0].url.params["rows"], "1")
        self.assertEqual(
            self.requests[0].url.params["filter"],
            "from-pub-date:2020-01-01,until-pub-date:2024-12-31",
        )

    async def test_doi_resolve_cache_reused_by_status_and_references(self):
        request = IdentifierInput(identifier="https://doi.org/10.1234/test")
        first = await self.tools.resolve_paper_identifier(request)
        first.items[0]["external_ids"]["DOI"] = "mutated"
        second = await self.tools.resolve_paper_identifier(request)
        status = await self.tools.check_paper_status(request)
        graph = await self.tools.get_citation_graph(CitationGraphInput(identifier="10.1234/test"))
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(second.summary["cache_hit"])
        self.assertFalse(second.summary["network_accessed"])
        self.assertEqual(second.items[0]["external_ids"]["DOI"], "10.1234/test")
        self.assertFalse(status.summary["retraction_verified"])
        self.assertEqual(set(status.items[0]["updates"][0]), {"doi", "type"})
        self.assertTrue(graph.summary["partial"])
        self.assertEqual(graph.summary["unsupported_directions"], ("citations",))
        self.assertEqual(len(graph.items), 2)

    async def test_title_resolve_is_candidates(self):
        result = await self.tools.resolve_paper_identifier(IdentifierInput(identifier="Test title"))
        self.assertEqual(result.summary["match_type"], "title_candidates")

    async def test_inbound_citations_not_claimed_and_no_network(self):
        result = await self.tools.get_citation_graph(
            CitationGraphInput(identifier="10.1234/test", direction="citations")
        )
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "inbound_citations_not_supported")
        self.assertEqual(len(self.requests), 0)

    async def test_explicit_arxiv_source_skips_crossref(self):
        result = await self.tools.search_scholarly_sources(
            ScholarlySearchInput(
                query="attention transformer", source="arxiv", year_from=2017, year_to=2018
            )
        )
        self.assertEqual(result.summary["provider"], "arxiv")
        self.assertEqual(result.items[0]["external_ids"]["ArXiv"], "1706.03762v1")
        self.assertEqual(result.items[0]["url"], "https://arxiv.org/abs/1706.03762v1")
        self.assertEqual(len(self.requests), 1)
        self.assertIn(
            "submittedDate:[201701010000 TO 201812312359]",
            self.requests[0].url.params["search_query"],
        )

    async def test_auto_arxiv_identifier_does_not_search_crossref_as_title(self):
        result = await self.tools.resolve_paper_identifier(IdentifierInput(identifier="1706.03762"))
        self.assertEqual(result.summary["provider"], "arxiv")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.params["id_list"], "1706.03762")

    async def test_arxiv_cache_and_minimum_interval(self):
        await self.arxiv.execute("resolve", IdentifierInput(identifier="1706.03762"))
        with patch(
            "paper_research_agent.agent.tooling.public_scholarly.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            cached = await self.arxiv.execute("resolve", IdentifierInput(identifier="1706.03762"))
            sleep.assert_not_awaited()
            await self.arxiv.execute("resolve", IdentifierInput(identifier="1706.03763"))
            self.assertGreater(sleep.call_args.args[0], 2.9)
        self.assertTrue(cached.summary["cache_hit"])
        self.assertEqual(len(self.requests), 2)

    async def test_concurrent_identical_requests_share_one_fetch(self):
        results = await asyncio.gather(
            *(
                self.tools.resolve_paper_identifier(IdentifierInput(identifier="10.1234/test"))
                for _ in range(5)
            )
        )
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(sum(bool(r.summary["cache_hit"]) for r in results), 4)

    async def test_cache_expires_and_is_bounded(self):
        with patch(
            "paper_research_agent.agent.tooling.public_scholarly.time.monotonic", return_value=1000
        ):
            await self.crossref.execute("resolve", IdentifierInput(identifier="10.1234/0"))
        with patch(
            "paper_research_agent.agent.tooling.public_scholarly.time.monotonic",
            side_effect=range(1400, 3000, 4),
        ):
            for index in range(130):
                await self.crossref.execute(
                    "resolve", IdentifierInput(identifier=f"10.1234/{index}")
                )
        self.assertEqual(len(self.requests), 131)
        self.assertEqual(len(self.crossref._transport._cache), 128)

    async def test_explicit_unconfigured_source_and_unsupported_operation(self):
        result = await self.tools.search_scholarly_sources(
            ScholarlySearchInput(query="test", source="semantic_scholar")
        )
        self.assertEqual(result.status, "insufficient")
        result = await self.tools.check_paper_status(
            IdentifierInput(identifier="1706.03762", source="arxiv")
        )
        self.assertEqual(result.status, "insufficient")
        self.assertFalse(self.requests)

    async def test_offline_remains_offline_for_explicit_source(self):
        registry = ScholarlyProviderRegistry((OfflineScholarlyProvider(),))
        result = await registry.execute(
            "search", ScholarlySearchInput(query="test", source="arxiv")
        )
        self.assertEqual(result.summary["reason_code"], "provider_not_configured")

    async def test_http_errors_and_invalid_payloads_are_sanitized(self):
        cases = [
            (404, b"private response", "not_found", None),
            (403, b"private response", "insufficient", "provider_unavailable"),
            (503, b"private response", "insufficient", "provider_unavailable"),
            (302, b"private response", "insufficient", "provider_unavailable"),
            (200, b"broken JSON", "insufficient", "provider_invalid_response"),
            (200, b'{"message": []}', "insufficient", "provider_invalid_response"),
            (200, b"x" * 2_000_001, "insufficient", "provider_invalid_response"),
        ]
        for code, body, status, reason in cases:
            with self.subTest(code=code, reason=reason):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda _, code=code, body=body: httpx.Response(code, content=body)
                    )
                ) as client:
                    tools = ScholarlyResearchTools(
                        ScholarlyProviderRegistry((CrossrefProvider(client),))
                    )
                    result = await tools.resolve_paper_identifier(
                        IdentifierInput(identifier="10.1234/test")
                    )
                    self.assertEqual(result.status, status)
                    if reason:
                        self.assertEqual(
                            result.summary["provider_attempts"][0]["reason_code"], reason
                        )
                    self.assertNotIn("private response", result.model_dump_json())

    async def test_arxiv_invalid_xml_entities_and_error_entry(self):
        for body in (
            b"bad XML",
            b"<html/>",
            b'<!DOCTYPE feed [<!ENTITY x "secret">]><feed/>',
            '<!DOCTYPE feed [<!ENTITY x "secret">]><feed/>'.encode("utf-16"),
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>error</id></entry></feed>',
        ):
            with self.subTest(body_length=len(body)):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda _, body=body: httpx.Response(200, content=body)
                    )
                ) as client:
                    tools = ScholarlyResearchTools(
                        ScholarlyProviderRegistry((ArxivProvider(client),))
                    )
                    result = await tools.resolve_paper_identifier(
                        IdentifierInput(identifier="1706.03762")
                    )
                    self.assertEqual(result.status, "insufficient")

    async def test_rate_limit_cooldown_prevents_second_http_request(self):
        calls = []

        def limited(request):
            calls.append(request)
            return httpx.Response(429, headers={"Retry-After": "120"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
            tools = ScholarlyResearchTools(ScholarlyProviderRegistry((CrossrefProvider(client),)))
            for _ in range(2):
                result = await tools.resolve_paper_identifier(
                    IdentifierInput(identifier="10.1234/test")
                )
                self.assertEqual(result.status, "insufficient")
                self.assertEqual(
                    result.summary["provider_attempts"][0]["reason_code"], "provider_rate_limited"
                )
        self.assertEqual(len(calls), 1)

    async def test_timeout_falls_back_to_arxiv(self):
        def handler(request):
            if request.url.host == "api.crossref.org":
                raise httpx.ReadTimeout("private URL", request=request)
            return httpx.Response(200, content=FEED)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            tools = ScholarlyResearchTools(
                ScholarlyProviderRegistry((CrossrefProvider(client), ArxivProvider(client)))
            )
            result = await tools.search_scholarly_sources(ScholarlySearchInput(query="attention"))
        self.assertEqual(result.summary["provider"], "arxiv")
        self.assertEqual(result.summary["provider_attempts"][0]["reason_code"], "provider_timeout")
        self.assertNotIn("private URL", result.model_dump_json())

    async def test_optional_semantic_scholar_auth_failure_falls_back_without_leaking_key(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.host == "api.semanticscholar.org":
                return httpx.Response(403, text="private failure body")
            return httpx.Response(200, json={"message": {"items": [WORK]}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            tools = ScholarlyResearchTools(
                ScholarlyProviderRegistry(
                    (
                        SemanticScholarCrossrefProvider(client, api_key="test-key"),
                        CrossrefProvider(client),
                    )
                )
            )
            result = await tools.search_scholarly_sources(ScholarlySearchInput(query="test"))
        self.assertEqual(result.summary["provider"], "crossref")
        self.assertEqual(requests[0].headers["x-api-key"], "test-key")
        self.assertNotIn("x-api-key", requests[1].headers)
        self.assertNotIn("test-key", result.model_dump_json())
        self.assertNotIn("private failure body", result.model_dump_json())
