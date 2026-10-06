from __future__ import annotations

import unittest
from datetime import UTC, datetime

import httpx
from pydantic import ValidationError

from paper_research_agent.agent.tooling.contracts import IdentifierInput
from paper_research_agent.agent.tooling.provenance import (
    ScholarlyLookupRecord,
    ScholarlySourceRecord,
    bounded_lookups,
    source_record,
)
from paper_research_agent.agent.tooling.public_scholarly import CrossrefProvider
from paper_research_agent.agent.tooling.scholarly import ScholarlyResearchTools
from paper_research_agent.agent.tooling.scholarly_providers import ScholarlyProviderRegistry
from paper_research_agent.web.events import SafeRunEventDetail


class ScholarlyProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_and_cached_lookup_carry_only_actual_minimal_metadata(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "message": {
                        "DOI": "10.1234/test",
                        "title": ["Actual API title"],
                        "abstract": "abstract excluded from source cards",
                        "secret": "raw payload excluded",
                    }
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            tools = ScholarlyResearchTools(ScholarlyProviderRegistry((CrossrefProvider(client),)))
            first = await tools.resolve_paper_identifier(IdentifierInput(identifier="10.1234/test"))
            cached = await tools.resolve_paper_identifier(
                IdentifierInput(identifier="10.1234/test")
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(first.scholarly_lookup.provider, "crossref")
        self.assertTrue(first.scholarly_lookup.network_accessed)
        self.assertFalse(first.scholarly_lookup.cache_hit)
        self.assertTrue(cached.scholarly_lookup.cache_hit)
        self.assertFalse(cached.scholarly_lookup.network_accessed)
        self.assertEqual(first.scholarly_lookup.sources, cached.scholarly_lookup.sources)
        source = first.scholarly_lookup.sources[0]
        self.assertEqual(source.title, "Actual API title")
        self.assertEqual(source.url, "https://doi.org/10.1234/test")
        detail = SafeRunEventDetail(scholarly_lookups=(cached.scholarly_lookup,))
        for forbidden in ("abstract excluded", "raw payload", "secret"):
            self.assertNotIn(forbidden, detail.model_dump_json())
        self.assertNotIn("query", cached.scholarly_lookup.model_dump())
        self.assertEqual(SafeRunEventDetail.model_validate_json(detail.model_dump_json()), detail)

    def test_unknown_or_unresolved_identifiers_do_not_become_sources(self):
        self.assertIsNone(source_record({"title": "Invented title", "url": "https://evil.test"}))
        self.assertIsNone(source_record({"external_ids": {"DOI": "javascript:alert(1)"}}))

    def test_source_urls_reject_scripts_arbitrary_hosts_and_credentials(self):
        for url in (
            "javascript:alert(1)",
            "https://evil.test",
            "http://doi.org/10.1234/test",
            "https://doi.org.evil.test/",
            "https://secret@doi.org/",
            "https://doi.org:8443/",
        ):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                ScholarlySourceRecord(
                    source_id="X" + "a" * 16, title="Title", identifier="ID", url=url
                )

    def test_long_result_lists_fit_event_budget_and_preserve_call_status(self):
        sources = tuple(
            source_record({"title": "Title " * 400, "external_ids": {"DOI": f"10.1234/{index}"}})
            for index in range(50)
        )
        lookup = ScholarlyLookupRecord(
            provider="crossref",
            tool_name="search_scholarly_sources",
            queried_at=datetime.now(UTC),
            status="ok",
            cache_hit=False,
            network_accessed=True,
            returned_count=50,
            sources=sources,
        )
        records = bounded_lookups((lookup,) * 20)
        self.assertLess(
            len(SafeRunEventDetail(scholarly_lookups=records).model_dump_json()), 13_000
        )
        self.assertEqual(records[0].returned_count, 50)
        self.assertLess(len(records[0].sources), 50)
        self.assertEqual(records[0].status, "ok")

    def test_no_query_arguments_or_raw_fields_can_be_added(self):
        with self.assertRaises(ValidationError):
            ScholarlyLookupRecord(
                provider="arxiv",
                tool_name="resolve_paper_identifier",
                queried_at=datetime.now(UTC),
                status="ok",
                network_accessed=True,
                returned_count=0,
                query="private question",
            )
