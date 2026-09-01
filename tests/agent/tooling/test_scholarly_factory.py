from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from paper_research_agent.agent.tooling.factory import create_extended_research_toolkit
from paper_research_agent.agent.tooling.scholarly_providers import (
    SCHOLARLY_OPERATIONS,
    ScholarlyProviderUnavailable,
)


class _UnavailableLiveProvider:
    provider_id = "unavailable_live"
    capabilities = SCHOLARLY_OPERATIONS

    async def execute(self, operation, request):
        del operation, request
        raise ScholarlyProviderUnavailable("provider unavailable")


class ScholarlyFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_mode_is_offline_and_creates_no_http_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = create_extended_research_toolkit(
                project_root=Path(directory),
                rag=Mock(),
                chunks=(),
                storage_classes={},
                environ={},
            )
            try:
                result = await handle.toolkit.execute(
                    "search_scholarly_sources",
                    {"query": "RAG", "limit": 5},
                )
            finally:
                await handle.aclose()

        self.assertIsNone(handle.client)
        self.assertEqual(handle.toolkit.scholarly.provider_ids, ("offline",))
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "provider_not_configured")
        self.assertFalse(result.summary["network_accessed"])
        self.assertFalse(handle.scholarly_readiness.ready)
        self.assertEqual(handle.scholarly_readiness.reason_code, "provider_offline")

    async def test_live_mode_builds_adapter_without_calling_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = create_extended_research_toolkit(
                project_root=Path(directory),
                rag=Mock(),
                chunks=(),
                storage_classes={},
                environ={
                    "PRA_SCHOLARLY_MODE": "live",
                    "SEMANTIC_SCHOLAR_API_KEY": "test-key",
                },
            )
            try:
                self.assertIsNotNone(handle.client)
                self.assertEqual(
                    handle.toolkit.scholarly.provider_ids,
                    ("semantic_scholar_crossref",),
                )
                self.assertTrue(handle.scholarly_readiness.ready)
                self.assertIsNone(handle.scholarly_readiness.reason_code)
            finally:
                await handle.aclose()

    async def test_live_adapter_can_be_ready_while_a_request_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = create_extended_research_toolkit(
                project_root=Path(directory),
                rag=Mock(),
                chunks=(),
                storage_classes={},
                scholarly_provider=_UnavailableLiveProvider(),
            )
            try:
                result = await handle.toolkit.execute(
                    "search_scholarly_sources",
                    {"query": "RAG", "limit": 5},
                )
            finally:
                await handle.aclose()

        self.assertTrue(handle.scholarly_readiness.ready)
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.summary["reason_code"], "all_providers_unavailable")

    def test_rejects_unknown_mode_before_building_toolkit(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "PRA_SCHOLARLY_MODE"
        ):
            create_extended_research_toolkit(
                project_root=Path(directory),
                rag=Mock(),
                chunks=(),
                storage_classes={},
                environ={"PRA_SCHOLARLY_MODE": "auto"},
            )


if __name__ == "__main__":
    unittest.main()
