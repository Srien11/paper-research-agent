from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from paper_research_agent.agent.tooling.factory import create_extended_research_toolkit


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
            finally:
                await handle.aclose()

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
