from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from paper_research_agent.agent.langchain_tools import build_langchain_tools
from paper_research_agent.agent.models import (
    EvidenceRecord,
    GetEvidenceInput,
    GetEvidenceResult,
    SearchCorpusHit,
    SearchCorpusInput,
    SearchCorpusResult,
)


class LangChainToolAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = AsyncMock()
        self.service.search_corpus.return_value = SearchCorpusResult(
            query="grounded RAG",
            index_id="idx-test",
            degraded=False,
            hits=(
                SearchCorpusHit(
                    chunk_id="chunk-1",
                    corpus_id="C001",
                    section_id="results",
                    page_start=2,
                    page_end=2,
                    text_sha256="a" * 64,
                    storage_class="internal_research_only",
                    final_rank=1,
                ),
            ),
        )
        self.service.get_evidence.return_value = GetEvidenceResult(
            records=(
                EvidenceRecord(
                    chunk_id="chunk-1",
                    corpus_id="C001",
                    section_id="results",
                    page_start=2,
                    page_end=2,
                    text="Grounded evidence.",
                    text_sha256="a" * 64,
                    storage_class="internal_research_only",
                ),
            ),
        )
        self.tools = build_langchain_tools(self.service)

    async def test_registers_fixed_names_and_invokes_typed_search(self) -> None:
        search, evidence = self.tools

        self.assertEqual([tool.name for tool in self.tools], ["search_corpus", "get_evidence"])
        self.assertIs(search.args_schema, SearchCorpusInput)
        self.assertIs(evidence.args_schema, GetEvidenceInput)
        result = await search.ainvoke({"query": "grounded RAG", "top_k": 2})

        self.service.search_corpus.assert_awaited_once_with(
            SearchCorpusInput(query="grounded RAG", top_k=2)
        )
        self.assertEqual(result["schema_version"], "research-search-tool-v1")
        self.assertNotIn("text", result["hits"][0])

    async def test_evidence_tool_returns_structured_records_and_validates_input(self) -> None:
        _, evidence = self.tools

        result = await evidence.ainvoke({"chunk_ids": ["chunk-1"]})

        self.service.get_evidence.assert_awaited_once_with(
            GetEvidenceInput(chunk_ids=("chunk-1",))
        )
        self.assertEqual(result["records"][0]["text"], "Grounded evidence.")
        with self.assertRaises(ValidationError):
            await evidence.ainvoke({"chunk_ids": ["chunk-1", "chunk-1"]})


if __name__ == "__main__":
    unittest.main()
