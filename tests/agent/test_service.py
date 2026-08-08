from __future__ import annotations

import hashlib
import unittest
from collections.abc import Mapping

from paper_research_agent.agent.models import GetEvidenceInput, SearchCorpusInput
from paper_research_agent.agent.service import ResearchToolService
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import EvidenceJoinError
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk(chunk_id: str, corpus_id: str, text: str, page: int) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        asset_id=f"asset-{corpus_id}",
        corpus_id=corpus_id,
        section_id="results",
        element_ids=(f"element-{chunk_id}",),
        page_start=page,
        page_end=page,
        token_start=0,
        token_end=6,
        text=text,
        text_sha256=_digest(text),
        config_sha256="a" * 64,
    )


class FakeRetriever:
    def __init__(self, chunks: tuple[EvidenceChunk, ...]):
        self.chunks = chunks
        self.calls: list[
            tuple[
                str,
                int | None,
                int | None,
                Mapping[str, str] | None,
                int | None,
                int | None,
                bool,
            ]
        ] = []
        self.tamper_hash = False

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
        filters: Mapping[str, str] | None = None,
        candidate_k: int | None = None,
        recall_k: int | None = None,
        rerank: bool = True,
    ) -> BilingualRetrievalRun:
        self.calls.append(
            (query, top_k, privacy_ttl_days, filters, candidate_k, recall_k, rerank)
        )
        candidates = tuple(
            chunk
            for chunk in self.chunks
            if filters is None
            or all(getattr(chunk, key, None) == value for key, value in filters.items())
        )
        selected = candidates[: top_k or len(candidates)]
        hits = tuple(
            SearchHit(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                asset_id=chunk.asset_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text_sha256=("f" * 64 if self.tamper_hash else chunk.text_sha256),
                final_score=1.0 / rank,
                final_rank=rank,
            )
            for rank, chunk in enumerate(selected, start=1)
        )
        return BilingualRetrievalRun(
            pipeline_id="test-pipeline",
            original_query=query,
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="test query",
                requested_model="qwen-test",
                actual_model="qwen-test",
                prompt_version="rewrite-v1",
                latency_ms=1.0,
            ),
            degraded=False,
            top_k=top_k or len(hits),
            hits=hits,
            index_id="idx-test",
            config_sha256="b" * 64,
            storage_classes={
                chunk.corpus_id: (
                    "redistributable"
                    if chunk.corpus_id.startswith("T")
                    else "internal_research_only"
                )
                for chunk in selected
            },
            rights_status="loaded",
        )


class ResearchToolServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.chunks = (
            _chunk("chunk-1", "C001", "First private evidence.", 2),
            _chunk("chunk-2", "T001", "Second public evidence.", 5),
        )
        self.retriever = FakeRetriever(self.chunks)
        self.service = ResearchToolService(
            retriever=self.retriever,
            chunks=self.chunks,
            storage_classes={
                "C001": "internal_research_only",
                "T001": "redistributable",
            },
        )

    async def test_search_preserves_rank_lineage_and_loaded_rights(self) -> None:
        result = await self.service.search_corpus(
            SearchCorpusInput(query="  evidence grounding  ", top_k=2)
        )

        self.assertEqual(
            self.retriever.calls,
            [("evidence grounding", 2, None, None, None, None, True)],
        )
        self.assertEqual([hit.chunk_id for hit in result.hits], ["chunk-1", "chunk-2"])
        self.assertEqual(result.hits[0].storage_class, "internal_research_only")
        self.assertEqual(result.hits[1].storage_class, "redistributable")
        self.assertEqual(result.index_id, "idx-test")

    async def test_search_scopes_hybrid_retrieval_to_one_corpus(self) -> None:
        result = await self.service.search_corpus(
            SearchCorpusInput(query="method", top_k=2, corpus_id="T001")
        )

        self.assertEqual(
            self.retriever.calls,
            [("method", 2, None, {"corpus_id": "T001"}, 50, None, True)],
        )
        self.assertEqual(result.corpus_id, "T001")
        self.assertEqual([hit.corpus_id for hit in result.hits], ["T001"])

    async def test_get_evidence_preserves_request_order_and_reports_missing(self) -> None:
        result = await self.service.get_evidence(
            GetEvidenceInput(chunk_ids=("chunk-2", "missing", "chunk-1"))
        )

        self.assertEqual([record.chunk_id for record in result.records], ["chunk-2", "chunk-1"])
        self.assertEqual(result.missing_chunk_ids, ("missing",))
        self.assertEqual(result.records[0].text, "Second public evidence.")
        self.assertEqual(result.records[0].storage_class, "redistributable")

    async def test_search_fails_closed_on_retrieval_chunk_mismatch(self) -> None:
        self.retriever.tamper_hash = True

        with self.assertRaises(EvidenceJoinError):
            await self.service.search_corpus(SearchCorpusInput(query="tampered", top_k=1))

    def test_constructor_requires_rights_for_every_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "rights"):
            ResearchToolService(
                retriever=self.retriever,
                chunks=self.chunks,
                storage_classes={"C001": "internal_research_only"},
            )


if __name__ == "__main__":
    unittest.main()
