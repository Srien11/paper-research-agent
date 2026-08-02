from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.rag import answer_question
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)


class FakeGenerator:
    model_id = "qwen3.7-plus-2026-05-26"
    prompt_version = "rag-answer-json-v1"

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.sent_messages = request.context.messages
        return GenerationResult(
            content=(
                '{"status":"answered","claims":[{"text":"端到端结论。",'
                '"citation_ids":["E1"]}],"insufficient_reason":null}'
            ),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=100,
            output_tokens=20,
            latency_ms=5,
            attempts=1,
        )


class FakeRetriever:
    def __init__(self, run: BilingualRetrievalRun):
        self.run = run
        self.queries: list[str] = []

    async def search(self, query: str, *, top_k: int | None = None) -> BilingualRetrievalRun:
        self.queries.append(query)
        self.top_k = top_k
        return self.run


class AnswerPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_bilingual_hit_to_validated_answer_preserves_private_rights(self) -> None:
        text = "private evidence sentinel"
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunk = EvidenceChunk(
            chunk_id="chunk-1",
            asset_id="asset-1",
            corpus_id="C001",
            element_ids=("element-1",),
            page_start=2,
            page_end=2,
            token_start=0,
            token_end=3,
            text=text,
            text_sha256=digest,
            config_sha256="a" * 64,
        )
        hit = SearchHit(
            chunk_id=chunk.chunk_id,
            corpus_id=chunk.corpus_id,
            asset_id=chunk.asset_id,
            page_start=2,
            page_end=2,
            text_sha256=digest,
            final_score=1,
            final_rank=1,
        )
        run = BilingualRetrievalRun(
            pipeline_id="pipeline",
            original_query="中文问题",
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="research question",
                requested_model="qwen3.7-plus-2026-05-26",
                actual_model="qwen3.7-plus-2026-05-26",
                prompt_version="query-rewrite-v2",
                latency_ms=1,
            ),
            degraded=False,
            top_k=1,
            hits=(hit,),
            index_id="index",
            config_sha256="b" * 64,
            storage_classes={"C001": "internal_research_only"},
            rights_status="loaded",
        )
        retriever = FakeRetriever(run)
        generator = FakeGenerator()
        result = await answer_question(
            " 中文问题 ",
            retriever=retriever,
            chunks=[chunk],
            generator=generator,
            token_budget=2400,
        )
        self.assertEqual(retriever.queries, ["中文问题"])
        self.assertIn(text, generator.sent_messages[-1].content)
        self.assertEqual(result.answer_markdown, "端到端结论。[E1]")
        self.assertEqual(result.citations[0].storage_class, "internal_research_only")
        self.assertNotIn(text, result.model_dump_json())


if __name__ == "__main__":
    unittest.main()
