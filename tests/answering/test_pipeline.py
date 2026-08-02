from __future__ import annotations

import hashlib
import json
import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import ShortTermMemoryTurn
from paper_research_agent.rag import answer_question
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)


class FakeGenerator:
    model_id = "qwen3.7-plus-2026-05-26"
    prompt_version = "rag-answer-json-v1"

    def __init__(self, claim_text: str = "端到端结论。"):
        self.claim_text = claim_text

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.sent_messages = request.context.messages
        return GenerationResult(
            content=json.dumps(
                {
                    "status": "answered",
                    "claims": [{"text": self.claim_text, "citation_ids": ["E1"]}],
                    "insufficient_reason": None,
                },
                ensure_ascii=False,
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

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun:
        self.queries.append(query)
        self.top_k = top_k
        self.privacy_ttl_days = privacy_ttl_days
        return self.run


class FakeMemoryStore:
    def __init__(self, turns: tuple[ShortTermMemoryTurn, ...]):
        self.turns = turns
        self.appended: list[ShortTermMemoryTurn] = []

    def recent(
        self, session_id: str, *, now: datetime | None = None
    ) -> tuple[ShortTermMemoryTurn, ...]:
        del now
        self.loaded_session = session_id
        return self.turns

    def append(self, turn: ShortTermMemoryTurn, *, now: datetime | None = None) -> bool:
        del now
        self.appended.append(turn)
        return True


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
        self.assertIsNone(retriever.privacy_ttl_days)
        self.assertIn(text, generator.sent_messages[-1].content)
        self.assertEqual(result.answer_markdown, "端到端结论。[E1]")
        self.assertEqual(result.citations[0].storage_class, "internal_research_only")
        self.assertNotIn(text, result.model_dump_json())

    async def test_follow_up_uses_session_topic_but_current_question_for_answer(self) -> None:
        text = "current private evidence"
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunk = EvidenceChunk(
            chunk_id="chunk-2",
            asset_id="asset-2",
            corpus_id="C002",
            element_ids=("element-2",),
            page_start=3,
            page_end=3,
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
            page_start=3,
            page_end=3,
            text_sha256=digest,
            final_score=1,
            final_rank=1,
        )
        run = BilingualRetrievalRun(
            pipeline_id="pipeline",
            original_query="resolved query",
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="resolved research query",
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
            storage_classes={"C002": "internal_research_only"},
            rights_status="loaded",
        )
        now = datetime.now(UTC)
        previous = ShortTermMemoryTurn(
            turn_id="c" * 32,
            session_id="session-1",
            created_at=now,
            expires_at=now + timedelta(hours=24),
            user_question="BEIR 包含哪些检索任务？",
            standalone_question="BEIR 包含哪些检索任务？",
            status="answered",
            assistant_claims=("BEIR 包含多类检索任务。",),
        )
        memory = FakeMemoryStore((previous,))
        retriever = FakeRetriever(run)
        generator = FakeGenerator()

        result = await answer_question(
            "它和 MTEB 有什么区别？",
            retriever=retriever,
            chunks=[chunk],
            generator=generator,
            token_budget=2600,
            session_id="session-1",
            memory_store=memory,
            memory_config=ShortTermMemoryConfig(context_token_budget=500),
        )

        self.assertIn("BEIR 包含哪些检索任务", retriever.queries[0])
        self.assertIn("它和 MTEB 有什么区别", retriever.queries[0])
        rendered_messages = "\n".join(message.content for message in generator.sent_messages)
        self.assertIn('"user_question":"它和 MTEB 有什么区别？"', rendered_messages)
        self.assertIn("UNTRUSTED CONVERSATION MEMORY", rendered_messages)
        self.assertEqual(result.status, "answered")
        self.assertEqual(retriever.privacy_ttl_days, 1)
        self.assertEqual(len(memory.appended), 1)
        self.assertEqual(memory.appended[0].assistant_claims, ("端到端结论。",))
        self.assertEqual(memory.appended[0].standalone_question, retriever.queries[0])
        self.assertEqual(memory.appended[0].source_refs[0].chunk_id, "chunk-2")
        self.assertNotIn("[E1]", memory.appended[0].model_dump_json())

    async def test_oversized_claim_returns_answer_but_is_not_persisted_as_memory(self) -> None:
        text = "current evidence"
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunk = EvidenceChunk(
            chunk_id="chunk-3",
            asset_id="asset-3",
            corpus_id="C003",
            element_ids=("element-3",),
            page_start=1,
            page_end=1,
            token_start=0,
            token_end=2,
            text=text,
            text_sha256=digest,
            config_sha256="a" * 64,
        )
        run = BilingualRetrievalRun(
            pipeline_id="pipeline",
            original_query="问题",
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="question",
                requested_model="qwen3.7-plus-2026-05-26",
                actual_model="qwen3.7-plus-2026-05-26",
                prompt_version="query-rewrite-v2",
                latency_ms=1,
            ),
            degraded=False,
            top_k=1,
            hits=(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    corpus_id=chunk.corpus_id,
                    asset_id=chunk.asset_id,
                    page_start=1,
                    page_end=1,
                    text_sha256=digest,
                    final_score=1,
                    final_rank=1,
                ),
            ),
            index_id="index",
            config_sha256="b" * 64,
            storage_classes={"C003": "redistributable"},
            rights_status="loaded",
        )
        memory = FakeMemoryStore(())
        result = await answer_question(
            "问题",
            retriever=FakeRetriever(run),
            chunks=[chunk],
            generator=FakeGenerator("长" * 1001),
            token_budget=2600,
            session_id="session-1",
            memory_store=memory,
        )
        self.assertEqual(result.status, "answered")
        self.assertEqual(memory.appended, [])


if __name__ == "__main__":
    unittest.main()
