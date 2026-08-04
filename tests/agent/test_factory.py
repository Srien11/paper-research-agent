from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from paper_research_agent.agent.factory import create_research_agent_runtime
from paper_research_agent.agent.models import ResearchPlan, ResearchStep
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)


def _chunk() -> EvidenceChunk:
    text = "Bounded local evidence."
    return EvidenceChunk(
        chunk_id="chunk-1",
        asset_id="asset-1",
        corpus_id="C001",
        element_ids=("element-1",),
        page_start=1,
        page_end=1,
        token_start=0,
        token_end=3,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        config_sha256="a" * 64,
    )


class ResearchAgentFactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_provider_key_before_creating_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.sqlite3"
            with self.assertRaisesRegex(RuntimeError, "credentials"):
                await create_research_agent_runtime(
                    retriever=Mock(),
                    chunks=(_chunk(),),
                    storage_classes={"C001": "internal_research_only"},
                    model_id="qwen-test-2026-01-01",
                    checkpoint_path=checkpoint,
                    api_key="",
                )
            self.assertFalse(checkpoint.exists())

    async def test_builds_qwen_planner_and_closes_sqlite_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "runtime" / "state.sqlite3"
            model = Mock()
            model.with_structured_output.return_value = AsyncMock()
            model.root_async_client = Mock(close=AsyncMock())
            with patch(
                "paper_research_agent.agent.factory.ChatOpenAI",
                return_value=model,
            ) as chat_open_ai:
                runtime = await create_research_agent_runtime(
                    retriever=Mock(),
                    chunks=(_chunk(),),
                    storage_classes={"C001": "internal_research_only"},
                    model_id="qwen-test-2026-01-01",
                    checkpoint_path=checkpoint,
                    api_key="test-key",
                    base_url="https://dashscope.example/v1/",
                    policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
                )

            kwargs = chat_open_ai.call_args.kwargs
            self.assertEqual(kwargs["model"], "qwen-test-2026-01-01")
            self.assertEqual(kwargs["base_url"], "https://dashscope.example/v1")
            self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})
            self.assertEqual(runtime.policy.max_steps, 2)
            self.assertTrue(checkpoint.exists())

            await runtime.aclose()
            model.root_async_client.close.assert_awaited_once()

    async def test_persists_completed_graph_state_in_sqlite(self) -> None:
        chunk = _chunk()
        structured = AsyncMock()
        structured.ainvoke.return_value = ResearchPlan(
                steps=(
                    ResearchStep(
                        step_id="search",
                        objective="查找本地证据",
                        query="bounded evidence",
                        top_k=2,
                    ),
                )
            )
        model = Mock()
        model.with_structured_output.return_value = structured
        model.root_async_client = Mock(close=AsyncMock())
        retriever = FakeRetriever(chunk)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.sqlite3"
            with patch(
                "paper_research_agent.agent.factory.ChatOpenAI",
                return_value=model,
            ):
                runtime = await create_research_agent_runtime(
                    retriever=retriever,
                    chunks=(chunk,),
                    storage_classes={"C001": "internal_research_only"},
                    model_id="qwen-test-2026-01-01",
                    checkpoint_path=checkpoint,
                    api_key="test-key",
                    policy=ResearchRuntimePolicy(max_steps=1, max_tool_calls=2),
                )

            result = await runtime.run("验证持久化", thread_id="thread-sqlite")
            self.assertEqual(result.tool_call_count, 2)
            connection = sqlite3.connect(checkpoint)
            try:
                checkpoint_count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(checkpoint_count, 0)

            await runtime.clear("thread-sqlite")
            await runtime.aclose()
            connection = sqlite3.connect(checkpoint)
            try:
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(remaining, 0)


class FakeRetriever:
    def __init__(self, chunk: EvidenceChunk):
        self.chunk = chunk

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun:
        del privacy_ttl_days
        return BilingualRetrievalRun(
            pipeline_id="test-pipeline",
            original_query=query,
            rewrite=QueryRewriteTrace(
                status="success",
                english_query=query,
                requested_model="qwen-test",
                actual_model="qwen-test",
                prompt_version="query-rewrite-v2",
                latency_ms=1,
            ),
            degraded=False,
            top_k=top_k or 2,
            hits=(
                SearchHit(
                    chunk_id=self.chunk.chunk_id,
                    corpus_id=self.chunk.corpus_id,
                    asset_id=self.chunk.asset_id,
                    page_start=self.chunk.page_start,
                    page_end=self.chunk.page_end,
                    text_sha256=self.chunk.text_sha256,
                    final_score=1,
                    final_rank=1,
                ),
            ),
            index_id="idx-test",
            config_sha256="b" * 64,
            storage_classes={"C001": "internal_research_only"},
            rights_status="loaded",
        )


if __name__ == "__main__":
    unittest.main()
