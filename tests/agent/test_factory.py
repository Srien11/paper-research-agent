from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from paper_research_agent.agent.dynamic.memory import MemoryProposal
from paper_research_agent.agent.dynamic.models import ToolDecision
from paper_research_agent.agent.factory import create_research_agent_runtime
from paper_research_agent.agent.models import EvidenceAssessment, ResearchPlan, ResearchStep
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
                    paper_candidate_retriever=AsyncMock(),
                    paper_candidate_query_resolver=AsyncMock(),
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
                    paper_candidate_retriever=AsyncMock(),
                    paper_candidate_query_resolver=AsyncMock(),
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
            self.assertEqual(model.with_structured_output.call_count, 4)
            self.assertEqual(
                [call.args[0] for call in model.with_structured_output.call_args_list],
                [
                    ResearchPlan,
                    EvidenceAssessment,
                    ToolDecision,
                    MemoryProposal,
                ],
            )
            self.assertEqual(runtime.policy.max_steps, 2)
            self.assertTrue(runtime.extended_tools_enabled)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_name("agent-events-v1.sqlite3").exists())

            calculation = await runtime.execute_tool("calculate", {"expression": "6 * 7"})
            self.assertEqual(calculation.items[0]["value"], 42)
            with self.assertRaisesRegex(PermissionError, "unknown extended"):
                await runtime.execute_tool("run_shell", {"command": "blocked"})

            await runtime.aclose()
            model.root_async_client.close.assert_awaited_once()

    async def test_persists_completed_graph_state_in_sqlite(self) -> None:
        chunk = _chunk()
        planner_structured = AsyncMock()
        planner_structured.ainvoke.return_value = ResearchPlan(
            steps=(
                ResearchStep(
                    step_id="search",
                    objective="Find local evidence",
                    query="bounded evidence",
                    top_k=2,
                ),
            )
        )
        reasoner_structured = AsyncMock()
        reasoner_structured.ainvoke.return_value = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
        )
        router_structured = AsyncMock()
        memory_structured = AsyncMock()
        model = Mock()
        model.with_structured_output.side_effect = (
            planner_structured,
            reasoner_structured,
            router_structured,
            memory_structured,
        )
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
                    paper_candidate_retriever=AsyncMock(),
                    paper_candidate_query_resolver=AsyncMock(),
                    chunks=(chunk,),
                    storage_classes={"C001": "internal_research_only"},
                    model_id="qwen-test-2026-01-01",
                    checkpoint_path=checkpoint,
                    api_key="test-key",
                    policy=ResearchRuntimePolicy(max_steps=1, max_tool_calls=2),
                )

            result = await runtime.run("验证持久化", thread_id="thread-sqlite")
            self.assertEqual(result.tool_call_count, 2)
            self.assertEqual(result.termination_reason, "evidence_sufficient")
            connection = sqlite3.connect(checkpoint)
            try:
                checkpoint_count = connection.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(checkpoint_count, 0)
            event_path = checkpoint.with_name("agent-events-v1.sqlite3")
            with closing(sqlite3.connect(event_path)) as event_connection:
                event_types = [
                    row[0]
                    for row in event_connection.execute(
                        "SELECT event_type FROM agent_events ORDER BY event_id"
                    ).fetchall()
                ]
            self.assertEqual(event_types[0], "run_started")
            self.assertEqual(event_types[-1], "run_completed")
            self.assertIn("tool_completed", event_types)

            await runtime.clear("thread-sqlite")
            await runtime.aclose()
            connection = sqlite3.connect(checkpoint)
            try:
                remaining = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
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
        filters=None,
        candidate_k: int | None = None,
        recall_k: int | None = None,
        rerank: bool = True,
    ) -> BilingualRetrievalRun:
        del privacy_ttl_days, filters, candidate_k, recall_k, rerank
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
