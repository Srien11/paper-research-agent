from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from paper_research_agent.agent.dynamic.memory import MemoryProposal
from paper_research_agent.agent.dynamic.models import ToolDecision
from paper_research_agent.agent.factory import _configure_mcp_toolkit, create_research_agent_runtime
from paper_research_agent.agent.mcp.client import McpServerStatus
from paper_research_agent.agent.models import EvidenceAssessment, ResearchPlan, ResearchStep
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.registry import RegisteredTool, builtin_registry_snapshot
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


class _BuiltinProvider:
    provider_id = "builtin"

    async def execute(
        self,
        tool: RegisteredTool,
        arguments: dict[str, Any],
        *,
        run_id: str,
    ) -> ToolExecutionResult:
        del arguments, run_id
        return ToolExecutionResult(tool_name=tool.public_name)


class _McpManager:
    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.started = False
        self.closed = False
        self.degraded: tuple[str, str] | None = None

    async def start(self) -> None:
        self.started = True

    def status(self, server_id: str) -> McpServerStatus:
        return McpServerStatus(
            server_id=server_id,
            state="ready" if self.ready else "degraded",
            reason_code=None if self.ready else "mcp_server_unavailable",
            tool_count=1 if self.ready else 0,
        )

    def tools_for(self, server_id: str) -> tuple[object, ...]:
        del server_id
        if not self.ready:
            return ()
        return (
            SimpleNamespace(
                name="search_items",
                description="untrusted remote description",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        )

    def degrade(self, server_id: str, reason_code: str) -> None:
        self.degraded = (server_id, reason_code)

    async def call_tool(self, *args: Any, **kwargs: Any) -> object:
        del args, kwargs
        raise AssertionError("assembly test must not call tools")

    async def aclose(self) -> None:
        self.closed = True


def _mcp_payload() -> dict[str, object]:
    return {
        "schema_version": "mcp-host-v1",
        "servers": [
            {
                "server_id": "zotero",
                "enabled": True,
                "command": sys.executable,
                "tools": [
                    {
                        "remote_name": "search_items",
                        "public_name": "zotero__search_items",
                        "description": "在本机 Zotero 文献库中搜索条目。",
                        "risk": "local_read",
                        "timeout_seconds": 5,
                        "max_result_items": 20,
                    }
                ],
            }
        ],
    }


class McpFactoryAssemblyTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_does_not_read_config_or_change_eighteen_tools(self) -> None:
        toolkit = SimpleNamespace(registry=builtin_registry_snapshot(_BuiltinProvider()))
        handle = SimpleNamespace(mcp_manager=None)
        await _configure_mcp_toolkit(
            toolkit=toolkit,
            handle=handle,
            project_root=Path("missing-project"),
            environ={"PRA_MCP_CONFIG_PATH": "missing.json"},
        )
        self.assertEqual(len(toolkit.registry.names), 18)
        self.assertIsNone(handle.mcp_manager)

    async def test_enabled_requires_valid_config(self) -> None:
        toolkit = SimpleNamespace(registry=builtin_registry_snapshot(_BuiltinProvider()))
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(FileNotFoundError),
        ):
            await _configure_mcp_toolkit(
                toolkit=toolkit,
                handle=SimpleNamespace(mcp_manager=None),
                project_root=Path(directory),
                environ={"PRA_MCP_ENABLED": "true", "PRA_MCP_CONFIG_PATH": "missing.json"},
            )

    async def test_ready_server_is_merged_and_degraded_server_is_omitted(self) -> None:
        for ready in (True, False):
            with self.subTest(ready=ready), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "mcp.json").write_text(
                    json.dumps(_mcp_payload(), ensure_ascii=False), encoding="utf-8"
                )
                toolkit = SimpleNamespace(registry=builtin_registry_snapshot(_BuiltinProvider()))
                handle = SimpleNamespace(mcp_manager=None)
                manager = _McpManager(ready=ready)
                await _configure_mcp_toolkit(
                    toolkit=toolkit,
                    handle=handle,
                    project_root=root,
                    environ={"PRA_MCP_ENABLED": "true", "PRA_MCP_CONFIG_PATH": "mcp.json"},
                    manager_factory=lambda _servers, current=manager: current,
                )
                self.assertTrue(manager.started)
                self.assertIs(handle.mcp_manager, manager)
                self.assertEqual(
                    "zotero__search_items" in toolkit.registry.names,
                    ready,
                )
                self.assertEqual(len(toolkit.registry.names), 19 if ready else 18)

    async def test_rejects_unknown_boolean_value(self) -> None:
        toolkit = SimpleNamespace(registry=builtin_registry_snapshot(_BuiltinProvider()))
        with self.assertRaisesRegex(ValueError, "PRA_MCP_ENABLED"):
            await _configure_mcp_toolkit(
                toolkit=toolkit,
                handle=SimpleNamespace(mcp_manager=None),
                project_root=Path("."),
                environ={"PRA_MCP_ENABLED": "sometimes"},
            )


if __name__ == "__main__":
    unittest.main()
