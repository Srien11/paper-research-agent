"""Production construction for the Qwen-planned, SQLite-checkpointed Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import aiosqlite
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

from paper_research_agent.agent.graph import build_research_graph
from paper_research_agent.agent.models import StorageClass
from paper_research_agent.agent.planner import LangChainResearchPlanner
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.agent.runtime import ResearchAgentRuntime
from paper_research_agent.agent.service import AsyncResearchRetriever, ResearchToolService
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.query_rewrite import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BASE_URL_ENV,
)


async def create_research_agent_runtime(
    *,
    retriever: AsyncResearchRetriever,
    chunks: Sequence[EvidenceChunk],
    storage_classes: Mapping[str, StorageClass],
    model_id: str,
    checkpoint_path: Path,
    policy: ResearchRuntimePolicy | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url_env: str = DEFAULT_BASE_URL_ENV,
) -> ResearchAgentRuntime:
    """Open the durable checkpoint and return one closable production runtime."""
    normalized_model = model_id.strip()
    if not normalized_model:
        raise ValueError("research planner model cannot be blank")
    resolved_key = os.getenv(api_key_env) if api_key is None else api_key
    if not resolved_key or not resolved_key.strip():
        raise RuntimeError("research planner credentials are unavailable")
    resolved_base_url = (base_url or os.getenv(base_url_env) or DEFAULT_BASE_URL).rstrip("/")
    runtime_policy = policy or ResearchRuntimePolicy()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(checkpoint_path)
    model: ChatOpenAI | None = None
    try:
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        model = ChatOpenAI(
            model=normalized_model,
            api_key=SecretStr(resolved_key.strip()),
            base_url=resolved_base_url,
            temperature=0,
            top_p=0.7,
            timeout=runtime_policy.timeout_seconds,
            max_retries=2,
            extra_body={"enable_thinking": False},
        )
        planner = LangChainResearchPlanner(model)
        service = ResearchToolService(
            retriever=retriever,
            chunks=chunks,
            storage_classes=storage_classes,
        )
        graph = build_research_graph(
            planner=planner,
            service=service,
            policy=runtime_policy,
            checkpointer=checkpointer,
        )

        async def close() -> None:
            try:
                client = getattr(model, "root_async_client", None)
                close_client = getattr(client, "close", None)
                if close_client is not None:
                    await close_client()
            finally:
                await connection.close()

        async def clear(thread_id: str) -> None:
            await checkpointer.adelete_thread(thread_id)

        return ResearchAgentRuntime(
            graph=graph,
            chunks=chunks,
            storage_classes=storage_classes,
            policy=runtime_policy,
            close=close,
            clear=clear,
        )
    except BaseException:
        if model is not None:
            client = getattr(model, "root_async_client", None)
            close_client = getattr(client, "close", None)
            if close_client is not None:
                await close_client()
        await connection.close()
        raise
