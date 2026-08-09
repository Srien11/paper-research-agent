"""Production construction for the Qwen-driven, SQLite-checkpointed ReAct Agent."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

from paper_research_agent.agent.graph import build_research_graph
from paper_research_agent.agent.models import StorageClass
from paper_research_agent.agent.observability import (
    AgentEventSink,
    SQLiteAgentEventLogger,
)
from paper_research_agent.agent.planner import (
    ComparisonQueryResolver,
    LangChainComparisonTargetResolver,
    LangChainResearchPlanner,
)
from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.agent.reasoner import LangChainEvidenceReasoner
from paper_research_agent.agent.runtime import ResearchAgentRuntime
from paper_research_agent.agent.service import AsyncResearchRetriever, ResearchToolService
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.ingestion.models import DocumentElement, SectionRecord
from paper_research_agent.models import FrozenPaper
from paper_research_agent.retrieval.papers import AsyncPaperCandidateRetriever
from paper_research_agent.retrieval.query_rewrite import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_BASE_URL_ENV,
)


async def create_research_agent_runtime(
    *,
    retriever: AsyncResearchRetriever,
    paper_candidate_retriever: AsyncPaperCandidateRetriever,
    paper_candidate_query_resolver: ComparisonQueryResolver,
    chunks: Sequence[EvidenceChunk],
    storage_classes: Mapping[str, StorageClass],
    model_id: str,
    checkpoint_path: Path,
    policy: ResearchRuntimePolicy | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url_env: str = DEFAULT_BASE_URL_ENV,
    event_log_path: Path | None = None,
    event_sink: AgentEventSink | None = None,
    project_root: Path | None = None,
    papers: Sequence[FrozenPaper] = (),
    sections: Sequence[SectionRecord] = (),
    elements: Sequence[DocumentElement] = (),
    extended_tools_enabled: bool = True,
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
    resolved_event_sink = event_sink or SQLiteAgentEventLogger(
        event_log_path or checkpoint_path.with_name("agent-events-v1.sqlite3")
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(checkpoint_path)
    model: ChatOpenAI | None = None
    extended_handle: Any | None = None
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
        corpus_catalog = {paper.corpus_id: paper.title for paper in papers}
        target_resolver = LangChainComparisonTargetResolver(
            candidate_retriever=paper_candidate_retriever,
            query_resolver=paper_candidate_query_resolver,
            corpus_catalog=corpus_catalog,
        )
        planner = LangChainResearchPlanner(
            model,
            corpus_catalog=corpus_catalog,
            target_resolver=target_resolver,
        )
        reasoner = LangChainEvidenceReasoner(model)
        service = ResearchToolService(
            retriever=retriever,
            chunks=chunks,
            storage_classes=storage_classes,
        )
        if extended_tools_enabled:
            from paper_research_agent.agent.tooling.factory import (
                create_extended_research_toolkit,
            )

            resolved_root = project_root or checkpoint_path.parent.parent.parent
            extended_handle = create_extended_research_toolkit(
                project_root=resolved_root,
                rag=service,
                chunks=chunks,
                storage_classes=storage_classes,
                papers=papers,
                sections=sections,
                elements=elements,
                event_sink=resolved_event_sink,
            )
        dynamic_runtime = None
        if extended_handle is not None:
            from paper_research_agent.agent.dynamic.graph import build_dynamic_tool_graph
            from paper_research_agent.agent.dynamic.memory import LangChainMemoryProposer
            from paper_research_agent.agent.dynamic.router import LangChainToolRouter
            from paper_research_agent.agent.dynamic.runtime import DynamicResearchRuntime

            dynamic_graph = build_dynamic_tool_graph(
                router=LangChainToolRouter(model),
                toolkit=extended_handle.toolkit,
                max_steps=runtime_policy.max_dynamic_tool_steps,
                checkpointer=checkpointer,
                event_sink=resolved_event_sink,
                memory_proposer=LangChainMemoryProposer(model),
            )
            dynamic_runtime = DynamicResearchRuntime(
                graph=dynamic_graph,
                max_steps=runtime_policy.max_dynamic_tool_steps,
                timeout_seconds=runtime_policy.timeout_seconds,
            )
        graph = build_research_graph(
            planner=planner,
            reasoner=reasoner,
            service=service,
            policy=runtime_policy,
            checkpointer=checkpointer,
            event_sink=resolved_event_sink,
            extended_toolkit=(extended_handle.toolkit if extended_handle else None),
        )

        async def close() -> None:
            try:
                client = getattr(model, "root_async_client", None)
                close_client = getattr(client, "close", None)
                if close_client is not None:
                    await close_client()
            finally:
                try:
                    if extended_handle is not None:
                        await extended_handle.aclose()
                finally:
                    await connection.close()

        async def clear(thread_id: str) -> None:
            await checkpointer.adelete_thread(thread_id)
            await checkpointer.adelete_thread(f"dynamic::{thread_id}")

        return ResearchAgentRuntime(
            graph=graph,
            chunks=chunks,
            storage_classes=storage_classes,
            policy=runtime_policy,
            close=close,
            clear=clear,
            event_sink=resolved_event_sink,
            extended_tools=(extended_handle.toolkit if extended_handle else None),
            dynamic_tools=dynamic_runtime,
        )
    except BaseException:
        if extended_handle is not None:
            await extended_handle.aclose()
        if model is not None:
            client = getattr(model, "root_async_client", None)
            close_client = getattr(client, "close", None)
            if close_client is not None:
                await close_client()
        await connection.close()
        raise


async def create_main_agent_runtime(
    *,
    store,
    hydrator,
    interpreter,
    goal_reconciler,
    task_planner,
    dispatcher,
    timeout_seconds: float = 180,
    checkpointer=None,
    approval_resumer=None,
    close=None,
    clear=None,
):
    """Assemble the cross-turn main Agent runtime from ready components."""
    from paper_research_agent.agent.orchestrator.factory import build_main_agent_runtime

    return build_main_agent_runtime(
        store=store,
        hydrator=hydrator,
        interpreter=interpreter,
        goal_reconciler=goal_reconciler,
        task_planner=task_planner,
        dispatcher=dispatcher,
        timeout_seconds=timeout_seconds,
        checkpointer=checkpointer,
        approval_resumer=approval_resumer,
        close=close,
        clear=clear,
    )
