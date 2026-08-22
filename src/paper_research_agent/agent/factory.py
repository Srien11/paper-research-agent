"""Production construction for the Qwen-driven, SQLite-checkpointed ReAct Agent."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import SecretStr

from paper_research_agent.agent.graph import build_research_graph
from paper_research_agent.agent.models import StorageClass
from paper_research_agent.agent.observability import (
    AgentEvent,
    AgentEventSink,
    AgentEventTap,
    SQLiteAgentEventLogger,
    emit_agent_event,
)
from paper_research_agent.agent.orchestrator.identifiers import (
    dynamic_checkpoint_thread_id,
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


async def _configure_mcp_toolkit(
    *,
    toolkit: Any,
    handle: Any,
    project_root: Path,
    environ: Mapping[str, str] | None = None,
    manager_factory: Any = None,
) -> None:
    """Optionally merge ready, allowlisted MCP tools into one immutable snapshot."""
    source = os.environ if environ is None else environ
    raw_enabled = source.get("PRA_MCP_ENABLED", "false").strip().casefold()
    if raw_enabled in {"", "false", "0", "no"}:
        return
    if raw_enabled not in {"true", "1", "yes"}:
        raise ValueError("PRA_MCP_ENABLED must be true or false")

    from paper_research_agent.agent.mcp.client import McpClientManager
    from paper_research_agent.agent.mcp.config import load_mcp_host_config
    from paper_research_agent.agent.mcp.provider import McpToolProvider
    from paper_research_agent.agent.tooling.registry import ToolRegistrySnapshot

    raw_path = source.get("PRA_MCP_CONFIG_PATH", "deploy/mcp-servers.json").strip()
    if not raw_path:
        raise ValueError("PRA_MCP_CONFIG_PATH cannot be blank when MCP is enabled")
    config_path = Path(raw_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    host_config = load_mcp_host_config(config_path)
    enabled_servers = tuple(server for server in host_config.servers if server.enabled)
    manager = (
        manager_factory(enabled_servers)
        if manager_factory is not None
        else McpClientManager(enabled_servers)
    )
    try:
        await manager.start()
        tools = dict(toolkit.registry.tools)
        providers = dict(toolkit.registry.providers)
        for server in enabled_servers:
            if manager.status(server.server_id).state != "ready":
                pass
            else:
                provider = McpToolProvider(server, manager)
                discovered = provider.discover()
                if manager.status(server.server_id).state == "ready":
                    providers[provider.provider_id] = provider
                    tools.update({tool.public_name: tool for tool in discovered})
            status = manager.status(server.server_id)
            emit_agent_event(
                getattr(toolkit, "event_sink", None),
                AgentEvent(
                    run_id=uuid.uuid4().hex,
                    occurred_at=datetime.now(UTC),
                    event_type="mcp_server_status",
                    status="succeeded" if status.state == "ready" else "failed",
                    component="runtime",
                    name=server.server_id,
                    reason_code=status.reason_code,
                    degraded=status.state != "ready",
                    returned_count=status.tool_count,
                ),
            )
        toolkit.registry = ToolRegistrySnapshot(tools, providers)
        handle.mcp_manager = manager
    except BaseException:
        await manager.aclose()
        raise


def create_main_agent_model(
    *,
    model_id: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 180,
) -> ChatOpenAI:
    """Create the one structured-output model shared by main-Agent stages."""
    normalized_model = model_id.strip()
    if not normalized_model:
        raise ValueError("main agent model cannot be blank")
    normalized_key = api_key.strip()
    if not normalized_key:
        raise RuntimeError("main agent credentials are unavailable")
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        raise ValueError("main agent timeout must be between 0 and 3600 seconds")
    return ChatOpenAI(
        model=normalized_model,
        api_key=SecretStr(normalized_key),
        base_url=base_url.rstrip("/"),
        temperature=0,
        top_p=0.7,
        timeout=timeout_seconds,
        max_retries=2,
        extra_body={"enable_thinking": False},
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
    tapped_event_sink = AgentEventTap(resolved_event_sink)

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
                event_sink=tapped_event_sink,
            )
            await _configure_mcp_toolkit(
                toolkit=extended_handle.toolkit,
                handle=extended_handle,
                project_root=resolved_root,
            )
        dynamic_runtime = None
        if extended_handle is not None:
            from paper_research_agent.agent.dynamic.graph import build_dynamic_tool_graph
            from paper_research_agent.agent.dynamic.memory import LangChainMemoryProposer
            from paper_research_agent.agent.dynamic.router import LangChainToolRouter
            from paper_research_agent.agent.dynamic.runtime import DynamicResearchRuntime

            dynamic_graph = build_dynamic_tool_graph(
                router=LangChainToolRouter(model, extended_handle.toolkit.registry),
                toolkit=extended_handle.toolkit,
                max_steps=runtime_policy.max_dynamic_tool_steps,
                checkpointer=checkpointer,
                event_sink=tapped_event_sink,
                memory_proposer=LangChainMemoryProposer(model),
                registry=extended_handle.toolkit.registry,
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
            event_sink=tapped_event_sink,
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
            await checkpointer.adelete_thread(dynamic_checkpoint_thread_id(thread_id))

        return ResearchAgentRuntime(
            graph=graph,
            chunks=chunks,
            storage_classes=storage_classes,
            policy=runtime_policy,
            close=close,
            clear=clear,
            event_sink=tapped_event_sink,
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
    store: Any,
    hydrator: Any,
    interpreter: Any,
    goal_reconciler: Any,
    task_planner: Any,
    dispatcher: Any,
    synthesizer: Any = None,
    timeout_seconds: float = 180,
    checkpointer: Any = None,
    approval_resumer: Any = None,
    close: Any = None,
    clear: Any = None,
    event_sink: AgentEventSink | None = None,
) -> Any:
    """Assemble the cross-turn main Agent runtime from ready components."""
    from paper_research_agent.agent.orchestrator.factory import build_main_agent_runtime

    return build_main_agent_runtime(
        store=store,
        hydrator=hydrator,
        interpreter=interpreter,
        goal_reconciler=goal_reconciler,
        task_planner=task_planner,
        dispatcher=dispatcher,
        synthesizer=synthesizer,
        timeout_seconds=timeout_seconds,
        checkpointer=checkpointer,
        approval_resumer=approval_resumer,
        close=close,
        clear=clear,
        event_sink=event_sink,
    )


def create_main_agent_runtime_from_model(
    *,
    store: Any,
    model: Any,
    dispatcher: Any,
    timeout_seconds: float = 180,
    checkpointer: Any = None,
    close: Any = None,
    clear: Any = None,
    event_sink: AgentEventSink | None = None,
    memory_provider: Any = None,
    run_event_publisher: Any = None,
    fast_path_enabled: bool = False,
) -> Any:
    """Assemble all model-backed main-Agent stages from one shared client."""
    from paper_research_agent.agent.orchestrator.factory import (
        build_main_agent_runtime_from_model,
    )

    return build_main_agent_runtime_from_model(
        store=store,
        model=model,
        dispatcher=dispatcher,
        timeout_seconds=timeout_seconds,
        checkpointer=checkpointer,
        close=close,
        clear=clear,
        event_sink=event_sink,
        memory_provider=memory_provider,
        run_event_publisher=run_event_publisher,
        fast_path_enabled=fast_path_enabled,
    )
