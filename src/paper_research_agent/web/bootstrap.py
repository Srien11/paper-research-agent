"""Production ownership boundary for all Web application services."""

from __future__ import annotations

import asyncio
import os
import warnings
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from paper_research_agent.agent.factory import (
    create_main_agent_model,
    create_main_agent_runtime_from_model,
)
from paper_research_agent.agent.observability import AgentEvent, SQLiteAgentEventLogger
from paper_research_agent.agent.orchestrator.children import ChildGraphDispatcher
from paper_research_agent.agent.orchestrator.identifiers import dynamic_thread_id
from paper_research_agent.agent.orchestrator.memory import ToolkitLongTermMemoryProvider
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
from paper_research_agent.conversation.store import ConversationStore, SQLiteConversationStore
from paper_research_agent.web.chat_runtime import ConversationRuntime
from paper_research_agent.web.child_executors import (
    ConversationChildExecutor,
    RAGRuntimeChildExecutor,
)
from paper_research_agent.web.events import (
    AgentStreamEventDraft,
    AgentStreamEventType,
    RunNodeStatus,
    SafeRunEventDetail,
)
from paper_research_agent.web.files import AttachmentStore
from paper_research_agent.web.run_event_bus import RunEventBus, RunEventPublisher

MainAgentMode = Literal["legacy", "primary"]


class ClosableRuntime(Protocol):
    async def aclose(self) -> None: ...


class DynamicResearchRuntimeLike(Protocol):
    async def run_dynamic_tools(self, question: str, *, thread_id: str) -> object: ...

    async def resume_dynamic_tools(self, *, thread_id: str, approved: bool) -> object: ...

    def capture_agent_events(
        self, observer: Callable[[AgentEvent], None]
    ) -> AbstractContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class ApplicationEnvironment:
    """Validated non-secret paths and provider settings used during startup."""

    mode: MainAgentMode
    project_root: Path
    conversation_path: Path
    attachment_path: Path
    main_checkpoint_path: Path
    api_key: str
    base_url: str
    main_model: str
    corpus_configured: bool
    timeout_seconds: float = 180
    main_agent_fast_path_enabled: bool = True

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> ApplicationEnvironment:
        source = os.environ if environ is None else environ
        root = Path(
            source.get("PRA_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
        ).resolve()
        return cls(
            mode=main_agent_mode_from_environment(source),
            project_root=root,
            conversation_path=_environment_path(
                source, root, "PRA_CONVERSATION_PATH", "data/runtime/conversation-v1.sqlite3"
            ),
            attachment_path=_environment_path(
                source, root, "PRA_ATTACHMENT_PATH", "data/runtime/uploads"
            ),
            main_checkpoint_path=_environment_path(
                source,
                root,
                "PRA_MAIN_AGENT_CHECKPOINT_PATH",
                "data/runtime/main-agent-state-v1.sqlite3",
            ),
            api_key=source.get("DASHSCOPE_API_KEY", "").strip(),
            base_url=source.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/"),
            main_model=source.get(
                "PRA_MAIN_AGENT_MODEL",
                source.get("PRA_CHAT_MODEL", "qwen3.7-plus-2026-05-26"),
            ).strip(),
            corpus_configured=bool(source.get("PRA_CORPUS_DIR", "").strip()),
            timeout_seconds=_timeout_from_environment(source),
            main_agent_fast_path_enabled=_strict_boolean_from_environment(
                source,
                "PRA_MAIN_AGENT_FAST_PATH_ENABLED",
                default=True,
            ),
        )


@dataclass(slots=True)
class ApplicationServices:
    """Own every production runtime and close each resource exactly once."""

    conversation_store: ConversationStore
    rag_runtime: ClosableRuntime | None
    chat_runtime: ClosableRuntime
    attachment_store: AttachmentStore
    run_event_bus: RunEventBus
    main_agent_runtime: MainAgentRuntime | None
    mode: MainAgentMode
    _closers: tuple[Callable[[], Awaitable[None]], ...] = field(
        default=(), repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def main_agent_repository(self) -> ConversationStore:
        return self.conversation_store

    @property
    def runtime(self) -> ClosableRuntime:
        return self.rag_runtime or self.chat_runtime

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for closer in reversed(self._closers):
            try:
                await closer()
            except BaseException as error:  # noqa: BLE001 - continue owned cleanup
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


@dataclass(slots=True)
class _MainCheckpoint:
    checkpointer: AsyncSqliteSaver
    connection: aiosqlite.Connection
    _closed: bool = False

    async def clear_threads(self, thread_ids: tuple[str, ...]) -> None:
        for thread_id in thread_ids:
            await self.checkpointer.adelete_thread(thread_id)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.connection.close()


class _DynamicChildAdapter:
    def __init__(
        self,
        runtime: DynamicResearchRuntimeLike,
        *,
        run_event_publisher: RunEventPublisher | None = None,
    ) -> None:
        self._runtime = runtime
        self._run_event_publisher = run_event_publisher

    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        memory_context: tuple[dict[str, object], ...] = (),
        child_context: dict[str, object] | None = None,
    ) -> object:
        del memory_context, child_context
        return await self._runtime.run_dynamic_tools(question, thread_id=thread_id)

    async def resume(self, *, thread_id: str, approved: bool) -> object:
        return await self._runtime.resume_dynamic_tools(
            thread_id=thread_id, approved=approved
        )

    async def run_task(self, request: ChildTaskRequest) -> object:
        return await self._run_with_product_events(
            request,
            self._runtime.run_dynamic_tools(
                request.objective or request.current_message,
                thread_id=dynamic_thread_id(
                    request.conversation_id, request.run_id, request.task_id
                ),
            ),
        )

    async def resume_task(
        self, request: ChildTaskRequest, *, approved: bool
    ) -> object:
        return await self._run_with_product_events(
            request,
            self._runtime.resume_dynamic_tools(
                thread_id=dynamic_thread_id(
                    request.conversation_id, request.run_id, request.task_id
                ),
                approved=approved,
            ),
        )

    async def _run_with_product_events(
        self, request: ChildTaskRequest, operation: Awaitable[object]
    ) -> object:
        publisher = self._run_event_publisher
        if publisher is None:
            return await operation
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        forwarder = asyncio.create_task(
            _forward_dynamic_tool_events(queue, publisher, request)
        )
        try:
            with self._runtime.capture_agent_events(queue.put_nowait):
                return await operation
        finally:
            queue.put_nowait(None)
            await forwarder


async def _forward_dynamic_tool_events(
    queue: asyncio.Queue[AgentEvent | None],
    publisher: RunEventPublisher,
    request: ChildTaskRequest,
) -> None:
    sequence = 0
    active: dict[str, tuple[str, int]] = {}
    while True:
        event = await queue.get()
        if event is None:
            return
        if event.component != "tool" or event.event_type not in {
            "tool_started",
            "tool_completed",
            "tool_failed",
            "runtime_intercepted",
        }:
            continue
        node_id: str
        call_sequence: int
        event_type: AgentStreamEventType
        status: RunNodeStatus
        if event.event_type == "tool_started":
            sequence += 1
            call_sequence = sequence
            node_id = f"tool:{request.task_id}:{event.name}:{call_sequence}"
            active[event.name] = (node_id, call_sequence)
            event_type = "tool_started"
            status = "running"
            title = f"调用工具 {event.name}"
            summary = "工具正在执行"
        else:
            current = active.pop(event.name, None)
            if current is None:
                sequence += 1
                current = (
                    f"tool:{request.task_id}:{event.name}:{sequence}",
                    sequence,
                )
            node_id, call_sequence = current
            if event.event_type == "tool_completed":
                event_type = "tool_completed"
                status = "completed"
                title = f"工具 {event.name} 已完成"
                summary = (
                    f"返回 {event.returned_count} 项结果"
                    if event.returned_count is not None
                    else "工具执行完成"
                )
            elif (
                event.event_type == "runtime_intercepted"
                and event.reason_code == "approval_required"
            ):
                event_type = "tool_completed"
                status = "waiting_approval"
                title = f"工具 {event.name} 等待审批"
                summary = "敏感工具已暂停，等待批准或拒绝"
            else:
                event_type = "tool_failed"
                status = "failed"
                title = f"工具 {event.name} 未完成"
                summary = "工具执行失败或被策略拦截"
        await publisher.publish(
            AgentStreamEventDraft(
                type=event_type,
                occurred_at=datetime.now(UTC),
                request_id=request.request_id,
                run_id=request.run_id,
                turn_id=request.turn_id,
                node_id=node_id,
                parent_node_id=f"task:{request.task_id}",
                task_id=request.task_id,
                status=status,
                title=title,
                summary=summary,
                duration_ms=(
                    max(0, round(event.duration_ms))
                    if event.duration_ms is not None
                    else None
                ),
                detail=SafeRunEventDetail(
                    tool_name=event.name,
                    returned_count=event.returned_count,
                    reason_code=event.reason_code,
                    delivery_mode="event_only",
                ),
            ),
            idempotency_key=(
                f"task:{request.task_id}:attempt:{request.attempt_count}:"
                f"tool:{call_sequence}:{event_type}:"
                f"{int(event.occurred_at.timestamp() * 1_000_000)}"
            ),
        )


def main_agent_mode_from_environment(
    environ: Mapping[str, str] | None = None,
) -> MainAgentMode:
    source = os.environ if environ is None else environ
    explicit = source.get("PRA_MAIN_AGENT_MODE")
    if explicit is not None and explicit.strip():
        normalized = explicit.strip().casefold()
        if normalized not in {"legacy", "primary"}:
            raise ValueError("PRA_MAIN_AGENT_MODE must be legacy or primary")
        return cast(MainAgentMode, normalized)
    enabled = source.get("PRA_MAIN_AGENT_ENABLED")
    if enabled is None or not enabled.strip():
        return "legacy"
    normalized_enabled = enabled.strip().casefold()
    if normalized_enabled not in {"true", "false"}:
        raise ValueError("PRA_MAIN_AGENT_ENABLED must be true or false")
    warnings.warn(
        "PRA_MAIN_AGENT_ENABLED is deprecated; use PRA_MAIN_AGENT_MODE",
        DeprecationWarning,
        stacklevel=2,
    )
    return "primary" if normalized_enabled == "true" else "legacy"


async def create_application_services(
    environment: ApplicationEnvironment,
    *,
    conversation_store: ConversationStore | None = None,
    attachment_store: AttachmentStore | None = None,
) -> ApplicationServices:
    """Build one shared service graph and clean partial construction failures."""
    if environment.mode == "primary" and not environment.api_key.strip():
        raise RuntimeError("main agent credentials are unavailable")
    store = conversation_store or SQLiteConversationStore(environment.conversation_path)
    attachments = attachment_store or AttachmentStore(environment.attachment_path)
    run_events = RunEventBus(store)
    closers: list[Callable[[], Awaitable[None]]] = []
    owned_ids: set[int] = set()

    def own(resource: object, closer: Callable[[], Awaitable[None]] | None = None) -> None:
        if id(resource) in owned_ids:
            return
        resolved = closer or getattr(resource, "aclose", None)
        if resolved is None:
            return
        owned_ids.add(id(resource))
        closers.append(cast(Callable[[], Awaitable[None]], resolved))

    try:
        own(run_events)
        chat = _create_chat_runtime(environment, store)
        own(chat)
        rag = await _create_rag_runtime(environment)
        if rag is not None:
            own(rag)
        main: MainAgentRuntime | None = None
        if environment.mode == "primary":
            model = _create_main_model(environment)
            model_client = getattr(model, "root_async_client", None)
            model_close = getattr(model_client, "close", None)
            if model_client is not None and model_close is not None:
                own(model_client, model_close)
            checkpoint = await _open_main_checkpoint(environment.main_checkpoint_path)
            own(checkpoint)
            event_sink = SQLiteAgentEventLogger(
                environment.main_checkpoint_path.with_name("agent-events-v1.sqlite3")
            )
            conversation_executor = ConversationChildExecutor(
                runtime=cast(Any, chat),
                attachments=attachments,
                run_event_publisher=run_events.publisher,
            )
            research_agent = getattr(rag, "research_agent", None)
            memory_provider = (
                ToolkitLongTermMemoryProvider(research_agent)
                if research_agent is not None
                and bool(getattr(research_agent, "extended_tools_enabled", False))
                else None
            )

            async def clear_main_state(conversation_id: str) -> None:
                threads = await asyncio.to_thread(
                    store.agent_checkpoint_threads, conversation_id
                )
                await checkpoint.clear_threads(threads.main)
                if research_agent is not None:
                    for thread_id in threads.research:
                        await research_agent.clear(thread_id)

            dispatcher = ChildGraphDispatcher(
                direct_chat=conversation_executor,
                local_rag=(
                    RAGRuntimeChildExecutor(
                        cast(Any, rag), run_event_publisher=run_events.publisher
                    )
                    if rag
                    else None
                ),
                dynamic_tools=(
                    cast(
                        Any,
                        _DynamicChildAdapter(
                            research_agent,
                            run_event_publisher=run_events.publisher,
                        ),
                    )
                    if research_agent is not None
                    else None
                ),
                attachment_qa=conversation_executor,
                file_edit=conversation_executor,
            )
            main = create_main_agent_runtime_from_model(
                store=store,
                model=model,
                dispatcher=dispatcher,
                timeout_seconds=environment.timeout_seconds,
                checkpointer=checkpoint.checkpointer,
                clear=clear_main_state,
                event_sink=event_sink,
                memory_provider=memory_provider,
                run_event_publisher=run_events.publisher,
                fast_path_enabled=environment.main_agent_fast_path_enabled,
            )
            own(main)
        return ApplicationServices(
            conversation_store=store,
            rag_runtime=rag,
            chat_runtime=chat,
            attachment_store=attachments,
            run_event_bus=run_events,
            main_agent_runtime=main,
            mode=environment.mode,
            _closers=tuple(closers),
        )
    except BaseException:
        cleanup_errors: list[BaseException] = []
        for closer in reversed(closers):
            try:
                await closer()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary error
                cleanup_errors.append(cleanup_error)
        raise


async def create_application_services_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    conversation_store: ConversationStore | None = None,
    attachment_store: AttachmentStore | None = None,
) -> ApplicationServices:
    return await create_application_services(
        ApplicationEnvironment.from_environment(environ),
        conversation_store=conversation_store,
        attachment_store=attachment_store,
    )


def _create_chat_runtime(
    environment: ApplicationEnvironment, store: ConversationStore
) -> ConversationRuntime:
    return ConversationRuntime(
        api_key=environment.api_key,
        model=environment.main_model,
        base_url=environment.base_url,
        conversation_store=store,
    )


async def _create_rag_runtime(environment: ApplicationEnvironment) -> ClosableRuntime | None:
    if not environment.corpus_configured:
        return None
    from paper_research_agent.web.runtime import RAGRuntime

    if environment.mode == "primary" or RAGRuntime.research_agent_enabled_from_environment():
        return await RAGRuntime.from_environment_with_agent()
    return RAGRuntime.from_environment()


def _create_main_model(environment: ApplicationEnvironment) -> object:
    return create_main_agent_model(
        model_id=environment.main_model,
        api_key=environment.api_key,
        base_url=environment.base_url,
        timeout_seconds=environment.timeout_seconds,
    )


async def _open_main_checkpoint(path: Path) -> _MainCheckpoint:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(path)
    try:
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()
        return _MainCheckpoint(checkpointer=checkpointer, connection=connection)
    except BaseException:
        await connection.close()
        raise


def _environment_path(
    source: Mapping[str, str], root: Path, name: str, default: str
) -> Path:
    raw = source.get(name, default).strip() or default
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


def _timeout_from_environment(source: Mapping[str, str]) -> float:
    raw = source.get("PRA_MAIN_AGENT_TIMEOUT_SECONDS", "180").strip()
    try:
        timeout = float(raw)
    except ValueError as error:
        raise ValueError("PRA_MAIN_AGENT_TIMEOUT_SECONDS must be numeric") from error
    if timeout <= 0 or timeout > 3600:
        raise ValueError("PRA_MAIN_AGENT_TIMEOUT_SECONDS must be between 0 and 3600")
    return timeout


def _strict_boolean_from_environment(
    source: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true or false")
