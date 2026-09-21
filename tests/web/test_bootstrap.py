from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.dynamic.models import DynamicResearchResult
from paper_research_agent.agent.observability import AgentEvent, SQLiteAgentEventLogger
from paper_research_agent.agent.orchestrator.memory import ToolkitLongTermMemoryProvider
from paper_research_agent.agent.orchestrator.models import ChildTaskRequest
from paper_research_agent.agent.tooling.scholarly_providers import CapabilityReadiness
from paper_research_agent.conversation.store import SQLiteConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.bootstrap import (
    ApplicationEnvironment,
    _DynamicChildAdapter,
    create_application_services,
    main_agent_mode_from_environment,
)
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.files import AttachmentStore


class _CapturingPublisher:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event, *, idempotency_key: str):
        del idempotency_key
        self.events.append(event)
        return event


class _StreamingDynamicRuntime:
    extended_tools_enabled = True

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._observer = None
        self.scope = None

    @contextmanager
    def capture_agent_events(self, observer):
        self._observer = observer
        try:
            yield
        finally:
            self._observer = None

    async def run_dynamic_tools(
        self,
        question: str,
        *,
        thread_id: str,
        memory_context=None,
        child_context=None,
        allowed_tool_risks=None,
        allowed_tool_names=None,
    ):
        del question
        self.scope = (
            memory_context,
            child_context,
            allowed_tool_risks,
            allowed_tool_names,
        )
        assert self._observer is not None
        self._observer(
            AgentEvent(
                run_id="d" * 32,
                occurred_at=datetime.now(UTC),
                event_type="tool_started",
                status="started",
                component="tool",
                name="search_corpus",
            )
        )
        self.started.set()
        await self.release.wait()
        self._observer(
            AgentEvent(
                run_id="d" * 32,
                occurred_at=datetime.now(UTC),
                event_type="tool_completed",
                status="succeeded",
                component="tool",
                name="search_corpus",
                duration_ms=18.4,
                returned_count=2,
            )
        )
        return DynamicResearchResult(
            run_id="d" * 32,
            thread_id=thread_id,
            status="completed",
            final_summary="公开汇总",
            termination_reason="router_finished",
        )

    async def resume_dynamic_tools(self, *, thread_id: str, approved: bool):
        del thread_id, approved
        raise AssertionError("resume is not expected")


def _dynamic_request() -> ChildTaskRequest:
    return ChildTaskRequest(
        run_id="run-product-1",
        request_id="req_dynamic_product_1234",
        conversation_id="conversation-product",
        turn_id="b" * 32,
        goal_id="a" * 32,
        goal_objective="检索论文",
        task_id="task-dynamic",
        objective="查找相关工作",
        success_criteria=("找到来源",),
        capability="dynamic_tools",
        current_message="查找相关工作",
        rag_mode="preferred",
        allowed_tool_risks=("network_read",),
        allowed_tool_names=("search_scholarly_sources",),
    )


class _ClosableRuntime:
    def __init__(self) -> None:
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


class _ResearchRuntime:
    extended_tools_enabled = True

    def __init__(self) -> None:
        self.cleared: list[str] = []

    async def clear(self, thread_id: str) -> None:
        self.cleared.append(thread_id)

    async def execute_tool(self, tool_name, arguments, *, run_id=None):
        del tool_name, arguments, run_id
        raise AssertionError("bootstrap must not search memory while building")


class _Checkpoint:
    def __init__(self) -> None:
        self.checkpointer = object()
        self.close_count = 0
        self.cleared_threads: list[tuple[str, ...]] = []

    async def clear_threads(self, thread_ids: tuple[str, ...]) -> None:
        self.cleared_threads.append(thread_ids)

    async def aclose(self) -> None:
        self.close_count += 1


def _environment(root: Path, *, mode: str, api_key: str = "test-key") -> ApplicationEnvironment:
    return ApplicationEnvironment(
        mode=mode,  # type: ignore[arg-type]
        project_root=root,
        conversation_path=root / "conversation.sqlite3",
        attachment_path=root / "uploads",
        main_checkpoint_path=root / "main.sqlite3",
        knowledge_staging_path=root / "knowledge-base",
        intervention_path=root / "interventions.sqlite3",
        knowledge_output_root=root / "knowledge-output",
        api_key=api_key,
        base_url="https://dashscope.example/v1",
        main_model="qwen-test",
        corpus_configured=False,
        main_agent_fast_path_enabled=True,
        parallel_hybrid_research_enabled=True,
    )


class ApplicationBootstrapTests(unittest.IsolatedAsyncioTestCase):
    def test_main_agent_capacity_environment_is_bounded(self) -> None:
        default = ApplicationEnvironment.from_environment(
            {"PRA_PROJECT_ROOT": str(Path.cwd())}
        )
        configured = ApplicationEnvironment.from_environment(
            {
                "PRA_PROJECT_ROOT": str(Path.cwd()),
                "PRA_MAIN_AGENT_MAX_INFLIGHT_RUNS": "3",
            }
        )

        self.assertEqual(default.max_inflight_runs, 2)
        self.assertEqual(configured.max_inflight_runs, 3)
        for invalid in ("0", "17", "many"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "PRA_MAIN_AGENT_MAX_INFLIGHT_RUNS",
            ):
                ApplicationEnvironment.from_environment(
                    {
                        "PRA_PROJECT_ROOT": str(Path.cwd()),
                        "PRA_MAIN_AGENT_MAX_INFLIGHT_RUNS": invalid,
                    }
                )

    def test_web_runtime_capacity_environment_is_bounded(self) -> None:
        default = ApplicationEnvironment.from_environment(
            {"PRA_PROJECT_ROOT": str(Path.cwd())}
        )
        configured = ApplicationEnvironment.from_environment(
            {
                "PRA_PROJECT_ROOT": str(Path.cwd()),
                "PRA_WEB_MAX_INFLIGHT_RUNS": "4",
            }
        )

        self.assertEqual(default.web_max_inflight_runs, 2)
        self.assertEqual(configured.web_max_inflight_runs, 4)
        for invalid in ("0", "17", "many"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "PRA_WEB_MAX_INFLIGHT_RUNS",
            ):
                ApplicationEnvironment.from_environment(
                    {
                        "PRA_PROJECT_ROOT": str(Path.cwd()),
                        "PRA_WEB_MAX_INFLIGHT_RUNS": invalid,
                    }
                )

    def test_environment_resolves_all_mutable_paths_outside_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory) / "shared"
            environment = ApplicationEnvironment.from_environment(
                {
                    "PRA_PROJECT_ROOT": str(Path(directory) / "release"),
                    "PRA_MAIN_AGENT_MODE": "legacy",
                    "PRA_CONVERSATION_PATH": str(
                        shared / "runtime/conversation.sqlite3"
                    ),
                    "PRA_ATTACHMENT_PATH": str(shared / "runtime/uploads"),
                    "PRA_MAIN_AGENT_CHECKPOINT_PATH": str(
                        shared / "runtime/checkpoint.sqlite3"
                    ),
                    "PRA_KNOWLEDGE_STAGING_PATH": str(
                        shared / "runtime/knowledge-base"
                    ),
                    "PRA_INTERVENTION_PATH": str(
                        shared / "runtime/interventions.sqlite3"
                    ),
                    "PRA_KNOWLEDGE_OUTPUT_ROOT": str(shared / "knowledge"),
                }
            )

            self.assertEqual(
                environment.knowledge_staging_path,
                (shared / "runtime/knowledge-base").resolve(),
            )
            self.assertEqual(
                environment.intervention_path,
                (shared / "runtime/interventions.sqlite3").resolve(),
            )
            self.assertEqual(
                environment.knowledge_output_root,
                (shared / "knowledge").resolve(),
            )

    def test_fast_path_environment_flag_is_strict_and_defaults_true(self) -> None:
        default = ApplicationEnvironment.from_environment(
            {"PRA_PROJECT_ROOT": str(Path.cwd())}
        )
        disabled = ApplicationEnvironment.from_environment(
            {
                "PRA_PROJECT_ROOT": str(Path.cwd()),
                "PRA_MAIN_AGENT_FAST_PATH_ENABLED": "FaLsE",
            }
        )

        self.assertTrue(default.main_agent_fast_path_enabled)
        self.assertFalse(disabled.main_agent_fast_path_enabled)
        with self.assertRaisesRegex(
            ValueError,
            "PRA_MAIN_AGENT_FAST_PATH_ENABLED",
        ):
            ApplicationEnvironment.from_environment(
                {
                    "PRA_PROJECT_ROOT": str(Path.cwd()),
                    "PRA_MAIN_AGENT_FAST_PATH_ENABLED": "1",
                }
            )

    def test_parallel_hybrid_environment_flag_is_strict_and_defaults_true(self) -> None:
        default = ApplicationEnvironment.from_environment(
            {"PRA_PROJECT_ROOT": str(Path.cwd())}
        )
        enabled = ApplicationEnvironment.from_environment(
            {
                "PRA_PROJECT_ROOT": str(Path.cwd()),
                "PRA_PARALLEL_HYBRID_RESEARCH_ENABLED": "TrUe",
            }
        )
        disabled = ApplicationEnvironment.from_environment(
            {
                "PRA_PROJECT_ROOT": str(Path.cwd()),
                "PRA_PARALLEL_HYBRID_RESEARCH_ENABLED": "FALSE",
            }
        )

        self.assertTrue(default.parallel_hybrid_research_enabled)
        self.assertTrue(enabled.parallel_hybrid_research_enabled)
        self.assertFalse(disabled.parallel_hybrid_research_enabled)
        for invalid in ("1", "yes", "", "   ", "enabled"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "PRA_PARALLEL_HYBRID_RESEARCH_ENABLED",
            ):
                ApplicationEnvironment.from_environment(
                    {
                        "PRA_PROJECT_ROOT": str(Path.cwd()),
                        "PRA_PARALLEL_HYBRID_RESEARCH_ENABLED": invalid,
                    }
                )

    async def test_dynamic_tool_events_arrive_before_child_completion(self) -> None:
        runtime = _StreamingDynamicRuntime()
        publisher = _CapturingPublisher()
        adapter = _DynamicChildAdapter(
            runtime,  # type: ignore[arg-type]
            run_event_publisher=publisher,  # type: ignore[arg-type]
        )

        task = asyncio.create_task(adapter.run_task(_dynamic_request()))
        await asyncio.wait_for(runtime.started.wait(), timeout=1)
        for _ in range(10):
            if publisher.events:
                break
            await asyncio.sleep(0)

        self.assertFalse(task.done())
        self.assertEqual([item.type for item in publisher.events], ["tool_started"])
        runtime.release.set()
        await asyncio.wait_for(task, timeout=1)

        self.assertEqual(
            [item.type for item in publisher.events],
            ["tool_started", "tool_completed"],
        )
        self.assertEqual(publisher.events[0].node_id, publisher.events[1].node_id)
        self.assertEqual(publisher.events[1].detail.returned_count, 2)
        self.assertEqual(runtime.scope[2], ("network_read",))
        self.assertEqual(runtime.scope[3], ("search_scholarly_sources",))
        serialized = "\n".join(item.model_dump_json() for item in publisher.events)
        self.assertNotIn(
            "arguments",
            publisher.events[1].detail.model_dump(exclude_none=True),
        )
        self.assertNotIn("查找相关工作", serialized)

    async def test_offline_scholarly_readiness_skips_dynamic_graph(self) -> None:
        runtime = _StreamingDynamicRuntime()
        runtime.external_scholarly_readiness = CapabilityReadiness(
            capability="external_scholarly",
            ready=False,
            reason_code="provider_offline",
            provider_ids=("offline",),
        )
        publisher = _CapturingPublisher()
        adapter = _DynamicChildAdapter(
            runtime,  # type: ignore[arg-type]
            run_event_publisher=publisher,  # type: ignore[arg-type]
        )
        request = _dynamic_request().model_copy(
            update={"parallel_group_id": "hybrid-offline"}
        )

        result = await adapter.run_task(request)

        self.assertEqual(result.termination_reason, "external_research_unavailable")
        self.assertFalse(runtime.started.is_set())
        self.assertEqual(publisher.events, [])

    async def test_primary_mode_builds_main_runtime_with_shared_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = _ClosableRuntime()
            rag = _ClosableRuntime()
            main = _ClosableRuntime()
            checkpoint = _Checkpoint()
            model = Mock()
            model.root_async_client = Mock(close=AsyncMock())
            with (
                patch(
                    "paper_research_agent.web.bootstrap._create_chat_runtime",
                    return_value=chat,
                ) as build_chat,
                patch(
                    "paper_research_agent.web.bootstrap._create_rag_runtime",
                    new=AsyncMock(return_value=rag),
                ) as build_rag,
                patch(
                    "paper_research_agent.web.bootstrap._create_main_model",
                    return_value=model,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._open_main_checkpoint",
                    new=AsyncMock(return_value=checkpoint),
                ),
                patch(
                    "paper_research_agent.web.bootstrap.create_main_agent_runtime_from_model",
                    return_value=main,
                ) as build_main,
            ):
                services = await create_application_services(
                    _environment(root, mode="primary")
                )

            self.assertIs(services.main_agent_runtime, main)
            self.assertIs(
                build_chat.call_args.args[2],
                build_rag.call_args.args[1],
            )
            self.assertEqual(build_chat.call_args.args[2].max_inflight_runs, 2)
            self.assertIs(services.conversation_store, services.main_agent_repository)
            self.assertIsNotNone(services.run_event_bus.publisher)
            self.assertIs(
                build_main.call_args.kwargs["store"], services.conversation_store
            )
            self.assertIs(build_main.call_args.kwargs["checkpointer"], checkpoint.checkpointer)
            self.assertIsInstance(
                build_main.call_args.kwargs["event_sink"], SQLiteAgentEventLogger
            )
            self.assertIsNone(build_main.call_args.kwargs["memory_provider"])
            self.assertTrue(build_main.call_args.kwargs["fast_path_enabled"])
            self.assertTrue(
                build_main.call_args.kwargs["parallel_hybrid_research_enabled"]
            )
            started = services.conversation_store.begin_agent_run(
                request_id="request-checkpoint",
                conversation_id="conversation-a",
                user_question="first",
            )
            await build_main.call_args.kwargs["clear"]("conversation-a")
            self.assertEqual(
                checkpoint.cleared_threads,
                [(f"main::conversation-a::{started.run_id}",)],
            )

            await services.aclose()
            await services.aclose()
            self.assertEqual(main.close_count, 1)
            self.assertEqual(rag.close_count, 1)
            self.assertEqual(chat.close_count, 1)
            self.assertEqual(checkpoint.close_count, 1)
            model.root_async_client.close.assert_awaited_once()

    async def test_primary_mode_injects_long_term_memory_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = _ClosableRuntime()
            rag = _ClosableRuntime()
            rag.research_agent = _ResearchRuntime()
            main = _ClosableRuntime()
            checkpoint = _Checkpoint()
            model = Mock()
            model.root_async_client = Mock(close=AsyncMock())
            with (
                patch(
                    "paper_research_agent.web.bootstrap._create_chat_runtime",
                    return_value=chat,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_rag_runtime",
                    new=AsyncMock(return_value=rag),
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_main_model",
                    return_value=model,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._open_main_checkpoint",
                    new=AsyncMock(return_value=checkpoint),
                ),
                patch(
                    "paper_research_agent.web.bootstrap.create_main_agent_runtime_from_model",
                    return_value=main,
                ) as build_main,
            ):
                services = await create_application_services(
                    _environment(root, mode="primary")
                )

            self.assertIsInstance(
                build_main.call_args.kwargs["memory_provider"],
                ToolkitLongTermMemoryProvider,
            )
            await services.aclose()

    async def test_legacy_mode_does_not_construct_main_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chat = _ClosableRuntime()
            with (
                patch(
                    "paper_research_agent.web.bootstrap._create_chat_runtime",
                    return_value=chat,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_rag_runtime",
                    new=AsyncMock(return_value=None),
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_main_model"
                ) as create_model,
            ):
                services = await create_application_services(
                    _environment(Path(directory), mode="legacy", api_key="")
                )

            self.assertIsNone(services.main_agent_runtime)
            create_model.assert_not_called()
            await services.aclose()
            self.assertEqual(chat.close_count, 1)

    async def test_primary_mode_fails_readiness_without_main_model_key(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "main agent"),
        ):
            await create_application_services(
                _environment(Path(directory), mode="primary", api_key="")
            )

    async def test_partial_primary_failure_closes_every_opened_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chat = _ClosableRuntime()
            rag = _ClosableRuntime()
            checkpoint = _Checkpoint()
            model = Mock()
            model.root_async_client = Mock(close=AsyncMock())
            with (
                patch(
                    "paper_research_agent.web.bootstrap._create_chat_runtime",
                    return_value=chat,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_rag_runtime",
                    new=AsyncMock(return_value=rag),
                ),
                patch(
                    "paper_research_agent.web.bootstrap._create_main_model",
                    return_value=model,
                ),
                patch(
                    "paper_research_agent.web.bootstrap._open_main_checkpoint",
                    new=AsyncMock(return_value=checkpoint),
                ),
                patch(
                    "paper_research_agent.web.bootstrap.create_main_agent_runtime_from_model",
                    side_effect=RuntimeError("main agent build failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "main agent build failed"),
            ):
                await create_application_services(
                    _environment(Path(directory), mode="primary")
                )

            self.assertEqual(chat.close_count, 1)
            self.assertEqual(rag.close_count, 1)
            self.assertEqual(checkpoint.close_count, 1)
            model.root_async_client.close.assert_awaited_once()

    def test_deprecated_enabled_flag_maps_to_primary_only_without_mode(self) -> None:
        with self.assertWarns(DeprecationWarning):
            mode = main_agent_mode_from_environment(
                {"PRA_MAIN_AGENT_ENABLED": "true"}
            )
        self.assertEqual(mode, "primary")
        self.assertEqual(
            main_agent_mode_from_environment(
                {
                    "PRA_MAIN_AGENT_MODE": "legacy",
                    "PRA_MAIN_AGENT_ENABLED": "true",
                }
            ),
            "legacy",
        )


class ApplicationLifespanTests(unittest.TestCase):
    def test_create_app_uses_injected_services_and_delegates_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = _ClosableRuntime()
            services = Mock()
            services.runtime = chat
            services.chat_runtime = chat
            services.main_agent_runtime = None
            services.mode = "legacy"
            services.conversation_store = SQLiteConversationStore(
                root / "conversation.sqlite3"
            )
            services.attachment_store = AttachmentStore(root / "uploads")
            services.aclose = AsyncMock()
            config = WebConfig(
                credentials=OwnerCredentials(username="owner", password="secret"),
                session_secret=b"s" * 32,
                allowed_origins=frozenset({"https://example.test"}),
            )

            app = create_app(
                config=config,
                serve_static=False,
                services_factory=AsyncMock(return_value=services),
            )
            with TestClient(app):
                self.assertIs(app.state.runtime, chat)
                self.assertIs(app.state.chat_runtime, chat)
                self.assertIs(app.state.attachments, services.attachment_store)
                self.assertIs(
                    app.state.conversation.store, services.conversation_store
                )

            services.aclose.assert_awaited_once()
            self.assertEqual(chat.close_count, 0)


if __name__ == "__main__":
    unittest.main()
