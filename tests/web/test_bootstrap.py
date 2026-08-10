from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from paper_research_agent.conversation.store import SQLiteConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.bootstrap import (
    ApplicationEnvironment,
    create_application_services,
    main_agent_mode_from_environment,
)
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.files import AttachmentStore


class _ClosableRuntime:
    def __init__(self) -> None:
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


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
        api_key=api_key,
        base_url="https://dashscope.example/v1",
        main_model="qwen-test",
        corpus_configured=False,
    )


class ApplicationBootstrapTests(unittest.IsolatedAsyncioTestCase):
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

            self.assertIs(services.main_agent_runtime, main)
            self.assertIs(services.conversation_store, services.main_agent_repository)
            self.assertIs(
                build_main.call_args.kwargs["store"], services.conversation_store
            )
            self.assertIs(build_main.call_args.kwargs["checkpointer"], checkpoint.checkpointer)
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
