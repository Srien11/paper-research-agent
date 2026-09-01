from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from datetime import UTC, datetime
from typing import cast
from unittest.mock import patch

from fastapi.testclient import TestClient
from httpx import Response

from paper_research_agent.agent.orchestrator.artifacts import (
    AttachmentArtifact,
    ChatArtifact,
    ChildArtifact,
    DynamicToolArtifact,
    FileArtifact,
    LocalRAGArtifact,
    LocalRAGTrace,
)
from paper_research_agent.agent.orchestrator.graph import (
    _selected_recalled_context,
    build_main_agent_graph,
)
from paper_research_agent.agent.orchestrator.hydrator import ContextHydrator
from paper_research_agent.agent.orchestrator.interpreter import TurnInterpreter
from paper_research_agent.agent.orchestrator.models import (
    Capability,
    ChildTaskRequest,
    ChildTaskResult,
    ConversationWorkspace,
    MainAgentRequest,
    MainAgentResult,
    RunStatus,
)
from paper_research_agent.agent.orchestrator.planner import GoalReconciler, TaskPlanner
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
from paper_research_agent.agent.orchestrator.synthesizer import AnswerSynthesizer
from paper_research_agent.answering.models import AnswerCitation, AnswerClaim, RAGAnswer
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.chat_runtime import ConversationRuntime, DirectResponseRequest
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.run_event_bus import RunEventBus

ORIGIN = "https://main-agent-e2e.test"


class _MemoryProvider:
    def __init__(self, memories: tuple[dict[str, object], ...]) -> None:
        self.memories = memories
        self.calls = 0

    async def search(self, query: str, *, limit: int = 5) -> tuple[dict[str, object], ...]:
        del query
        self.calls += 1
        return self.memories[:limit]


class _InterpreterModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.messages: list[object] = []

    def with_structured_output(self, schema: object, method: str = "function_calling") -> object:
        del schema, method
        return self

    async def ainvoke(self, messages: object) -> object:
        self.messages.append(messages)
        return self.response


class _ParallelEvidenceDispatcher:
    def __init__(self) -> None:
        self.calls: list[ChildTaskRequest] = []
        self.started_at: dict[str, float] = {}
        self._both_started = asyncio.Event()

    async def dispatch(self, request: ChildTaskRequest) -> ChildTaskResult:
        self.calls.append(request)
        self.started_at[request.task_id] = time.perf_counter()
        if len(self.started_at) == 2:
            self._both_started.set()
        await asyncio.wait_for(self._both_started.wait(), timeout=1)
        await asyncio.sleep(0.2)
        if request.capability == "local_rag":
            artifact = _local_rag_artifact()
            return ChildTaskResult(
                child_run_id=f"{request.run_id}:local",
                task_id=request.task_id,
                capability="local_rag",
                status="completed",
                summary=artifact.text,
                source_ids=artifact.source_ids,
                citation_kind="local_paper",
                artifact=artifact,
            )
        if request.capability == "dynamic_tools":
            artifact = DynamicToolArtifact(
                text="外部学术元数据已核验。",
                source_ids=("semantic-scholar:paper-1",),
                tool_names=("search_scholarly_sources",),
            )
            return ChildTaskResult(
                child_run_id=f"{request.run_id}:external",
                task_id=request.task_id,
                capability="dynamic_tools",
                status="completed",
                summary=artifact.text,
                source_ids=artifact.source_ids,
                citation_kind="external",
                artifact=artifact,
            )
        raise AssertionError(f"unexpected child capability: {request.capability}")


def _empty_workspace() -> ConversationWorkspace:
    return ConversationWorkspace(
        conversation_id="memory-e2e",
        version=0,
        updated_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


class LongTermMemoryConsumptionIntegrationTests(unittest.TestCase):
    def test_greeting_with_no_memory_keeps_empty_selection(self) -> None:
        provider = _MemoryProvider(())
        envelope = asyncio.run(
            ContextHydrator(
                InMemoryConversationStore(), memory_provider=provider
            ).hydrate(
                MainAgentRequest(
                    request_id="request-greeting",
                    conversation_id="memory-e2e",
                    message="你好",
                    rag_mode="disabled",
                ),
                _empty_workspace(),
                turn_id="a" * 32,
            )
        )
        model = _InterpreterModel(
            {
                "relation": "meta_conversation",
                "resolved_request": "你好",
                "selected_context_ids": (),
                "confidence": 0.9,
            }
        )
        interpretation = asyncio.run(TurnInterpreter(model).interpret(envelope))

        self.assertEqual(provider.calls, 1)
        self.assertEqual(_selected_recalled_context(envelope, interpretation), ())

    def test_selected_preference_reaches_direct_chat_as_untrusted_data(self) -> None:
        memory_id = "b" * 32
        provider = _MemoryProvider(
            (
                {
                    "memory_id": memory_id,
                    "kind": "preference",
                    "content": "用户偏好中文回答",
                    "relevance": 0.9,
                },
            )
        )
        envelope = asyncio.run(
            ContextHydrator(
                InMemoryConversationStore(), memory_provider=provider
            ).hydrate(
                MainAgentRequest(
                    request_id="request-preference",
                    conversation_id="memory-e2e",
                    message="继续",
                    rag_mode="disabled",
                ),
                _empty_workspace(),
                turn_id="b" * 32,
            )
        )
        model = _InterpreterModel(
            {
                "relation": "new_goal",
                "resolved_request": "继续",
                "selected_context_ids": (memory_id,),
                "confidence": 0.9,
            }
        )
        interpretation = asyncio.run(TurnInterpreter(model).interpret(envelope))
        selected = _selected_recalled_context(envelope, interpretation)
        request = DirectResponseRequest(
            session_id="memory-e2e",
            current_message="继续",
            recalled_context=selected,
        )
        runtime = ConversationRuntime.__new__(ConversationRuntime)
        messages = runtime._contextual_messages(request, "继续")

        self.assertEqual(selected[0].memory_kind, "preference")
        self.assertIn("用户偏好中文回答", messages[-2]["content"])
        self.assertEqual(messages[-2]["role"], "user")
        self.assertNotIn("用户偏好中文回答", messages[0]["content"])

    def test_project_context_guides_resolved_continuation(self) -> None:
        memory_id = "c" * 32
        provider = _MemoryProvider(
            (
                {
                    "memory_id": memory_id,
                    "kind": "project_context",
                    "content": "项目正在比较 RAG 与 GraphRAG 的评测指标",
                    "relevance": 0.95,
                },
            )
        )
        envelope = asyncio.run(
            ContextHydrator(
                InMemoryConversationStore(), memory_provider=provider
            ).hydrate(
                MainAgentRequest(
                    request_id="request-project",
                    conversation_id="memory-e2e",
                    message="继续这个项目",
                    rag_mode="preferred",
                ),
                _empty_workspace(),
                turn_id="c" * 32,
            )
        )
        model = _InterpreterModel(
            {
                "relation": "new_goal",
                "resolved_request": "继续比较 RAG 与 GraphRAG 的评测指标",
                "selected_context_ids": (memory_id,),
                "confidence": 0.95,
            }
        )

        interpretation = asyncio.run(TurnInterpreter(model).interpret(envelope))

        self.assertIn("RAG 与 GraphRAG", interpretation.resolved_request)
        self.assertEqual(
            _selected_recalled_context(envelope, interpretation)[0].memory_kind,
            "project_context",
        )


class _LegacyRuntime:
    is_ready = True
    is_busy = False

    async def clear_conversation(self, conversation_id: str) -> int:
        return 0

    async def aclose(self) -> None:
        return None


class _RunStore(InMemoryConversationStore):
    def __init__(self) -> None:
        super().__init__()
        self.results: dict[str, MainAgentResult] = {}

    def load_agent_run(self, request_id: str) -> MainAgentResult | None:
        return self.results.get(request_id)


class _ScenarioRuntime:
    def __init__(self, store: _RunStore) -> None:
        self.store = store
        self.requests: list[MainAgentRequest] = []
        self.side_effect_count = 0

    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        self.requests.append(request)
        cached = self.store.results.get(request.request_id)
        if cached is not None:
            return cached
        scenario = request.message.split("::", maxsplit=1)[0]
        if scenario == "commit-rejected":
            result = self._result(request, status="failed", workspace_version=0)
        else:
            children = self._children(scenario)
            if scenario in {"dynamic", "duplicate-write"}:
                self.side_effect_count += 1
            result = self._result(request, children=children, workspace_version=1)
        self.store.results[request.request_id] = result
        return result

    async def clear(self, conversation_id: str) -> None:
        return None

    def _result(
        self,
        request: MainAgentRequest,
        *,
        children: tuple[ChildTaskResult, ...] = (),
        status: RunStatus = "completed",
        workspace_version: int,
    ) -> MainAgentResult:
        return MainAgentResult(
            run_id=request.request_id.replace("req_", "run_"),
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            status=status,
            answer="generated response" if status == "completed" else "",
            child_results=children,
            workspace_version=workspace_version,
        )

    def _children(self, scenario: str) -> tuple[ChildTaskResult, ...]:
        artifacts: dict[
            str, tuple[tuple[Capability, ChildArtifact, tuple[str, ...]], ...]
        ] = {
            "direct": (("direct_chat", ChatArtifact(text="generated response"), ()),),
            "local": (("local_rag", _local_rag_artifact(), ("chunk-1",)),),
            "hybrid": (
                ("local_rag", _local_rag_artifact(), ("chunk-1",)),
                (
                    "dynamic_tools",
                    DynamicToolArtifact(text="verified", tool_names=("web_search",)),
                    ("external-1",),
                ),
            ),
            "attachment": (
                (
                    "attachment_qa",
                    AttachmentArtifact(text="attachment summary", source_attachment_ids=()),
                    (),
                ),
            ),
            "file": (
                (
                    "file_edit",
                    FileArtifact(text="new file", output_attachment_ids=("f" * 32,)),
                    (),
                ),
            ),
            "dynamic": (
                (
                    "dynamic_tools",
                    DynamicToolArtifact(text="saved", tool_names=("write_file",)),
                    (),
                ),
            ),
            "duplicate-write": (
                (
                    "dynamic_tools",
                    DynamicToolArtifact(text="saved", tool_names=("write_file",)),
                    (),
                ),
            ),
        }
        return tuple(
            ChildTaskResult(
                child_run_id=f"child-{index}",
                task_id=f"task-{index}",
                capability=capability,
                status="completed",
                source_ids=source_ids,
                artifact=artifact,
            )
            for index, (capability, artifact, source_ids) in enumerate(
                artifacts[scenario], start=1
            )
        )


def _local_rag_artifact() -> LocalRAGArtifact:
    answer = RAGAnswer(
        status="answered",
        answer_markdown="supported answer [E1]",
        claims=(AnswerClaim(text="supported answer", citation_ids=("E1",)),),
        citations=(
            AnswerCitation(
                citation_id="E1",
                chunk_id="chunk-1",
                corpus_id="C001",
                asset_id="asset-1",
                page_start=1,
                page_end=1,
                text_sha256="a" * 64,
                storage_class="internal_research_only",
            ),
        ),
        requested_model="test-model",
        actual_model="test-model",
        prompt_version="test-v1",
        latency_ms=1,
        attempts=1,
    )
    return LocalRAGArtifact(
        text=answer.answer_markdown,
        source_ids=("chunk-1",),
        answer=answer,
        retrieval=LocalRAGTrace(
            index_id="test-index",
            resolved_question_sha256="b" * 64,
            hit_count=1,
        ),
    )


def _events(response: Response) -> list[dict[str, object]]:
    return [json.loads(line) for line in str(response.text).splitlines() if line]


class RealHybridMainGraphEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_planner_graph_dispatch_and_synthesis_are_parallel(self) -> None:
        store = InMemoryConversationStore()
        bus = RunEventBus(store)
        interpreter_model = _InterpreterModel(
            {
                "relation": "new_goal",
                "resolved_request": "结合本地论文 C001 和联网搜索核验最新状态",
                "confidence": 0.95,
            }
        )
        planner_model = _InterpreterModel(
            {
                "tasks": (
                    {
                        "task_id": "local-evidence",
                        "title": "检索本地论文",
                        "objective": "核验 C001 的论文结论",
                        "success_criteria": ("获得本地论文证据",),
                        "capability": "local_rag",
                        "execution_reason": "先建立本地证据",
                    },
                )
            }
        )
        synthesis_model = _InterpreterModel(
            {
                "conclusion": "模型背景补充：RAG 的结论应区分语料内证据与外部时效核验。",
                "evidence_sections": (
                    {
                        "task_id": "local-evidence",
                        "text": "本地论文支持该结论。",
                        "source_ids": ("chunk-1",),
                    },
                    {
                        "task_id": "external-research",
                        "text": "外部学术元数据确认了当前状态。",
                        "source_ids": ("semantic-scholar:paper-1",),
                    },
                ),
            }
        )
        dispatcher = _ParallelEvidenceDispatcher()
        graph = build_main_agent_graph(
            repository=store,
            hydrator=ContextHydrator(store),
            interpreter=TurnInterpreter(interpreter_model),  # type: ignore[arg-type]
            goal_reconciler=GoalReconciler(),
            task_planner=TaskPlanner(planner_model),  # type: ignore[arg-type]
            dispatcher=dispatcher,  # type: ignore[arg-type]
            synthesizer=AnswerSynthesizer(synthesis_model),  # type: ignore[arg-type]
            run_event_publisher=bus.publisher,
            parallel_hybrid_research_enabled=True,
        )
        runtime = MainAgentRuntime(
            graph=graph,
            repository=store,
            run_event_publisher=bus.publisher,
        )
        request = MainAgentRequest(
            request_id="req_real_hybrid_e2e_1234",
            conversation_id="conversation-real-hybrid-e2e",
            message="结合本地论文 C001 和联网搜索核验最新状态",
            rag_mode="preferred",
        )

        started = time.perf_counter()
        result = await runtime.run(request)
        elapsed = time.perf_counter() - started

        self.assertEqual(result.status, "completed")
        self.assertLess(elapsed, 1.5)
        self.assertEqual(
            tuple(call.capability for call in dispatcher.calls),
            ("local_rag", "dynamic_tools"),
        )
        self.assertNotIn("direct_chat", tuple(call.capability for call in dispatcher.calls))
        start_times = tuple(dispatcher.started_at.values())
        self.assertLess(max(start_times) - min(start_times), 0.05)
        self.assertEqual(result.route_trace, ("local_rag", "dynamic_tools"))
        self.assertEqual(len(synthesis_model.messages), 1)
        self.assertIn("模型背景补充", result.answer)
        self.assertEqual(result.child_results[0].source_ids, ("chunk-1",))
        self.assertEqual(
            result.child_results[1].source_ids,
            ("semantic-scholar:paper-1",),
        )
        self.assertEqual(result.workspace_version, 1)
        self.assertEqual(store.load_workspace(request.conversation_id).version, 1)
        events = [
            item.to_stream_event()
            for item in store.run_events(request.request_id)
        ]
        self.assertEqual(
            [event.event_id for event in events],
            list(range(1, len(events) + 1)),
        )
        self.assertEqual(sum(event.type == "run_completed" for event in events), 1)
        self.assertIn("parallel_group_started", [event.type for event in events])
        self.assertIn("parallel_group_completed", [event.type for event in events])
        parallel_completed = next(
            event for event in events if event.type == "parallel_group_completed"
        )
        self.assertLess(parallel_completed.duration_ms or 10_000, 320)
        await runtime.aclose()
        await bus.aclose()


class MainAgentEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _RunStore()
        self.runtime = _ScenarioRuntime(self.store)
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="password"),
            session_secret=b"e" * 32,
            allowed_origins=frozenset({ORIGIN}),
        )
        self.environment = patch.dict(os.environ, {"PRA_MAIN_AGENT_MODE": "primary"})
        self.environment.start()
        self.context = TestClient(
            create_app(
                config=config,
                runtime=_LegacyRuntime(),  # type: ignore[arg-type]
                serve_static=False,
                conversation_store=self.store,
                main_agent_runtime=self.runtime,  # type: ignore[arg-type]
            ),
            base_url=ORIGIN,
        )
        self.client = self.context.__enter__()
        login = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "password"},
        )
        self.assertEqual(login.status_code, 200)

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        self.environment.stop()

    def test_all_five_capabilities_project_safe_ordered_events(self) -> None:
        scenarios = {
            "direct": ("task_completed",),
            "local": ("rag_result",),
            "hybrid": ("rag_result", "tool_result"),
            "attachment": ("attachment_result",),
            "file": ("file_result",),
        }
        for index, (message, expected_types) in enumerate(scenarios.items(), start=1):
            with self.subTest(message=message):
                user_message = f"{message}::USER_INPUT_MUST_NOT_BE_ECHOED_{index}"
                response = self._run(f"req_e2e_{index:016d}", user_message)
                events = _events(response)
                event_types = tuple(event["type"] for event in events)
                self.assertEqual(
                    tuple(event["event_id"] for event in events),
                    tuple(range(1, len(events) + 1)),
                )
                self.assertEqual(event_types[-1], "done")
                self.assertEqual(event_types.count("done"), 1)
                for event_type in expected_types:
                    self.assertIn(event_type, event_types)
                self.assertNotIn(user_message, response.text)
        local_events = _events(
            self._run("req_e2e_0000000000000099", "local::citation-preservation")
        )
        rag_event = next(event for event in local_events if event["type"] == "rag_result")
        self.assertEqual(rag_event["source_ids"], ["chunk-1"])

    def test_duplicate_request_reuses_result_without_repeating_write(self) -> None:
        request_id = "req_e2e_duplicate000001"
        first = self._run(request_id, "duplicate-write")
        second = self._run(request_id, "duplicate-write")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(_events(second)[0]["type"], "run_reused")
        self.assertEqual(self.runtime.side_effect_count, 1)

    def test_rag_mode_contract_reaches_runtime_unchanged(self) -> None:
        self._run("req_e2e_required0000001", "local", rag_mode="required")
        self._run("req_e2e_disabled0000001", "direct", rag_mode="disabled")

        self.assertEqual(self.runtime.requests[-2].rag_mode, "required")
        self.assertEqual(self.runtime.requests[-1].rag_mode, "disabled")

    def test_failed_commit_has_error_done_and_unchanged_workspace_version(self) -> None:
        response = self._run("req_e2e_rejected0000001", "commit-rejected")
        events = _events(response)

        self.assertEqual(response.status_code, 200)
        self.assertIn("error", [event["type"] for event in events])
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["status"], "failed")
        self.assertEqual(events[-1]["workspace_version"], 0)

    def _run(
        self, request_id: str, message: str, *, rag_mode: str = "disabled"
    ) -> Response:
        return cast(
            Response,
            self.client.post(
                "/paper-research/api/agent/runs",
                headers={"Origin": ORIGIN},
                json={
                    "request_id": request_id,
                    "message": message,
                    "rag_mode": rag_mode,
                },
            ),
        )


if __name__ == "__main__":
    unittest.main()
