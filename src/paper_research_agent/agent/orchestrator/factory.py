"""Assembly helpers for the main Agent graph and runtime."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from paper_research_agent.agent.observability import AgentEventSink
from paper_research_agent.agent.orchestrator.children import ChildGraphDispatcher
from paper_research_agent.agent.orchestrator.evaluator import (
    MAX_CHILD_CALLS_PER_RUN,
    MAX_REPLANS_PER_RUN,
)
from paper_research_agent.agent.orchestrator.graph import (
    MainAgentApprovalResumer,
    build_main_agent_graph,
)
from paper_research_agent.agent.orchestrator.hydrator import (
    ContextHydrator,
    LongTermMemoryProvider,
)
from paper_research_agent.agent.orchestrator.interpreter import TurnInterpreter
from paper_research_agent.agent.orchestrator.models import MainAgentResult
from paper_research_agent.agent.orchestrator.planner import GoalReconciler, TaskPlanner
from paper_research_agent.agent.orchestrator.runtime import (
    ApprovalResumer,
    Closer,
    ConversationClearer,
    MainAgentRuntime,
    RunEventPublisherLike,
)
from paper_research_agent.agent.orchestrator.synthesizer import AnswerSynthesizer
from paper_research_agent.conversation.store import ConversationStore


def build_main_agent_runtime(
    *,
    store: ConversationStore,
    hydrator: ContextHydrator,
    interpreter: TurnInterpreter,
    goal_reconciler: GoalReconciler,
    task_planner: TaskPlanner,
    dispatcher: ChildGraphDispatcher,
    synthesizer: AnswerSynthesizer | None = None,
    timeout_seconds: float = 180,
    max_child_calls: int = MAX_CHILD_CALLS_PER_RUN,
    max_replans: int = MAX_REPLANS_PER_RUN,
    checkpointer: Any | None = None,
    approval_resumer: ApprovalResumer | None = None,
    close: Closer | None = None,
    clear: ConversationClearer | None = None,
    event_sink: AgentEventSink | None = None,
    run_event_publisher: RunEventPublisherLike | None = None,
    fast_path_enabled: bool = False,
) -> MainAgentRuntime:
    """Assemble one closable main Agent runtime with a strict Pydantic graph."""
    resolved_synthesizer = synthesizer or AnswerSynthesizer()
    graph = build_main_agent_graph(
        repository=store,
        hydrator=hydrator,
        interpreter=interpreter,
        goal_reconciler=goal_reconciler,
        task_planner=task_planner,
        dispatcher=dispatcher,
        synthesizer=resolved_synthesizer,
        max_child_calls=max_child_calls,
        max_replans=max_replans,
        checkpointer=checkpointer,
        run_event_publisher=run_event_publisher,
        event_sink=event_sink,
        fast_path_enabled=fast_path_enabled,
    )
    resolved_resumer = approval_resumer or MainAgentApprovalResumer(
        repository=store,
        dispatcher=dispatcher,
        synthesizer=resolved_synthesizer,
    ).resume
    return MainAgentRuntime(
        graph=graph,
        repository=store,
        approval_resumer=resolved_resumer,
        timeout_seconds=timeout_seconds,
        close=close,
        clear=clear,
        event_sink=event_sink,
        run_event_publisher=run_event_publisher,
    )


def build_main_agent_runtime_from_model(
    *,
    store: ConversationStore,
    model: BaseChatModel,
    dispatcher: ChildGraphDispatcher,
    timeout_seconds: float = 180,
    max_child_calls: int = MAX_CHILD_CALLS_PER_RUN,
    max_replans: int = MAX_REPLANS_PER_RUN,
    checkpointer: Any | None = None,
    close: Closer | None = None,
    clear: ConversationClearer | None = None,
    event_sink: AgentEventSink | None = None,
    memory_provider: LongTermMemoryProvider | None = None,
    run_event_publisher: RunEventPublisherLike | None = None,
    fast_path_enabled: bool = False,
) -> MainAgentRuntime:
    """Build production stages while sharing one lifecycle-managed model client."""
    return build_main_agent_runtime(
        store=store,
        hydrator=ContextHydrator(
            store,
            memory_provider=memory_provider,
            event_sink=event_sink,
        ),
        interpreter=TurnInterpreter(model),
        goal_reconciler=GoalReconciler(model),
        task_planner=TaskPlanner(model),
        dispatcher=dispatcher,
        synthesizer=AnswerSynthesizer(model),
        timeout_seconds=timeout_seconds,
        max_child_calls=max_child_calls,
        max_replans=max_replans,
        checkpointer=checkpointer,
        close=close,
        clear=clear,
        event_sink=event_sink,
        run_event_publisher=run_event_publisher,
        fast_path_enabled=fast_path_enabled,
    )


async def noop_approval_resumer(request_id: str, approved: bool) -> MainAgentResult:
    """Deterministic placeholder; production wiring supplies real resume."""
    del request_id, approved
    raise RuntimeError("approval resume is not configured")
