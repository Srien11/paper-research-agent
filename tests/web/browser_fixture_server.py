"""Loopback-only primary-mode server for the real-browser release gate."""

from __future__ import annotations

import os
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.control import AgentRunControl
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ConversationWorkspace,
    GoalState,
    MainAgentRequest,
    MainAgentResult,
    TaskPlan,
)
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from tests.web.test_main_agent_end_to_end import (
    _LegacyRuntime,
    _RunStore,
    _ScenarioRuntime,
)


class _BrowserRuntime(_ScenarioRuntime):
    def __init__(self, store: _RunStore) -> None:
        super().__init__(store)
        self.controls: dict[str, AgentRunControl] = {}
        self.workspaces: dict[str, ConversationWorkspace] = {}

    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        self._prepare_plan(request)
        fixture_request = request.model_copy(
            update={"message": "local::real-browser-release-gate"}
        )
        result = await super().run(fixture_request)
        self.controls[request.request_id] = self.controls[request.request_id].model_copy(
            update={"status": "completed"}
        )
        workspace = self.workspaces[request.request_id]
        plan = workspace.task_plan
        assert plan is not None
        completed = plan.tasks[0].model_copy(
            update={"status": "completed", "result_ref": "child-1"}
        )
        self.workspaces[request.request_id] = workspace.model_copy(
            update={"task_plan": plan.model_copy(update={"tasks": (completed,)})}
        )
        return result

    async def load_workspace_for_run(self, request_id: str):
        control = self.controls.get(request_id)
        workspace = self.workspaces.get(request_id)
        return (control, workspace) if control is not None and workspace is not None else None

    def _prepare_plan(self, request: MainAgentRequest) -> None:
        now = datetime.now(UTC)
        goal_id = "a" * 32
        self.controls[request.request_id] = AgentRunControl(
            request_id=request.request_id,
            run_id=request.request_id.replace("web_", "run_"),
            conversation_id=request.conversation_id,
            updated_at=now,
        )
        self.workspaces[request.request_id] = ConversationWorkspace(
            conversation_id=request.conversation_id,
            active_goal=GoalState(
                goal_id=goal_id,
                objective="回答本地论文研究问题",
                origin_turn_id="b" * 32,
                created_at=now,
                updated_at=now,
            ),
            task_plan=TaskPlan(
                plan_id="c" * 32,
                goal_id=goal_id,
                tasks=(
                    AgentTask(
                        task_id="local-evidence",
                        goal_id=goal_id,
                        title="检索本地论文证据",
                        objective="形成有引用的回答",
                        success_criteria=("返回至少一个可追溯来源",),
                        capability="local_rag",
                        execution_reason="先建立可追溯的论文证据基础",
                    ),
                ),
                created_at=now,
                updated_at=now,
            ),
            updated_at=now,
        )


def create_browser_fixture_app():
    """Create the production UI and API with no provider or corpus network access."""
    os.environ["PRA_MAIN_AGENT_MODE"] = "primary"
    store = _RunStore()
    runtime = _BrowserRuntime(store)
    config = WebConfig(
        credentials=OwnerCredentials(
            username=os.environ.get("PRA_BROWSER_USER", "owner"),
            password=os.environ.get(
                "PRA_BROWSER_PASSWORD", "local-browser-test-password"
            ),
        ),
        session_secret=b"browser-release-gate-secret-32b!",
        allowed_origins=frozenset({"http://127.0.0.1:8092"}),
        cookie_secure=False,
    )
    return create_app(
        config=config,
        runtime=_LegacyRuntime(),  # type: ignore[arg-type]
        conversation_store=store,
        main_agent_runtime=runtime,  # type: ignore[arg-type]
    )


app = create_browser_fixture_app()
