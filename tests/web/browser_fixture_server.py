"""Loopback-only primary-mode server for the real-browser release gate."""

from __future__ import annotations

import os

from paper_research_agent.agent.orchestrator.models import MainAgentRequest, MainAgentResult
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from tests.web.test_main_agent_end_to_end import (
    _LegacyRuntime,
    _RunStore,
    _ScenarioRuntime,
)


class _BrowserRuntime(_ScenarioRuntime):
    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        fixture_request = request.model_copy(
            update={"message": "local::real-browser-release-gate"}
        )
        return await super().run(fixture_request)


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
