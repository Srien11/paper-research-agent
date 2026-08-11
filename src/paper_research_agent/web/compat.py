"""Pure protocol projections for deprecated Web endpoints."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from paper_research_agent.agent.observability import (
    AgentEvent,
    AgentEventSink,
    DeprecatedEndpoint,
    emit_agent_event,
)
from paper_research_agent.agent.orchestrator.models import MainAgentRequest, MainAgentResult
from paper_research_agent.web.events import AgentEventProjector, AgentStreamEvent
from paper_research_agent.web.models import (
    QuestionRequest,
    SafePendingToolApproval,
    ToolResearchResponse,
)

_PUBLIC_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_DEPRECATED_ENDPOINTS = frozenset(
    {"ask", "chat_stream", "tools_run", "tools_approval"}
)


class CompatibilityProjectionError(ValueError):
    """Raised when an old response would require inventing or dropping trusted fields."""


@dataclass(slots=True)
class CompatibilityAdapter:
    """Convert DTOs only; never execute a child runtime or inspect message bodies."""

    deprecated_counts: Counter[str] = field(default_factory=Counter)
    event_sink: AgentEventSink | None = field(default=None, repr=False)
    _pending_by_conversation: dict[str, str] = field(default_factory=dict, repr=False)

    def mark(self, endpoint: DeprecatedEndpoint) -> None:
        if endpoint not in _DEPRECATED_ENDPOINTS:
            raise ValueError("unknown deprecated endpoint")
        self.deprecated_counts[endpoint] += 1
        emit_agent_event(
            self.event_sink,
            AgentEvent(
                run_id="0" * 32,
                occurred_at=datetime.now(UTC),
                event_type="deprecated_endpoint_used",
                status="succeeded",
                component="runtime",
                name="compatibility",
                endpoint=endpoint,
                requested_count=self.deprecated_counts[endpoint],
            ),
        )

    def main_request(
        self,
        payload: QuestionRequest,
        *,
        conversation_id: str,
    ) -> MainAgentRequest:
        supplied = payload.request_id or ""
        request_id = (
            supplied
            if _PUBLIC_REQUEST_ID.fullmatch(supplied)
            else f"compat_{uuid.uuid4().hex}"
        )
        return MainAgentRequest(
            request_id=request_id,
            conversation_id=conversation_id,
            message=payload.question,
            rag_mode=payload.rag_mode,
            attachment_ids=payload.attachment_ids,
        )

    def stream_events(self, result: MainAgentResult) -> tuple[AgentStreamEvent, ...]:
        projector = AgentEventProjector(
            request_id=result.request_id,
            run_id=result.run_id,
        )
        return projector.project_result(result)

    def remember_pending(self, result: MainAgentResult) -> None:
        if result.status == "waiting_approval":
            self._pending_by_conversation[result.conversation_id] = result.request_id
        elif result.status in {"completed", "failed", "cancelled", "conflict"}:
            self._pending_by_conversation.pop(result.conversation_id, None)

    def pending_request_id(self, conversation_id: str) -> str | None:
        return self._pending_by_conversation.get(conversation_id)

    def tool_response(self, result: MainAgentResult) -> ToolResearchResponse:
        if result.status not in {"completed", "waiting_approval"}:
            raise CompatibilityProjectionError("legacy tool response is unavailable")
        return ToolResearchResponse(
            run_id=result.run_id,
            status=(
                "approval_required"
                if result.status == "waiting_approval"
                else "completed"
            ),
            observations=(),
            final_summary=result.answer[:2_000] or None,
            termination_reason=None,
            pending_approval=_safe_pending(result.pending_approval),
        )

    def reject_ask_projection(self) -> None:
        raise CompatibilityProjectionError(
            "旧 /ask 无法安全还原完整检索与生成审计字段；"
            "请改用统一接口 /paper-research/api/agent/runs 并复用 request_id"
        )


def _safe_pending(
    payload: dict[str, object] | None,
) -> SafePendingToolApproval | None:
    if payload is None:
        return None
    return SafePendingToolApproval.model_validate(
        {
            "tool_name": payload.get("tool_name"),
            "purpose": payload.get("purpose"),
            "arguments_sha256": payload.get("arguments_sha256"),
            "expires_at_epoch": payload.get("expires_at_epoch"),
        }
    )
