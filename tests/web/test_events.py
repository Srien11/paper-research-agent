from __future__ import annotations

import unittest

from pydantic import ValidationError

from paper_research_agent.web.events import AgentEventProjector
from paper_research_agent.web.models import AgentRunRequest


class AgentEventContractTests(unittest.TestCase):
    def test_agent_run_request_requires_client_request_id(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRunRequest(message="hello", rag_mode="disabled")
        with self.assertRaises(ValidationError):
            AgentRunRequest(request_id="short", message="hello", rag_mode="disabled")

        request = AgentRunRequest(
            request_id="req_1234567890123456",
            message=" hello ",
            rag_mode="disabled",
        )
        self.assertEqual(request.message, "hello")

    def test_event_sequence_is_monotonic_and_rejects_duplicate_done(self) -> None:
        projector = AgentEventProjector(
            request_id="req_1234567890123456",
            run_id="run-1",
        )
        started = projector.event("run_started")
        done = projector.done(status="completed", workspace_version=2)

        self.assertEqual(started.event_id, 1)
        self.assertEqual(done.event_id, 2)
        self.assertEqual(done.schema_version, "main-agent-stream-v1")
        with self.assertRaisesRegex(RuntimeError, "done event already emitted"):
            projector.done(status="completed", workspace_version=2)

    def test_done_requires_non_negative_workspace_version(self) -> None:
        projector = AgentEventProjector(
            request_id="req_1234567890123456",
            run_id="run-1",
        )
        with self.assertRaises(ValidationError):
            projector.done(status="completed", workspace_version=-1)


if __name__ == "__main__":
    unittest.main()
