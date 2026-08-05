from __future__ import annotations

import unittest

from paper_research_agent.web.routing import (
    RouteContext,
    RouteDecision,
    enforce_route_policy,
)


class RoutePolicyTests(unittest.TestCase):
    def test_attachment_cannot_be_routed_to_web_or_chat(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.9, reason="model choice"),
            RouteContext(
                has_attachments=True,
                local_only=False,
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "attachment_qa")

    def test_file_edit_requires_attachment(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="file_edit", confidence=0.9, reason="model choice"),
            RouteContext(
                has_attachments=False,
                local_only=False,
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "normal_chat")

    def test_local_only_overrides_model_and_never_falls_back_to_web(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.9, reason="model choice"),
            RouteContext(
                has_attachments=False,
                local_only=True,
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "local_rag")

    def test_optional_rag_safely_falls_back_when_unavailable(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="local_rag", confidence=0.7, reason="model choice"),
            RouteContext(
                has_attachments=False,
                local_only=False,
                rag_available=False,
                web_available=False,
            ),
        )
        self.assertEqual(decision.route, "normal_chat")

    def test_required_rag_fails_closed_when_unavailable(self) -> None:
        with self.assertRaises(RuntimeError):
            enforce_route_policy(
                RouteDecision(route="normal_chat", confidence=0.8, reason="model choice"),
                RouteContext(
                    has_attachments=False,
                    local_only=True,
                    rag_available=False,
                    web_available=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
