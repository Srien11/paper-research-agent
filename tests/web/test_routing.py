from __future__ import annotations

import unittest

from paper_research_agent.web.routing import (
    CapabilityPlan,
    RouteContext,
    RouteDecision,
    enforce_route_policy,
)


class RoutePolicyTests(unittest.TestCase):
    def test_preferred_explicit_knowledge_request_keeps_local_and_dynamic_capabilities(self) -> None:
        plan = CapabilityPlan(
            route="web_research",
            use_local_papers=True,
            use_web_research=True,
            use_dynamic_tools=True,
            research_mode="planned",
            reason="同时核对本地论文和外部研究",
        )

        resolved = plan.enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
                question="结合知识库和最新资料分析大模型测评",
                research_planning_available=True,
            )
        )

        self.assertTrue(resolved.use_local_papers)
        self.assertTrue(resolved.use_dynamic_tools)
        self.assertEqual(resolved.route, "web_research")

    def test_required_rag_disables_external_capabilities(self) -> None:
        plan = CapabilityPlan(
            route="web_research",
            use_local_papers=True,
            use_web_research=True,
            use_dynamic_tools=True,
            research_mode="planned",
            reason="模型建议组合研究",
        )

        resolved = plan.enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="required",
                rag_available=True,
                web_available=True,
                question="只根据本地论文回答",
                research_planning_available=True,
            )
        )

        self.assertEqual(resolved.route, "local_rag")
        self.assertTrue(resolved.use_local_papers)
        self.assertFalse(resolved.use_web_research)
        self.assertFalse(resolved.use_dynamic_tools)

    def test_preferred_mode_keeps_chat_route_but_requires_local_reference(self) -> None:
        resolved = CapabilityPlan(
            route="normal_chat",
            reason="模型判断为普通交流",
        ).enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
            )
        )

        self.assertTrue(resolved.use_local_papers)
        self.assertEqual(resolved.route, "normal_chat")

    def test_preferred_mode_rejects_retrieval_for_pure_greeting_even_if_model_selects_it(
        self,
    ) -> None:
        resolved = CapabilityPlan(
            route="local_rag",
            use_local_papers=True,
            reason="模型误判需要论文证据",
        ).enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
                question="你好！",
            )
        )

        self.assertEqual(resolved.route, "normal_chat")
        self.assertFalse(resolved.use_local_papers)
        self.assertFalse(resolved.use_web_research)
        self.assertFalse(resolved.use_dynamic_tools)

    def test_preferred_mode_honors_model_selected_local_papers(self) -> None:
        resolved = CapabilityPlan(
            route="normal_chat",
            use_local_papers=True,
            reason="语义规划判断需要论文证据",
        ).enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
            )
        )

        self.assertTrue(resolved.use_local_papers)
        self.assertEqual(resolved.route, "normal_chat")

    def test_web_research_uses_dynamic_graph_and_local_papers_when_enabled(self) -> None:
        resolved = CapabilityPlan(
            route="web_research",
            use_web_research=True,
            use_dynamic_tools=False,
            reason="需要联网研究",
        ).enforce(
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
            )
        )

        self.assertTrue(resolved.use_local_papers)
        self.assertTrue(resolved.use_web_research)
        self.assertTrue(resolved.use_dynamic_tools)

    def test_attachment_cannot_be_routed_to_web_or_chat(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.9, reason="model choice"),
            RouteContext(
                has_attachments=True,
                rag_mode="disabled",
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
                rag_mode="disabled",
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "normal_chat")

    def test_required_rag_overrides_model_and_never_falls_back_to_web(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.9, reason="model choice"),
            RouteContext(
                has_attachments=False,
                rag_mode="required",
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
                rag_mode="preferred",
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
                    rag_mode="required",
                    rag_available=False,
                    web_available=True,
                ),
            )

    def test_disabled_rag_rejects_model_local_route(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="local_rag", confidence=0.8, reason="model choice"),
            RouteContext(
                has_attachments=False,
                rag_mode="disabled",
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "normal_chat")

    def test_preferred_rag_keeps_non_rag_route_available(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.8, reason="latest sources"),
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
            ),
        )
        self.assertEqual(decision.route, "web_research")

    def test_preferred_rag_overrides_chat_for_scholarly_comparison(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="normal_chat", confidence=0.8, reason="model choice"),
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
                question="比较这两篇论文的方法和实验指标",
                research_planning_available=True,
            ),
        )

        self.assertEqual(decision.route, "local_rag")
        self.assertEqual(decision.research_mode, "planned")
        self.assertIn("比较研究", decision.reason)

    def test_preferred_rag_does_not_override_simple_chat_or_explicit_web_route(self) -> None:
        chat = enforce_route_policy(
            RouteDecision(route="normal_chat", confidence=0.8, reason="model choice"),
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
                question="你好",
            ),
        )
        web = enforce_route_policy(
            RouteDecision(route="web_research", confidence=0.8, reason="latest sources"),
            RouteContext(
                has_attachments=False,
                rag_mode="preferred",
                rag_available=True,
                web_available=True,
                question="联网比较最新发布的两个模型",
            ),
        )

        self.assertEqual(chat.route, "normal_chat")
        self.assertEqual(web.route, "web_research")

    def test_disabled_rag_never_forces_scholarly_comparison(self) -> None:
        decision = enforce_route_policy(
            RouteDecision(route="normal_chat", confidence=0.8, reason="model choice"),
            RouteContext(
                has_attachments=False,
                rag_mode="disabled",
                rag_available=True,
                web_available=True,
                question="比较这两篇论文的方法和实验指标",
            ),
        )

        self.assertEqual(decision.route, "normal_chat")


if __name__ == "__main__":
    unittest.main()
