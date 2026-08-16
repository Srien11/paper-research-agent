from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "src/paper_research_agent/web/static"
DEPLOY_ROOT = PROJECT_ROOT / "deploy"


class StaticWebContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        cls.css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
        cls.javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    def test_login_and_research_workspace_have_accessible_landmarks(self) -> None:
        required_ids = {
            "login-form",
            "username",
            "password",
            "main-content",
            "conversation-history",
            "messages",
            "ask-form",
            "file-input",
            "file-button",
            "file-list",
            "question",
            "pipeline-status",
            "inspector",
            "generation-metrics",
            "evidence-dialog",
            "memories-button",
            "memories-dialog",
            "memories-list",
            "new-conversation",
        }
        all_ids = re.findall(r'id="([^"]+)"', self.html)
        found_ids = set(all_ids)
        self.assertEqual(len(all_ids), len(found_ids))
        self.assertTrue(required_ids.issubset(found_ids))
        self.assertIn('autocomplete="username"', self.html)
        self.assertIn('autocomplete="current-password"', self.html)
        self.assertIn('role="alert"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('aria-controls="inspector"', self.html)
        self.assertIn('href="#main-content"', self.html)
        self.assertIn("clip-path: inset(50%)", self.css)
        self.assertIn(".skip-link:focus", self.css)

    def test_ui_contains_all_visible_operational_states(self) -> None:
        for text in (
            "可视化测试",
            "知识库包含什么",
            "检索证据",
            "重排与组装",
            "生成回答",
            "验证引用",
            "上下文检查器",
            "引用证据",
            "新对话",
        ):
            self.assertIn(text, self.html)
        for state_token in (
            "insufficient_evidence",
            "degraded_reason",
            "resolved_question",
            "standalone_question",
            "selected_history_turn_ids",
            "selected_history_relevances",
            "inherited_across_route",
            "capability_plan",
            'event.type === "tool_result"',
            'event.type === "task_started"',
            'event.type === "task_completed"',
            "renderRunMetrics",
            "stopPlanRefresh",
            "recent_context_turn_count",
            "recalled_candidate_count",
            "interpretation_source",
            "included_memory_turn_count",
            "omitted_evidence_count",
            "is-loading",
            "appendErrorMessage",
            "empty-state",
        ):
            self.assertIn(state_token, self.javascript + self.html)

    def test_server_text_is_only_projected_with_text_content(self) -> None:
        forbidden_property = "inner" + "HTML"
        self.assertNotIn(forbidden_property, self.javascript)
        self.assertIn("textContent", self.javascript)
        self.assertIn("replaceChildren", self.javascript)
        self.assertNotIn("insertAdjacentHTML", self.javascript)
        self.assertNotIn("document.write", self.javascript)

    def test_api_paths_match_the_private_web_contract(self) -> None:
        for path in (
            'session: "api/session"',
            'login: "api/login"',
            'logout: "api/logout"',
            'conversation: "api/conversation"',
            'conversations: "api/conversations"',
            'agentRuns: "api/agent/runs"',
            'memories: "api/memories"',
        ):
            self.assertIn(path, self.javascript)
        self.assertIn("restoreServerHistory", self.javascript)
        self.assertIn("loadConversationMessages", self.javascript)
        self.assertNotIn("hydrateConversation", self.javascript)
        self.assertIn("loadEarlierMessages", self.javascript)
        self.assertIn("message_limit", self.javascript)
        self.assertIn("加载更早消息", self.javascript)
        self.assertIn("activatePersistedConversation", self.javascript)
        self.assertIn("pendingRequest.question", self.javascript)
        self.assertNotIn('chatStream: "api/chat/stream"', self.javascript)
        self.assertNotIn('toolApproval: "api/tools/approval"', self.javascript)
        self.assertNotIn('ask: "api/ask"', self.javascript)
        self.assertIn('method: "DELETE"', self.javascript)
        self.assertIn("JSON.stringify({ username, password })", self.javascript)
        self.assertIn("session?.max_question_chars", self.javascript)
        self.assertIn("elements.question.maxLength = maxQuestionChars", self.javascript)

    def test_frontend_submits_unified_request_and_displays_server_route(self) -> None:
        self.assertIn("使用本地论文知识库", self.html)
        self.assertIn("仅依据本地论文回答", self.html)
        self.assertIn("await streamConversation(pendingRequest)", self.javascript)
        self.assertIn("function createRequestId()", self.javascript)
        self.assertIn("request_id: pendingRequest.requestId", self.javascript)
        self.assertIn("rag_mode: pendingRequest.ragMode", self.javascript)
        self.assertIn('event.type === "route_selected"', self.javascript)
        self.assertIn('event.type === "rag_result"', self.javascript)
        self.assertIn('event.type === "attachment_result"', self.javascript)
        self.assertIn('event.type === "file_result"', self.javascript)
        self.assertIn('event.type === "approval_required"', self.javascript)
        self.assertIn('id="conversation-history"', self.html)
        self.assertIn("archiveCurrentDialogue", self.javascript)
        self.assertNotIn("isFileEditInstruction", self.javascript)
        self.assertNotIn("file_action", self.javascript)
        self.assertNotIn("isCasualGreeting", self.javascript)

    def test_pending_request_persists_only_retry_contract(self) -> None:
        self.assertIn("sessionStorage", self.javascript)
        self.assertIn('saveHistoryItem({ role: "assistant", text', self.javascript)
        self.assertIn("localStorage.setItem(PENDING_REQUEST_KEY", self.javascript)
        self.assertIn("localStorage.removeItem(PENDING_REQUEST_KEY)", self.javascript)
        self.assertIn("loadPendingRequest()", self.javascript)
        self.assertIn("pendingRequest.requestId", self.javascript)
        self.assertNotRegex(
            self.javascript,
            r"(?:sessionStorage|localStorage)\.setItem\([^\n]+(?:citation|evidence|payload)",
        )

    def test_approval_and_file_outputs_use_unified_request_identity(self) -> None:
        self.assertIn("API.agentApproval(state.pendingTool.requestId)", self.javascript)
        self.assertIn("pending_approval", self.javascript)
        self.assertIn("output_attachment_ids", self.javascript)
        self.assertIn("createServerDownloadButton", self.javascript)
        self.assertIn("clearPendingRequest()", self.javascript)

    def test_hybrid_answer_citation_markers_open_evidence_dialog(self) -> None:
        self.assertIn("function renderTextWithCitations", self.javascript)
        self.assertIn(
            'button.addEventListener("click", () => openEvidence(citationId))',
            self.javascript,
        )
        self.assertIn(
            "copy.replaceChildren(renderTextWithCitations(finalText))",
            self.javascript,
        )

    def test_styles_cover_focus_reduced_motion_and_mobile_layout(self) -> None:
        self.assertIn(":focus-visible", self.css)
        self.assertIn("prefers-reduced-motion", self.css)
        self.assertIn("@media (max-width: 430px)", self.css)
        self.assertIn("min-height: 44px", self.css)
        self.assertIn("env(safe-area-inset-bottom)", self.css)


class RecommendationContractTests(unittest.TestCase):
    def test_recommendations_are_versioned_unique_and_nonempty(self) -> None:
        path = PROJECT_ROOT / "configs/web/recommended-questions-v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "recommended-questions-v1")
        questions = payload["questions"]
        self.assertGreaterEqual(len(questions), 5)
        self.assertEqual(len({item["id"] for item in questions}), len(questions))
        for item in questions:
            self.assertTrue(item["category"].strip())
            self.assertTrue(item["title"].strip())
            self.assertTrue(item["prompt"].strip())


class DeploymentAssetContractTests(unittest.TestCase):
    def test_systemd_is_loopback_single_worker_and_hardened(self) -> None:
        service = (DEPLOY_ROOT / "paper-research-agent.service").read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1 --port 8092", service)
        self.assertNotIn("--workers", service)
        self.assertIn("EnvironmentFile=-/etc/zhimo-site-admin.env", service)
        for directive in (
            "NoNewPrivileges=true",
            "PrivateTmp=true",
            "ProtectSystem=strict",
            "ProtectHome=true",
            "UMask=0077",
            "FASTEMBED_CACHE_PATH=/srv/paper-research-agent/model-cache",
            "ReadWritePaths=/srv/paper-research-agent/current/data/runtime /srv/paper-research-agent/model-cache",
        ):
            self.assertIn(directive, service)

    def test_nginx_uses_canonical_prefix_rate_limits_and_16k_body(self) -> None:
        zones = (DEPLOY_ROOT / "nginx-paper-research-zones.conf").read_text(encoding="utf-8")
        locations = (DEPLOY_ROOT / "nginx-paper-research-locations.conf").read_text(
            encoding="utf-8"
        )
        self.assertIn("limit_req_zone", zones)
        self.assertNotIn("location ", zones)
        self.assertIn("location = /paper-research", locations)
        self.assertIn("location ^~ /paper-research/", locations)
        self.assertIn("127.0.0.1:8092", locations)
        self.assertIn("client_max_body_size 16k", locations)
        self.assertIn("client_max_body_size 10m", locations)
        self.assertIn("location = /paper-research/api/files", locations)
        self.assertIn("limit_req zone=paper_research_ask", locations)
        self.assertNotIn("limit_req_zone", locations)
        self.assertNotIn("location ^~ /research/", locations)

    def test_deployment_requires_external_environment_and_can_rollback(self) -> None:
        script = (DEPLOY_ROOT / "deploy_private_web.sh").read_text(encoding="utf-8")
        self.assertIn("paper-research-agent.env", script)
        self.assertIn("mode 600", script)
        self.assertIn("rollback", script)
        self.assertIn("PREVIOUS_TARGET", script)
        self.assertIn("nginx -t", script)
        self.assertIn("/paper-research/readyz", script)
        self.assertIn("Refusing to switch releases or guess-edit", script)
        self.assertIn("paper-research-agent-zones.conf", script)
        self.assertIn("paper-research-agent-locations.conf", script)
        self.assertRegex(script, r"\\\.pdf\$")

    def test_bundle_builder_copies_only_explicit_private_runtime_inputs(self) -> None:
        script = (DEPLOY_ROOT / "build_private_bundle.ps1").read_text(encoding="utf-8")
        for artifact in (
            "chunks.jsonl",
            "vectors.faiss",
            "metadata.sqlite",
            "core_frozen.jsonl",
            "challenge_frozen.jsonl",
        ):
            self.assertIn(artifact, script)
        self.assertIn("Get-FileHash", script)
        self.assertIn("includes_environment = $false", script)
        self.assertIn("includes_pdfs = $false", script)
        self.assertNotIn('Copy-SafeTree "$repo\\data"', script)


if __name__ == "__main__":
    unittest.main()
