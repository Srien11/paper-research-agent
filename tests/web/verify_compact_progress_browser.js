"use strict";

const path = require("path");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));
const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";

async function main() {
  const options = { headless: true };
  if (process.env.PRA_BROWSER_EXECUTABLE) options.executablePath = process.env.PRA_BROWSER_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.route("**/paper-research/api/**", async (route) => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      if (pathname.endsWith("/api/session")) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({ authenticated: true, conversation_id: "conversation-progress" }) });
      }
      if (pathname.endsWith("/api/conversations")) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({ current_conversation_id: "conversation-progress", conversations: [] }) });
      }
      if (/\/api\/agent\/runs\/[^/]+\/plan$/.test(pathname)) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({ objective: "紧凑进度", control: { status: "completed", revision: 0 }, tasks: [] }) });
      }
      return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "mock route missing" }) });
    });
    await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
    await page.locator("#app-view").waitFor({ state: "visible" });

    const progress = await page.evaluate(() => {
      const transcript = document.createElement("div");
      document.body.append(transcript);
      const runView = { transcript, registry: new Map(), state: { nodes: {}, order: [], lastEventId: 0, lastEventType: "" } };
      const common = { schema_version: "main-agent-stream-v2", occurred_at: "2026-08-31T00:00:00Z", request_id: "request_progress_1234", run_id: "run-progress", turn_id: "b".repeat(32), parent_node_id: null, task_id: null, duration_ms: null, detail: {} };
      [
        { ...common, event_id: 1, type: "plan_updated", node_id: "plan:1", status: "running", title: "制定计划", summary: "拆解问题" },
        { ...common, event_id: 2, type: "task_started", node_id: "task:1", task_id: "research", status: "running", title: "检索论文", summary: "查找证据" },
        { ...common, event_id: 3, type: "tool_started", node_id: "tool:1", task_id: "research", status: "running", title: "搜索语料", summary: "执行中" },
      ].forEach((event) => renderRunEvent(runView, event));
      const rows = [...transcript.querySelectorAll(".progress-row")].map((row) => row.innerText);
      const hasCards = transcript.querySelectorAll("details, .run-node").length;
      transcript.remove();
      return { rows, hasCards };
    });
    if (progress.rows.length !== 3 || progress.hasCards) throw new Error(`compact progress contract failed: ${JSON.stringify(progress)}`);

    const terminal = await page.evaluate(async () => {
      const transcript = document.createElement("div");
      document.body.append(transcript);
      const runView = { transcript, copy: document.createElement("div"), registry: new Map(), state: { nodes: {}, order: [], lastEventId: 0, lastEventType: "" } };
      const common = { schema_version: "main-agent-stream-v2", occurred_at: "2026-08-31T00:00:00Z", request_id: "request_progress_5678", run_id: "run-progress", turn_id: "b".repeat(32), parent_node_id: null, task_id: null, duration_ms: null, detail: { delivery_mode: "provider_live" } };
      renderRunEvent(runView, { ...common, event_id: 1, type: "answer_started", node_id: "answer:1", status: "running" });
      renderRunEvent(runView, { ...common, event_id: 2, type: "answer_delta", node_id: "answer:1", status: "running", delta: "最终答案" });
      finalizeRunAnswers(runView);
      finalizeProgress(runView, "completed");
      await new Promise((resolve) => setTimeout(resolve, 240));
      const result = { answer: runView.copy.textContent, progressRows: transcript.querySelectorAll(".progress-row").length };
      transcript.remove();
      return result;
    });
    if (terminal.answer !== "最终答案" || terminal.progressRows !== 0) throw new Error(`terminal cleanup failed: ${JSON.stringify(terminal)}`);
    process.stdout.write(JSON.stringify({ ok: true }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
