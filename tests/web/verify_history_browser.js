"use strict";

const path = require("path");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));

const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";
const conversationId = "conversation-reconnect";
const requestId = "request_reconnect_1234";
const now = "2026-08-20T08:00:00Z";

function event(eventId, type, values = {}) {
  return {
    schema_version: "main-agent-stream-v2",
    event_id: eventId,
    type,
    occurred_at: now,
    request_id: requestId,
    run_id: "run-reconnect",
    turn_id: "b".repeat(32),
    node_id: values.node_id || "run:reconnect",
    parent_node_id: null,
    task_id: null,
    status: values.status || "running",
    title: values.title || null,
    summary: values.summary || null,
    duration_ms: null,
    detail: values.detail || {},
    delta: values.delta || null,
  };
}

const historyEvents = [
  event(1, "run_started", { title: "开始运行" }),
  event(2, "answer_started", { node_id: "answer:main", title: "生成回答", detail: { delivery_mode: "provider_live" } }),
  event(3, "answer_delta", { node_id: "answer:main", delta: "前半段", detail: { delivery_mode: "provider_live" } }),
];
const resumedEvents = [
  event(4, "answer_delta", { node_id: "answer:main", delta: "与后半段", detail: { delivery_mode: "provider_live" } }),
  event(5, "answer_completed", { node_id: "answer:main", status: "completed", title: "回答完成", detail: { delivery_mode: "provider_live" } }),
  event(6, "run_completed", { status: "completed", title: "运行完成", summary: "运行完成" }),
];

async function main() {
  const options = { headless: true };
  if (process.env.PRA_BROWSER_EXECUTABLE) options.executablePath = process.env.PRA_BROWSER_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    const requests = { getEvents: [], postRuns: 0 };
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.route("**/paper-research/api/**", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      const pathname = url.pathname;
      if (pathname.endsWith("/api/session")) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({ authenticated: true, conversation_id: conversationId }) });
      }
      if (pathname.endsWith("/api/conversations")) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({
          current_conversation_id: conversationId,
          conversations: [{ conversation_id: conversationId, title: "断线恢复", created_at: now, updated_at: now }],
        }) });
      }
      if (pathname.endsWith(`/api/conversations/${conversationId}`)) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({
          conversation_id: conversationId,
          title: "断线恢复",
          created_at: now,
          updated_at: now,
          has_more_messages: false,
          message_count: 2,
          messages: [
            { role: "user", text: "需要断线恢复的问题", status: "sent", created_at: now },
            { role: "assistant", text: "前半段", status: "processing", created_at: now, request_id: requestId, run_id: "run-reconnect", turn_id: "b".repeat(32), events: historyEvents },
          ],
        }) });
      }
      if (pathname.endsWith(`/api/agent/runs/${requestId}/events`)) {
        requests.getEvents.push(Number(url.searchParams.get("after_event_id")));
        return route.fulfill({
          contentType: "application/x-ndjson",
          body: `${resumedEvents.map((item) => JSON.stringify(item)).join("\n")}\n`,
        });
      }
      if (pathname.endsWith("/api/agent/runs") && request.method() === "POST") {
        requests.postRuns += 1;
      }
      if (/\/api\/agent\/runs\/[^/]+\/plan$/.test(pathname)) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify({ objective: "恢复", control: { status: "completed", revision: 0 }, tasks: [] }) });
      }
      return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "mock route missing" }) });
    });

    await page.addInitScript(({ requestId, conversationId }) => {
      localStorage.setItem("paper-research.pending-request.v1", JSON.stringify({
        request_id: requestId,
        conversation_id: conversationId,
        last_event_id: 3,
      }));
    }, { requestId, conversationId });
    await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
    await page.locator(".run-node-turn-tail").waitFor();

    const answer = await page.locator(".run-answer-copy").innerText();
    if (answer !== "前半段与后半段") throw new Error(`reconnected answer is incomplete: ${answer}`);
    if (requests.getEvents.length !== 1 || requests.getEvents[0] !== 3) {
      throw new Error(`unexpected reconnect cursor: ${JSON.stringify(requests.getEvents)}`);
    }
    if (requests.postRuns !== 0) throw new Error("refresh reposted the original request");
    if (await page.evaluate(() => localStorage.getItem("paper-research.pending-request.v1")) !== null) {
      throw new Error("terminal reconnect cursor was not cleared");
    }
    const cached = await page.evaluate(() => sessionStorage.getItem("paper-research.current-dialogue.v1") || "");
    if (cached.includes("main-agent-stream-v2") || cached.includes("answer_delta")) {
      throw new Error("event ledger leaked into browser history storage");
    }
    if (errors.length) throw new Error(`browser errors: ${JSON.stringify(errors)}`);
    process.stdout.write(JSON.stringify({ ok: true, cursor: requests.getEvents[0] }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
