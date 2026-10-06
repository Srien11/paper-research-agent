"use strict";

const path = require("path");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));

const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";
const conversationId = "conversation-reconnect";
const requestId = "request_reconnect_1234";
const now = "2026-08-20T08:00:00Z";
const scholarlyLookups = [{
  provider: "crossref", tool_name: "resolve_paper_identifier", queried_at: now,
  status: "ok", cache_hit: false, network_accessed: true, returned_count: 1,
  sources: [{ source_id: `X${"a".repeat(16)}`, title: "公开书目标题", identifier: "doi:10.1234/test", url: "https://doi.org/10.1234/test" }],
}];
const cachedLookup = {
  ...scholarlyLookups[0], provider: "arxiv", cache_hit: true, network_accessed: false,
  sources: [{ source_id: `X${"b".repeat(16)}`, title: "公开预印本标题", identifier: "arxiv:1706.03762", url: "https://arxiv.org/abs/1706.03762" }],
};

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
  event(2, "task_completed", { node_id: "task:external", status: "completed", detail: { scholarly_lookups: scholarlyLookups } }),
  event(3, "answer_started", { node_id: "answer:main", title: "生成回答", detail: { delivery_mode: "provider_live" } }),
  event(4, "answer_delta", { node_id: "answer:main", delta: "前半段", detail: { delivery_mode: "provider_live" } }),
];
const resumedEvents = [
  event(5, "task_completed", { node_id: "task:cached", status: "completed", detail: { scholarly_lookups: [cachedLookup] } }),
  event(6, "answer_delta", { node_id: "answer:main", delta: "与后半段 [E1]\n[论文全文](https://arxiv.org/abs/1706.03762)\nhttps://doi.org/10.1234/test。\n[无效链接](javascript:alert(1))", detail: { delivery_mode: "provider_live" } }),
  event(7, "answer_completed", { node_id: "answer:main", status: "completed", title: "回答完成", detail: { delivery_mode: "provider_live", citations: [{ citation_id: "E1", chunk_id: "chunk-history", corpus_id: "C001", title: "历史引用论文", official_url: "https://example.com/history", section_id: null, page_start: 7, page_end: 7, evidence_type: "text", storage_class: "redistributable", excerpt: "历史安全证据预览", final_rank: 1 }] } }),
  event(8, "run_completed", { status: "completed", title: "运行完成", summary: "运行完成" }),
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
        last_event_id: 4,
      }));
    }, { requestId, conversationId });
    await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
    await page.locator(".message-assistant .answer-copy").last().waitFor({ state: "visible" });

    const answer = await page.locator(".message-assistant .answer-copy").last().innerText();
    if (!answer.startsWith("前半段与后半段 [E1]\n论文全文\nhttps://doi.org/10.1234/test。")) throw new Error("reconnected answer or hyperlinks are incomplete");
    const paperLink = page.locator('.answer-copy a[href="https://arxiv.org/abs/1706.03762"]');
    if (await paperLink.count() !== 1) throw new Error("Markdown paper link is not clickable");
    if (await page.locator('.answer-copy a[href="https://doi.org/10.1234/test"]').count() !== 1) throw new Error("plain URL is not clickable or includes trailing punctuation");
    if (await page.locator('.answer-copy a[href^="javascript:"]').count()) throw new Error("unsafe URL became a hyperlink");
    await page.context().route("https://arxiv.org/abs/1706.03762", (route) => route.fulfill({contentType: "text/html", body: "<p>Public paper link destination</p>"}));
    const popupPromise = page.waitForEvent("popup");
    await paperLink.click();
    const popup = await popupPromise;
    await popup.waitForLoadState();
    if (popup.url() !== "https://arxiv.org/abs/1706.03762") throw new Error("paper link opened the wrong destination");
    await popup.close();
    if (await page.locator("#messages details, #messages .run-node").count()) throw new Error("historical run cards were restored");
    await page.getByRole("button", { name: "查看引用 E1" }).click();
    if (!(await page.locator("#evidence-content").innerText()).includes("历史引用论文")) {
      throw new Error("historical citation metadata was not restored");
    }
    await page.locator("#evidence-close").click();
    if (requests.getEvents.length !== 1 || requests.getEvents[0] !== 4) {
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
    const cards = page.locator(".message-assistant .scholarly-source-card");
    if (await cards.count() !== 2) throw new Error("API source cards missing or duplicated");
    const cardText = await page.locator(".scholarly-sources").innerText();
    if (!cardText.includes("Crossref · 实时 API 查询") || !cardText.includes("arXiv · 缓存命中")) throw new Error("source provider or cache label missing");
    if (await page.locator('.scholarly-sources a[href="https://doi.org/10.1234/test"]').count() !== 1) throw new Error("official DOI source link missing");
    await page.reload({ waitUntil: "networkidle" });
    await page.locator(".message-assistant .answer-copy").last().waitFor({ state: "visible" });
    if (await page.locator(".scholarly-source-card").count() !== 2) throw new Error("refresh lost or duplicated source cards");
    if (await page.locator('.answer-copy a[href="https://arxiv.org/abs/1706.03762"]').count() !== 1) throw new Error("refresh lost the answer hyperlink");
    if (requests.postRuns !== 0) throw new Error("source restoration generated a new run");
    if (errors.length) throw new Error(`browser errors after refresh: ${JSON.stringify(errors)}`);
    process.stdout.write(JSON.stringify({ ok: true, cursor: requests.getEvents[0] }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
