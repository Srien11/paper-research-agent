"use strict";

const fs = require("fs");
const path = require("path");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));

const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";
const outputDir = process.env.PRA_BROWSER_OUTPUT || path.join("data", "runtime", "web-browser-v2");
fs.mkdirSync(outputDir, { recursive: true });

function streamEvents(requestId) {
  const common = {
    schema_version: "main-agent-stream-v2",
    occurred_at: "2026-08-20T08:00:00Z",
    request_id: requestId,
    run_id: "run-browser",
    turn_id: "b".repeat(32),
    parent_node_id: null,
    task_id: null,
    duration_ms: null,
    detail: {},
    delta: null,
  };
  return [
    { ...common, event_id: 1, type: "run_started", node_id: "run:browser", status: "running", title: "开始运行", summary: "准备上下文" },
    { ...common, event_id: 2, type: "reasoning_started", node_id: "reasoning:main", status: "running", title: "理解当前对话", summary: "正在准备" },
    { ...common, event_id: 3, type: "task_started", node_id: "task:one", task_id: "one", status: "running", title: "检索论文", summary: "查找公开来源", detail: { capability: "local_rag" } },
    { ...common, event_id: 4, type: "tool_started", node_id: "tool:one:search:1", parent_node_id: "task:one", task_id: "one", status: "running", title: "调用工具 search_corpus", summary: "执行中", detail: { tool_name: "search_corpus", delivery_mode: "event_only" } },
    { ...common, event_id: 5, type: "tool_completed", node_id: "tool:one:search:1", parent_node_id: "task:one", task_id: "one", status: "completed", title: "工具已完成", summary: "返回 2 项结果", duration_ms: 18, detail: { tool_name: "search_corpus", returned_count: 2, delivery_mode: "event_only" } },
    { ...common, event_id: 6, type: "answer_started", node_id: "answer:one", parent_node_id: "task:one", task_id: "one", status: "running", title: "生成回答", detail: { delivery_mode: "provider_live" } },
    { ...common, event_id: 7, type: "answer_delta", node_id: "answer:one", parent_node_id: "task:one", task_id: "one", status: "running", detail: { delivery_mode: "provider_live" }, delta: "这是实时生成的可验证回答。" },
    { ...common, event_id: 8, type: "answer_completed", node_id: "answer:one", parent_node_id: "task:one", task_id: "one", status: "completed", title: "回答完成", detail: { delivery_mode: "provider_live", total_tokens: 42 } },
    { ...common, event_id: 9, type: "reasoning_completed", node_id: "reasoning:main", status: "completed", title: "研究过程完成", summary: "已完成研究与整理" },
    { ...common, event_id: 10, type: "run_completed", node_id: "run:browser", status: "completed", title: "运行完成", summary: "运行完成" },
  ];
}

async function installRoutes(page, diagnostics) {
  page.on("pageerror", (error) => diagnostics.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") diagnostics.push(message.text());
  });
  await page.route("**/paper-research/api/**", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname.endsWith("/api/session")) {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify({ authenticated: true, conversation_id: "conversation-browser", max_question_chars: 2000 }) });
    }
    if (pathname.endsWith("/api/conversations") && request.method() === "GET") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify({ current_conversation_id: "conversation-browser", conversations: [] }) });
    }
    if (/\/api\/agent\/runs\/[^/]+\/plan$/.test(pathname)) {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify({ objective: "浏览器验证", control: { status: "completed", revision: 0 }, tasks: [] }) });
    }
    if (pathname.endsWith("/api/agent/runs") && request.method() === "POST") {
      const payload = JSON.parse(request.postData() || "{}");
      const events = streamEvents(payload.request_id);
      return route.fulfill({
        status: 200,
        contentType: "application/x-ndjson",
        headers: { "X-Accel-Buffering": "no" },
        body: `${events.map((event) => JSON.stringify(event)).join("\n")}\n`,
      });
    }
    return route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "mock route missing" }) });
  });
}

async function desktopChecks(browser, diagnostics) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await installRoutes(page, diagnostics);
  await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
  await page.locator("#app-view").waitFor({ state: "visible" });

  const messagesWidth = (await page.locator("#messages").boundingBox()).width;
  if (messagesWidth < 720) throw new Error(`center transcript is too narrow: ${messagesWidth}`);
  if (await page.locator("#inspector").evaluate((node) => node.classList.contains("is-open"))) {
    throw new Error("inspector must be closed initially");
  }
  const composerHeight = (await page.locator("#ask-form").boundingBox()).height;
  if (composerHeight > 76) throw new Error(`idle composer is too tall: ${composerHeight}`);

  await page.getByLabel("研究问题").fill("浏览器验证问题");
  await page.getByRole("button", { name: "发送消息" }).click();
  await page.locator(".run-node-turn-tail").waitFor();
  const toolNodes = page.locator('[data-node-id="tool:one:search:1"]');
  if (await toolNodes.count() !== 1) throw new Error("tool node was not updated in place");
  if (!(await toolNodes.first().getAttribute("class")).includes("is-completed")) {
    throw new Error("tool node did not reach completed state");
  }
  if (await page.evaluate(() => localStorage.getItem("paper-research.pending-request.v1")) !== null) {
    throw new Error("completed cursor was not cleared");
  }

  const perf = await page.evaluate(() => {
    const transcript = document.createElement("div");
    document.body.append(transcript);
    const runView = { transcript, registry: new Map(), state: { nodes: {}, order: [], lastEventId: 0, lastEventType: "" } };
    const started = performance.now();
    for (let index = 1; index <= 1000; index += 1) {
      renderRunEvent(runView, {
        event_id: index,
        type: index === 1 ? "answer_started" : "answer_delta",
        node_id: "answer:performance",
        status: "running",
        delta: index === 1 ? null : "x",
        detail: { delivery_mode: "provider_live" },
      });
    }
    const elapsed = performance.now() - started;
    transcript.remove();
    return elapsed;
  });
  if (perf > 100) throw new Error(`1000-event render exceeded 100ms: ${perf}`);

  const scroll = await page.evaluate(async () => {
    const messages = document.querySelector("#messages");
    messages.style.flex = "none";
    messages.style.height = "180px";
    messages.style.minHeight = "180px";
    messages.style.maxHeight = "180px";
    messages.style.overflowY = "auto";
    for (let index = 0; index < 80; index += 1) {
      const row = document.createElement("div");
      row.style.height = "20px";
      messages.append(row);
    }
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    messages.scrollTop = 0;
    const before = messages.scrollTop;
    scrollMessages();
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const anchored = messages.scrollTop === before;
    scrollMessages(true);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const followed = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 4;
    return { anchored, followed };
  });
  if (!scroll.anchored || !scroll.followed) throw new Error(`scroll anchoring failed: ${JSON.stringify(scroll)}`);

  const inspectorToggle = page.getByRole("button", { name: "详情", exact: true });
  await inspectorToggle.focus();
  await inspectorToggle.click();
  if (!(await page.locator("#inspector").evaluate((node) => node.classList.contains("is-open")))) {
    throw new Error("inspector did not open from the details toggle");
  }
  const inspectorClose = page.locator("#inspector-close");
  if (!(await inspectorClose.isVisible())) {
    throw new Error("desktop inspector close button is not visible");
  }
  await inspectorClose.click();
  if (await page.locator("#inspector").evaluate((node) => node.classList.contains("is-open"))) {
    throw new Error("desktop inspector close button did not close the drawer");
  }
  await inspectorToggle.click();
  await page.keyboard.press("Escape");
  if (!(await inspectorToggle.evaluate((node) => node === document.activeElement))) {
    throw new Error("inspector focus was not restored");
  }
  await page.screenshot({ path: path.join(outputDir, "desktop-transcript.png"), fullPage: true });
  await context.close();
}

async function mobileChecks(browser, diagnostics) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const page = await context.newPage();
  await installRoutes(page, diagnostics);
  await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
  await page.locator("#app-view").waitFor({ state: "visible" });
  if (!(await page.locator("#messages").isVisible()) || !(await page.locator("#ask-form").isVisible())) {
    throw new Error("mobile transcript or composer is not visible on first screen");
  }
  const navigationToggle = page.getByRole("button", { name: "会话", exact: true });
  await navigationToggle.click();
  await page.locator("#library-panel.is-open").waitFor();
  await page.keyboard.press("Escape");
  if (await page.locator("#library-panel").evaluate((node) => node.classList.contains("is-open"))) {
    throw new Error("mobile navigation did not close with Escape");
  }
  if (!(await navigationToggle.evaluate((node) => node === document.activeElement))) {
    throw new Error("mobile navigation focus was not restored");
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 1) throw new Error(`mobile horizontal overflow: ${overflow}`);
  await page.screenshot({ path: path.join(outputDir, "mobile-transcript.png"), fullPage: true });
  await context.close();
}

async function main() {
  const options = { headless: true };
  if (process.env.PRA_BROWSER_EXECUTABLE) options.executablePath = process.env.PRA_BROWSER_EXECUTABLE;
  const browser = await chromium.launch(options);
  const diagnostics = [];
  try {
    await desktopChecks(browser, diagnostics);
    await mobileChecks(browser, diagnostics);
    if (diagnostics.length) throw new Error(`browser diagnostics failed: ${JSON.stringify(diagnostics)}`);
    process.stdout.write(JSON.stringify({ ok: true, outputDir }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
