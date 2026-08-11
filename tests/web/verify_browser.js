"use strict";

const path = require("path");
const fs = require("fs");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));

const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";
const username = process.env.PRA_BROWSER_USER || "owner";
const password = process.env.PRA_BROWSER_PASSWORD || "local-browser-test-password";
const outputDir = process.env.PRA_BROWSER_OUTPUT || path.join("data", "runtime", "web-browser-v1");
fs.mkdirSync(outputDir, { recursive: true });

function attachDiagnostics(page, diagnostics) {
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      diagnostics.console.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    diagnostics.failedRequests.push(`${request.method()} ${request.url()} ${request.failure()?.errorText || ""}`);
  });
  page.on("response", (response) => {
    if (response.status() >= 400) {
      diagnostics.httpErrors.push(`${response.status()} ${response.url()}`);
    }
  });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (request.method() === "POST" && url.pathname === "/paper-research/api/agent/runs") {
      diagnostics.agentRunRequests.push({ url: request.url(), body: request.postData() || "" });
    }
    if (["/paper-research/api/chat/stream", "/paper-research/api/ask", "/paper-research/api/tools/run"].includes(url.pathname)) {
      diagnostics.legacyRequests.push(`${request.method()} ${url.pathname}`);
    }
  });
}

async function login(page) {
  await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
  await page.getByLabel("用户名").fill(username);
  await page.getByLabel("站长密码").fill(password);
  await page.getByRole("button", { name: "登录研究台" }).click();
  await page.locator("#app-view").waitFor({ state: "visible" });
  await page.getByRole("heading", { name: "向论文提问" }).waitFor();
}

async function main() {
  const launchOptions = { headless: true };
  if (process.env.PRA_BROWSER_EXECUTABLE) {
    launchOptions.executablePath = process.env.PRA_BROWSER_EXECUTABLE;
  }
  const browser = await chromium.launch(launchOptions);
  const diagnostics = {
    console: [],
    pageErrors: [],
    failedRequests: [],
    httpErrors: [],
    agentRunRequests: [],
    legacyRequests: [],
  };
  try {
    const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await desktop.newPage();
    attachDiagnostics(page, diagnostics);
    await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
    if (!(await page.locator("#login-view").isVisible())) throw new Error("anonymous login view is missing");
    if (await page.locator("#app-view").isVisible()) throw new Error("private workspace leaked anonymously");

    await page.getByLabel("用户名").fill(username);
    await page.getByLabel("站长密码").fill(password);
    await page.getByRole("button", { name: "登录研究台" }).click();
    await page.locator("#app-view").waitFor({ state: "visible" });
    await page.locator("#ask-form").waitFor({ state: "visible" });

    await page.getByLabel("使用本地论文知识库").check();
    await page.getByLabel("研究问题").fill("RAGAS 和 ARES 在评估忠实度时有什么主要区别？");
    await page.getByRole("button", { name: "发送" }).click();
    await page.waitForFunction(() => {
      const copy = document.querySelector(".message-assistant .answer-copy");
      return copy && copy.textContent && !copy.textContent.includes("正在");
    }, null, { timeout: 90000 });
    const pendingRequest = await page.evaluate(() => localStorage.getItem("paper-research.pending-request.v1"));
    if (pendingRequest !== null) throw new Error("completed request_id was not cleared");
    if (diagnostics.agentRunRequests.length !== 1) {
      throw new Error(`expected one unified Agent request, got ${diagnostics.agentRunRequests.length}`);
    }
    const agentPayload = JSON.parse(diagnostics.agentRunRequests[0].body);
    if (!/^[A-Za-z0-9_-]{16,128}$/.test(agentPayload.request_id || "")) {
      throw new Error("unified Agent request_id is missing or invalid");
    }
    if (agentPayload.rag_mode !== "preferred") {
      throw new Error(`unexpected rag_mode: ${agentPayload.rag_mode}`);
    }
    diagnostics.agentRunRequests = [{
      url: diagnostics.agentRunRequests[0].url,
      requestId: agentPayload.request_id,
      ragMode: agentPayload.rag_mode,
    }];
    if (diagnostics.legacyRequests.length) {
      throw new Error(`legacy API was used: ${JSON.stringify(diagnostics.legacyRequests)}`);
    }
    await page.locator("#inspector-content").waitFor({ state: "visible" });
    await page.locator("#plan-task-list .plan-task").waitFor({ state: "visible", timeout: 10000 });
    const inspectorText = await page.locator("#inspector-content").innerText();
    for (const expected of ["本地论文检索"]) {
      if (!inspectorText.includes(expected)) throw new Error(`inspector is missing: ${expected}`);
    }
    await page.screenshot({ path: path.join(outputDir, "desktop-answer.png"), fullPage: true });

    const bodyOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (bodyOverflow > 1) throw new Error(`desktop horizontal overflow: ${bodyOverflow}`);
    await page.getByRole("button", { name: "新对话" }).click();
    await page.getByText("从一个可验证的问题开始").waitFor();
    await desktop.close();

    const mobile = await browser.newContext({ viewport: { width: 390, height: 844 } });
    const mobilePage = await mobile.newPage();
    attachDiagnostics(mobilePage, diagnostics);
    await login(mobilePage);
    if (!(await mobilePage.getByRole("button", { name: "上下文", exact: true }).isVisible())) {
      throw new Error("mobile inspector control is missing");
    }
    const mobileOverflow = await mobilePage.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (mobileOverflow > 1) throw new Error(`mobile horizontal overflow: ${mobileOverflow}`);
    await mobilePage.screenshot({ path: path.join(outputDir, "mobile-workspace.png"), fullPage: true });
    await mobile.close();

    if (diagnostics.console.length || diagnostics.pageErrors.length || diagnostics.failedRequests.length || diagnostics.httpErrors.length) {
      throw new Error(`browser diagnostics failed: ${JSON.stringify(diagnostics)}`);
    }
    process.stdout.write(JSON.stringify({ ok: true, outputDir, diagnostics }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
