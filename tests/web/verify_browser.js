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
  const diagnostics = { console: [], pageErrors: [], failedRequests: [], httpErrors: [] };
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
    await page.locator(".question-card").first().waitFor();

    await page.getByLabel("研究问题").fill("RAGAS 和 ARES 在评估忠实度时有什么主要区别？");
    await page.getByRole("button", { name: "开始研究" }).click();
    await page.locator(".claim").first().waitFor({ timeout: 90000 });
    await page.locator("#inspector-content").waitFor({ state: "visible" });
    const inspectorText = await page.locator("#inspector-content").innerText();
    for (const expected of ["解析后问题", "英文检索式", "纳入记忆轮次", "上下文总预算", "输入 Token"]) {
      if (!inspectorText.includes(expected)) throw new Error(`inspector is missing: ${expected}`);
    }
    const citationButton = page.locator(".citation-button").first();
    await citationButton.click();
    await page.locator("#evidence-dialog[open]").waitFor();
    if (!(await page.locator(".evidence-preview").isVisible())) throw new Error("evidence excerpt is missing");
    await page.getByRole("button", { name: "关闭引用证据" }).click();
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
