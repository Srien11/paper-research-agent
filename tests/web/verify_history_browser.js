"use strict";

const fs = require("fs");
const path = require("path");

const nodeModules = process.env.CODEX_NODE_MODULES;
if (!nodeModules) throw new Error("CODEX_NODE_MODULES is required");
const { chromium } = require(path.join(nodeModules, "playwright"));

const baseURL = process.env.PRA_BROWSER_BASE_URL || "http://127.0.0.1:8092";
const outputDir = process.env.PRA_BROWSER_OUTPUT || path.join("data", "runtime", "web-browser-v1");
fs.mkdirSync(outputDir, { recursive: true });

const currentId = "current-conversation-persisted";
const oldId = "old-conversation-persisted";
const now = "2026-08-11T10:00:00+00:00";

async function main() {
  const options = { headless: true };
  if (process.env.PRA_BROWSER_EXECUTABLE) options.executablePath = process.env.PRA_BROWSER_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.route("**/paper-research/api/session", (route) => route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ authenticated: true, conversation_id: currentId, expires_at: 2000000000 }),
    }));
    await page.route("**/paper-research/api/conversations", (route) => route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        current_conversation_id: currentId,
        conversations: [
          {
            conversation_id: currentId,
            title: "当前问题",
            created_at: now,
            updated_at: now,
            messages: [
              { role: "user", text: "当前问题", status: "pending", created_at: now },
            ],
          },
          {
            conversation_id: oldId,
            title: "旧问题仍应保留",
            created_at: now,
            updated_at: now,
            messages: [
              { role: "user", text: "旧问题仍应保留", status: "sent", created_at: now },
              { role: "assistant", text: "旧回答已从服务端恢复", status: "completed", created_at: now },
            ],
          },
        ],
      }),
    }));
    await page.route("**/paper-research/api/conversations/*/activate", (route) => route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ authenticated: true, conversation_id: oldId, expires_at: 2000000000 }),
    }));

    await page.goto(`${baseURL}/paper-research/`, { waitUntil: "networkidle" });
    await page.evaluate(({ currentId }) => {
      localStorage.setItem("paper-research.pending-request.v1", JSON.stringify({
        requestId: "request_persisted_1234",
        question: "未完成的原问题",
        ragMode: "disabled",
        attachmentIds: [],
        conversationId: currentId,
      }));
    }, { currentId });
    await page.reload({ waitUntil: "networkidle" });

    await page.getByText("旧问题仍应保留", { exact: true }).waitFor();
    const restoredQuestion = await page.getByLabel("研究问题").inputValue();
    if (restoredQuestion !== "未完成的原问题") {
      throw new Error(`pending question was not restored: ${restoredQuestion}`);
    }
    await page.getByText("旧问题仍应保留", { exact: true }).click();
    await page.getByText("旧回答已从服务端恢复", { exact: true }).waitFor();
    await page.screenshot({ path: path.join(outputDir, "persisted-history.png"), fullPage: true });
    if (errors.length) throw new Error(`browser errors: ${JSON.stringify(errors)}`);
    process.stdout.write(JSON.stringify({ ok: true, outputDir }, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
