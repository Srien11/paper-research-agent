# Chat Workspace Polish Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the chat workspace compact and reliable: centered composer, lightweight Codex-like progress, unambiguous return-to-chat navigation, and durable conversation history.

**Architecture:** Keep the existing static HTML/CSS/JavaScript client and the existing persisted run-event stream. The chat view will render a compact, transient progress strip from plan/execution/tool events and remove it at a terminal answer; the trace workspace remains the persistent detailed event viewer. Conversation-list consistency will be fixed at the server store/API boundary so page refreshes cannot replace an active persisted conversation with an empty list.

**Tech Stack:** FastAPI, Pydantic, SQLite conversation store, vanilla HTML/CSS/JavaScript, unittest, existing Playwright verification scripts.

---

## Confirmed interaction contract

- Composer: equal-width attachment and send affordances flank a visually centered textarea. RAG modes use unboxed, text-first toggles on a separate lightweight row.
- Chat progress: during a run, show compact status rows at the `plan`, `execution`, and `tool` levels. On a successful terminal answer, remove the entire transient progress strip and leave only the answer, citations, and downloadable outputs.
- Exceptions: paused, approval-required, cancelled, and failed runs retain one concise actionable status line; the full event history is available only in the trace workspace.
- Navigation: `对话` always returns to the current chat from knowledge or trace. Sidebar visibility is controlled independently and no longer overloads the meaning of `会话`.
- Persistence: after a first question is accepted, its conversation appears in the left list immediately and survives reload/workspace switching. API sync failure cannot overwrite known history with an empty list.
- Data boundary: do not alter corpus IDs, knowledge-base lineage, document contents, or event visibility policy. The client does not introduce a new body/message cache; server-side persisted conversations remain authoritative.

## Task 1: Lock down conversation listing semantics

**Files:**
- Modify: `src/paper_research_agent/conversation/store.py`
- Modify: `src/paper_research_agent/web/app.py`
- Modify: `tests/web/test_app.py`
- Modify: `tests/web/test_static.py`

**Step 1: Write the failing server tests**

- Create a conversation with its first pending turn/run, then assert `GET /paper-research/api/conversations` includes it with a stable title and explicit active status.
- Simulate a list reload while the run is pending, then assert the same `conversation_id` remains present and may be fetched by the conversation-detail endpoint.
- Verify a completed conversation keeps its existing title/order and that a request from another session cannot read it.

**Step 2: Run the focused tests and verify failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_app -v`

Expected: the new pending-list assertion fails because pending turns are currently excluded by `_persisted_dialogue`.

**Step 3: Implement the smallest authoritative fix**

- Preserve a conversation row when it has a pending turn, using the first user question as its title and the latest durable timestamp.
- Extend only the existing conversation response model with an explicit safe `status`/active indicator if the UI needs it; do not expose prompt bodies beyond the existing detail endpoint.
- Keep `ConversationStore.history()` behavior unchanged for retrieval context; only the list projection changes.
- Make the client merge a successful list payload with the currently active local entry, and retain the prior rendered list when the list request fails. Do not make session-storage archives authoritative.

**Step 4: Run the focused tests and verify pass**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_app tests.web.test_static -v`

Expected: PASS.

**Step 5: Commit**

Commit message: `修复: 保留运行中会话历史`

## Task 2: Make the workspace navigation truthful and reversible

**Files:**
- Modify: `src/paper_research_agent/web/static/index.html`
- Modify: `src/paper_research_agent/web/static/app.js`
- Modify: `src/paper_research_agent/web/static/app.css`
- Modify: `tests/web/test_static.py`
- Modify: `tests/web/verify_browser.js`

**Step 1: Write the failing static/browser checks**

- Assert a dedicated `对话` control exists in the persistent top bar and is available while either full workspace is active.
- Assert the sidebar control has separate naming/behavior and `setWorkspace("chat")` is invoked by every return-to-chat control.
- In the browser fixture, enter knowledge and trace, use `对话`, and assert `#main-content` becomes visible without clearing the current message list.

**Step 2: Run the focused checks and verify failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: FAIL because the existing `会话` control only toggles navigation and the full workspaces lack an explicit return action.

**Step 3: Implement the navigation split**

- Replace the ambiguous top-bar `会话` label with a text-first `对话` return action.
- Keep a separate accessible sidebar-toggle icon with a tooltip/aria-label; preserve mobile drawer behavior.
- Add a compact `返回对话` text action to both full-workspace headers as a redundant, local escape route.
- Preserve keyboard focus when returning from a workspace and never call `resetWorkspace()` as part of workspace navigation.

**Step 4: Run checks and verify pass**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: PASS; then run `node tests/web/verify_browser.js` when the local browser runtime is available.

**Step 5: Commit**

Commit message: `修复: 恢复工作区返回对话入口`

## Task 3: Rebuild the composer around visual symmetry

**Files:**
- Modify: `src/paper_research_agent/web/static/index.html`
- Modify: `src/paper_research_agent/web/static/app.css`
- Modify: `src/paper_research_agent/web/static/app.js`
- Modify: `tests/web/test_static.py`
- Modify: `tests/web/verify_browser.js`

**Step 1: Write the failing layout checks**

- Assert attachment and send controls are distinct, equal-sized controls on the composer input row.
- Assert RAG controls live in a separate text-first tools row and no longer dictate the textarea grid column.
- Add browser bounding-box checks that the textarea center is within a small tolerance of the composer center and attachment/send centers are vertically aligned.

**Step 2: Run checks and verify failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: FAIL because `.file-row` and the textarea currently share one grid row with unequal widths.

**Step 3: Implement the compact composer**

- Structure the composer as `input-row` (attachment, textarea, send) plus `tool-row` (local-paper mode, local-only mode, file chips).
- Use a native checkbox hidden only visually, with an accessible label, text label, selected underline/dot, focus ring, and no visible checkbox box.
- Keep Enter-to-send, Shift+Enter newline, attachment upload, drag/drop, busy/approval modes, and mobile sticky positioning intact.
- Use 13–14px controls, a 15px textarea, 40px symmetric icon controls, and a responsive full-width layout; no new UI framework or component library.

**Step 4: Run checks and verify pass**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: PASS; then run `node tests/web/verify_browser.js` when available and inspect the resulting desktop/mobile screenshots.

**Step 5: Commit**

Commit message: `功能: 优化对话输入工作区`

## Task 4: Replace persistent run cards with transient progress

**Files:**
- Modify: `src/paper_research_agent/web/static/app.js`
- Modify: `src/paper_research_agent/web/static/app.css`
- Modify: `tests/web/test_static.py`
- Modify: `tests/web/verify_history_browser.js`
- Add: `tests/web/verify_compact_progress_browser.js`

**Step 1: Write failing behavior checks**

- Feed stream events covering plan creation, execution progress, and a tool action; assert the chat renders concise ordered progress rows rather than expandable run cards.
- Feed a completed answer; assert the temporary progress container is removed and the assistant article contains only final answer/citations/outputs.
- Feed approval-required, paused, cancelled, and failed terminal events; assert one concise retained status line is visible and trace data stays available.
- Reload a completed conversation; assert restored chat does not recreate the hidden progress transcript while the trace endpoint retains the same events.

**Step 2: Run checks and verify failure**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: FAIL because `renderRunNode()` currently produces bordered `details`/`run-node` cards and restores the full transcript.

**Step 3: Implement the progressive renderer**

- Map safe existing event types into three presentation categories: `计划` (plan), `执行` (lifecycle/task), and `工具` (tool interaction). Do not expose hidden tool arguments or any newly unsafe data.
- Coalesce updates per category/task so a long stream stays compact; animate only the currently active line and respect reduced-motion settings.
- At successful completion, render final answer first, then remove the live progress container after a short unobtrusive transition.
- When restoring completed messages, render only final content; retain the full raw events in persisted storage and show them exclusively through the existing trace page.
- Remove assistant message card borders/backgrounds and use a small, flowing typography treatment; user messages stay visually distinguishable but compact.

**Step 4: Run checks and verify pass**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_static -v`

Expected: PASS; then run `node tests/web/verify_history_browser.js` and `node tests/web/verify_compact_progress_browser.js` when available.

**Step 5: Commit**

Commit message: `功能: 改为紧凑流式执行进度`

## Task 5: Final quality gate and restart

**Files:**
- Modify only if required by test failures: files from Tasks 1–4
- Verify: `tests/web/test_app.py`, `tests/web/test_static.py`, `tests/web/test_knowledge.py`, `tests/web/test_agent_api.py`, `tests/web/test_run_event_bus.py`, `tests/web/test_runtime.py`

**Step 1: Run the full relevant suite**

Run: `.venv\\Scripts\\python.exe -m unittest tests.web.test_app tests.web.test_static tests.web.test_knowledge tests.web.test_agent_api tests.web.test_run_event_bus tests.web.test_runtime -v`

Expected: PASS.

**Step 2: Run static quality checks**

Run: `.venv\\Scripts\\python.exe -m ruff check src tests` and `.venv\\Scripts\\python.exe -m mypy src/paper_research_agent/web src/paper_research_agent/conversation`

Expected: both pass.

**Step 3: Validate the local service**

- Restart the local service on `127.0.0.1:8092`.
- Confirm `/paper-research/readyz` returns `200` and manually verify: send question, observe compact `计划/执行/工具`, wait for answer, enter/leave both workspaces, refresh, and reopen history.

**Step 4: Check the diff and create the final commit**

Run: `git diff --check` then inspect `git status --short`.

Expected: no whitespace errors and no corpus, index, PDF, runtime database, or secret is staged.

**Step 5: Commit**

Commit message: `功能: 精简对话流与工作区导航`

## Out of scope

- New model/tool orchestration behavior, event-schema versioning, or a new frontend framework.
- Changing PDF ingestion, OCR confidence, corpus IDs, knowledge-base lineage, or data-retention policy.
- Persisting chain-of-thought or raw hidden reasoning; the UI uses only existing safe execution summaries.
