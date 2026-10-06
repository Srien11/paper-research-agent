from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "src/paper_research_agent/web/static/app.js"


@unittest.skipUnless(shutil.which("node"), "Node is required for UI behavior tests")
class StreamingUITests(unittest.TestCase):
    def test_failed_history_keeps_validated_child_answers_and_marks_partial(self) -> None:
        source = APP.read_text(encoding="utf-8")
        names = ("runNodeKind", "runNodeId", "reduceRunEvent", "restoreRunState", "renderCompletedRunAnswers", "finalizeProgress", "appendRestoredAssistant", "renderScholarlySources")
        functions = []
        for name in names:
            start = source.index(f"function {name}(")
            end = source.find("\nfunction ", start + 1)
            functions.append(source[start:end])
        script = "\n".join(functions) + r'''
const assert = require("node:assert/strict");
function createElement(tag, cls, text = "") {
  return {tag, className: cls, textContent: text, dataset: {}, children: [],
    append(...nodes) { this.children.push(...nodes); },
    replaceChildren(...nodes) { this.children = nodes; }};
}
const elements = {messages: createElement("div", "messages")};
const state = {runViews: new Map()};
function clearAnswerPreviews() {}
function terminalStatusLabel() { return ["运行失败", "部分结果未完成"]; }
function renderRunNode(node) { return {text: node.text, sources: node.detail.citations}; }
function event(id, type, node, status, delta, detail = {}) {
  return {schema_version: "main-agent-stream-v2", event_id: id, type, node_id: node, status, delta, detail};
}
const events = [
  event(1, "answer_started", "answer:local", "running"),
  event(2, "answer_delta", "answer:local", "running", "validated text"),
  event(3, "answer_completed", "answer:local", "completed", null, {citations: [{citation_id: "E1", chunk_id: "c1"}]}),
  event(4, "answer_delta", "answer:unfinished", "running", "unfinished draft"),
  event(5, "run_failed", "run", "failed"),
];
appendRestoredAssistant("validated text", "failed", events, "req");
const article = elements.messages.children[0];
assert.equal(article.children.length, 4);
assert.equal(article.children[2].textContent, "已完成部分（本轮运行尚未完成）");
assert.equal(article.children[3].text, "validated text");
assert.equal(article.children[3].sources[0].chunk_id, "c1");
const view = {state: restoreRunState(events), copy: {hidden: false}, transcript: createElement("div", "")};
finalizeProgress(view, "failed");
assert.equal(view.copy.hidden, true);
assert.equal(view.transcript.children.length, 3);
assert.equal(view.transcript.children[2].text, "validated text");
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_refresh_restores_exact_nodes_cursor_and_updates_same_history_message(self) -> None:
        source = APP.read_text(encoding="utf-8")
        names = ("runNodeKind", "runNodeId", "reduceRunEvent", "restoreRunState", "reconcilePendingHistory", "saveHistoryItem")
        functions = []
        for name in names:
            start = source.index(f"function {name}(")
            end = source.find("\nfunction ", start + 1)
            functions.append(source[start:end])
        script = "\n".join(functions) + r'''
const assert = require("node:assert/strict");
const state = {history: [], runViews: new Map(), serverConversations: []};
let clears = 0;
function clearPendingRequest() { clears++; }
function savePendingRequest(pending) { state.pendingRequest = pending; }
function renderHistoryList() {}
const sessionStorage = {setItem() {}};
const STORAGE_KEY = "test";
function event(id, type, node, delta, detail = {}) {
  return {schema_version: "main-agent-stream-v2", event_id: id, type, node_id: node, delta, detail};
}
const events = [event(1, "answer_started", "answer:child"), event(2, "answer_delta", "answer:child", "old")];
let restored = restoreRunState(events);
assert.equal(restored.lastEventId, 2);
assert.equal(restored.nodes["answer:restored"], undefined);
assert.equal(restored.nodes["answer:child"].text, "old");
restored = reduceRunEvent(restored, event(3, "answer_delta", "answer:child", " new"));
assert.equal(restored.nodes["answer:child"].text, "old new");
assert.equal(restored.order.length, 1);
restored = reduceRunEvent(restored, event(4, "answer_started", "answer:main", null, {reason_code: "final_answer"}));
restored = reduceRunEvent(restored, event(5, "answer_delta", "answer:main", "final"));
assert.deepEqual(restored.order, ["answer:main"]);
state.runViews.set("req", {state: restoreRunState(events)});
state.history = [{role: "assistant", requestId: "req", status: "running", text: "old", events}];
for (const browserCursor of [0, 1, 999]) {
  const pending = {requestId: "req", lastEventId: browserCursor};
  assert.equal(reconcilePendingHistory(pending).lastEventId, 2);
}
saveHistoryItem({role: "assistant", requestId: "req", text: "final", status: "completed"});
assert.equal(state.history.length, 1);
assert.equal(state.history[0].text, "final");
assert.equal(reconcilePendingHistory({requestId: "req", lastEventId: 1}), null);
assert.equal(clears, 1);
state.history[0].status = "failed";
assert.equal(reconcilePendingHistory({requestId: "req"}), null);
assert.equal(clears, 2);
assert.equal(restoreRunState([{schema_version: "main-agent-preview-v1", event_id: 999}]).lastEventId, 0);
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_refresh_only_subscribes_same_request_and_terminal_does_not_reconnect(self) -> None:
        source = APP.read_text(encoding="utf-8")
        start = source.index("async function retryPendingRequest(")
        end = source.index("\nfunction shouldAutoScroll", start)
        helper_start = source.index("async function fetchAgentStream(")
        helper_end = source.index("async function streamConversation(", helper_start)
        script = source[helper_start:helper_end] + source[start:end] + r'''
const assert = require("node:assert/strict");
const pending = {requestId: "req_existing_1234", conversationId: "conversation", lastEventId: 999};
const view = {state: {lastEventId: 3, lastEventType: "answer_delta"}};
const state = {busy: false, currentConversationId: "conversation", runViews: new Map([[pending.requestId, view]])};
const API = {agentEvents: (id, cursor) => `${id}/events?after=${cursor}`};
const fetched = [];
let fetchFailures = 0, readFailures = 0;
const window = {setTimeout: callback => callback()};
let clears = 0;
function loadPendingRequest() { return pending; }
function savePendingRequest() {}
function clearPendingRequest() { clears++; }
function hideNotice() {}
function showNotice() {}
function setBusy(value) { state.busy = value; }
function activatePlanControl() {}
function scrollMessages() {}
function appendErrorMessage(message) { throw new Error(message); }
async function refreshPlanControl() {}
async function restoreServerHistory() {}
async function fetch(url, options) {
  fetched.push({url, method: options.method});
  if (fetchFailures-- > 0) throw new TypeError("temporary network failure");
  return {};
}
async function consumeAgentStream() {
  if (readFailures-- > 0) {
    view.state.lastEventId = 4;
    throw new TypeError("interrupted reader");
  }
}
(async () => {
  await retryPendingRequest();
  assert.deepEqual(fetched, [{url: "req_existing_1234/events?after=3", method: "GET"}]);
  assert.equal(pending.requestId, "req_existing_1234");
  fetchFailures = 1;
  readFailures = 1;
  await retryPendingRequest();
  assert.deepEqual(fetched.slice(1), [
    {url: "req_existing_1234/events?after=3", method: "GET"},
    {url: "req_existing_1234/events?after=3", method: "GET"},
    {url: "req_existing_1234/events?after=4", method: "GET"},
  ]);
  view.state.lastEventType = "run_completed";
  await retryPendingRequest();
  assert.equal(fetched.length, 4);
  assert.equal(clears, 1);
  view.state.lastEventType = "answer_delta";
  state.pageUnloading = true;
  await retryPendingRequest();
  assert.equal(fetched.length, 4);
  state.pageUnloading = false;
  fetchFailures = 10;
  await assert.rejects(receiveRunEvents(pending), TypeError);
  assert.equal(fetched.length, 8);
  assert.equal(clears, 1);
  assert.equal(isTransientStreamError({status: 401}), false);
  assert.equal(isTransientStreamError({status: 404}), false);
  assert.equal(isTransientStreamError({status: 429}), true);
  assert.equal(isTransientStreamError({name: "AbortError"}), false);
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preview_is_volatile_and_final_answer_replaces_children(self) -> None:
        source = APP.read_text(encoding="utf-8")
        names = ("runNodeKind", "runNodeId", "reduceRunEvent", "clearAnswerPreviews", "renderAnswerPreview")
        functions = []
        for name in names:
            start = source.index(f"function {name}(")
            end = source.find("\nfunction ", start + 1)
            functions.append(source[start:end if end >= 0 else None])
        script = "\n".join(functions) + r'''
const assert = require("node:assert/strict");
function createElement(tag, cls, text = "") {
  return {tag, className: cls, textContent: text, children: [], removed: false,
    append(...items) { this.children.push(...items); },
    querySelector(cls) { return this.children.find(c => c.className === cls.slice(1)); },
    remove() { this.removed = true; }};
}
const view = {previewElements: new Map(), previewSequences: new Map(),
  transcript: createElement("div", ""), copy: {hidden: false},
  state: {nodes: {}, order: [], lastEventId: 0}};
const original = JSON.stringify(view.state);
renderAnswerPreview(view, {node_id: "answer:child", sequence: 1, text: "<script>draft</script>"});
assert.equal(view.previewElements.get("answer:child").children[1].textContent, "<script>draft</script>");
assert.equal(JSON.stringify(view.state), original);
const first = view.previewElements.get("answer:child");
renderAnswerPreview(view, {node_id: "answer:child", sequence: 2, text: ""});
assert.equal(view.previewElements.size, 0);
assert.equal(first.removed, true);
renderAnswerPreview(view, {node_id: "answer:child", sequence: 1, text: "stale"});
assert.equal(view.previewElements.size, 0);
let state = view.state;
function event(id, type, node, delta, detail = {}) {
  state = reduceRunEvent(state, {event_id: id, type, node_id: node, delta, detail});
}
event(1, "answer_started", "answer:child");
event(2, "answer_delta", "answer:child", "old");
event(3, "answer_started", "answer:main", null, {reason_code: "final_answer"});
event(4, "answer_delta", "answer:main", "validated [E1]");
event(5, "answer_completed", "answer:main", null, {citations: [{citation_id: "E1"}]});
assert.deepEqual(state.order, ["answer:main"]);
assert.equal(state.nodes["answer:child"], undefined);
assert.equal(state.nodes["answer:main"].text, "validated [E1]");
assert.equal(state.nodes["answer:main"].detail.citations[0].citation_id, "E1");
assert.equal(reduceRunEvent(state, {event_id: 4, type: "answer_delta", delta: "duplicate"}), state);
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
