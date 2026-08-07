"use strict";

const API = Object.freeze({
  session: "api/session",
  login: "api/login",
  logout: "api/logout",
  conversation: "api/conversation",
  ask: "api/ask",
  chatStream: "api/chat/stream",
  files: "api/files",
  toolRun: "api/tools/run",
  toolApproval: "api/tools/approval",
  memories: "api/memories",
});

const STORAGE_KEY = "paper-research.current-dialogue.v1";
const HISTORY_KEY = "paper-research.dialogue-history.v1";
const state = {
  busy: false,
  citations: new Map(),
  history: [],
  phaseTimer: null,
  pendingTool: null,
  viewingArchive: false,
  attachments: [],
};

const elements = {
  loginView: document.querySelector("#login-view"),
  appView: document.querySelector("#app-view"),
  loginForm: document.querySelector("#login-form"),
  username: document.querySelector("#username"),
  password: document.querySelector("#password"),
  loginError: document.querySelector("#login-error"),
  logout: document.querySelector("#logout"),
  newConversation: document.querySelector("#new-conversation"),
  memoriesButton: document.querySelector("#memories-button"),
  conversationHistory: document.querySelector("#conversation-history"),
  askForm: document.querySelector("#ask-form"),
  question: document.querySelector("#question"),
  askButton: document.querySelector("#ask-button"),
  fileInput: document.querySelector("#file-input"),
  fileButton: document.querySelector("#file-button"),
  fileList: document.querySelector("#file-list"),
  toolMode: document.querySelector("#tool-mode"),
  ragRequired: document.querySelector("#rag-required"),
  ragRequiredOption: document.querySelector("#rag-required-option"),
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#empty-state"),
  loadingTemplate: document.querySelector("#loading-message-template"),
  pipeline: document.querySelector("#pipeline-status"),
  notice: document.querySelector("#notice"),
  inspector: document.querySelector("#inspector"),
  inspectorToggle: document.querySelector("#inspector-toggle"),
  inspectorClose: document.querySelector("#inspector-close"),
  inspectorScrim: document.querySelector("#inspector-scrim"),
  inspectorEmpty: document.querySelector("#inspector-empty"),
  inspectorContent: document.querySelector("#inspector-content"),
  routeMetrics: document.querySelector("#route-metrics"),
  tokenMetrics: document.querySelector("#token-metrics"),
  generationMetrics: document.querySelector("#generation-metrics"),
  citationMap: document.querySelector("#citation-map"),
  evidenceDialog: document.querySelector("#evidence-dialog"),
  evidenceClose: document.querySelector("#evidence-close"),
  evidenceContent: document.querySelector("#evidence-content"),
  toolApprovalDialog: document.querySelector("#tool-approval-dialog"),
  toolApprovalDetails: document.querySelector("#tool-approval-details"),
  toolApprovalReject: document.querySelector("#tool-approval-reject"),
  toolApprovalAccept: document.querySelector("#tool-approval-accept"),
  memoriesDialog: document.querySelector("#memories-dialog"),
  memoriesClose: document.querySelector("#memories-close"),
  memoriesList: document.querySelector("#memories-list"),
  toastRegion: document.querySelector("#toast-region"),
  conversationLabel: document.querySelector("#conversation-label"),
};

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function selectedRagMode() {
  if (!elements.toolMode.checked) return "disabled";
  return elements.ragRequired.checked ? "required" : "preferred";
}

function syncRagControls() {
  elements.ragRequiredOption.hidden = !elements.toolMode.checked;
  if (!elements.toolMode.checked) elements.ragRequired.checked = false;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch (_error) {
      payload = null;
    }
  }
  if (!response.ok) {
    const message = payload && typeof payload.detail === "string"
      ? payload.detail
      : response.status === 401
        ? "登录已过期，请重新登录。"
        : "请求暂时未完成，请稍后重试。";
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return payload || {};
}

function setAuthenticated(authenticated) {
  elements.loginView.hidden = authenticated;
  elements.appView.hidden = !authenticated;
  if (authenticated) {
    restoreHistory();
    window.requestAnimationFrame(() => elements.question.focus());
  } else {
    elements.username.value = "";
    elements.password.value = "";
    window.requestAnimationFrame(() => elements.username.focus());
  }
}

async function checkSession() {
  try {
    const payload = await request(API.session, { method: "GET" });
    setAuthenticated(payload.authenticated !== false);
  } catch (error) {
    setAuthenticated(false);
    if (error.status !== 401) showLoginError("研究服务暂时不可用，请稍后刷新。", true);
  }
}

function showLoginError(message, visible) {
  elements.loginError.textContent = message;
  elements.loginError.hidden = !visible;
}

async function handleLogin(event) {
  event.preventDefault();
  const username = elements.username.value.trim();
  const password = elements.password.value;
  if (!username) {
    showLoginError("请输入用户名。", true);
    elements.username.focus();
    return;
  }
  if (!password) {
    showLoginError("请输入站长密码。", true);
    elements.password.focus();
    return;
  }
  const button = elements.loginForm.querySelector("button[type='submit']");
  button.disabled = true;
  button.classList.add("is-loading");
  showLoginError("", false);
  try {
    await request(API.login, { method: "POST", body: JSON.stringify({ username, password }) });
    setAuthenticated(true);
  } catch (error) {
    showLoginError(error.message, true);
    elements.password.select();
  } finally {
    button.disabled = false;
    button.classList.remove("is-loading");
  }
}

async function handleLogout() {
  try {
    await request(API.logout, { method: "POST", body: "{}" });
  } catch (_error) {
    showToast("退出请求未确认，本地会话已清理。");
  }
  clearLocalDialogue();
  setAuthenticated(false);
}

function setBusy(busy) {
  state.busy = busy;
  elements.askButton.disabled = busy;
  elements.newConversation.disabled = busy;
  elements.question.disabled = busy;
  elements.toolMode.disabled = busy;
  elements.ragRequired.disabled = busy;
  elements.fileButton.disabled = busy;
  elements.askButton.classList.toggle("is-loading", busy);
  elements.messages.setAttribute("aria-busy", String(busy));
  elements.pipeline.hidden = !busy;
  if (busy) startPhaseAnimation();
  else stopPhaseAnimation();
}

function startPhaseAnimation() {
  const phases = Array.from(elements.pipeline.querySelectorAll("[data-phase]"));
  let current = 0;
  phases.forEach((phase) => phase.classList.remove("is-active", "is-complete"));
  phases[0].classList.add("is-active");
  state.phaseTimer = window.setInterval(() => {
    if (current >= phases.length - 1) return;
    phases[current].classList.remove("is-active");
    phases[current].classList.add("is-complete");
    current += 1;
    phases[current].classList.add("is-active");
  }, 1700);
}

function stopPhaseAnimation() {
  if (state.phaseTimer !== null) window.clearInterval(state.phaseTimer);
  state.phaseTimer = null;
}

async function handleAsk(event) {
  event.preventDefault();
  if (state.busy) return;
  const question = elements.question.value.trim();
  if (!question) {
    showNotice("请输入一个研究问题。", "error");
    elements.question.focus();
    return;
  }
  hideNotice();
  if (state.viewingArchive) {
    resetWorkspace();
    state.history.forEach((item) => {
      if (item.role === "user") appendUserMessage(item.text, false);
      else appendRestoredAssistant(item.text, item.status);
    });
    state.viewingArchive = false;
  }
  elements.emptyState.hidden = true;
  appendUserMessage(question, true);
  elements.question.value = "";
  setBusy(true);
  scrollMessages();
  try {
    await streamConversation(question);
  } catch (error) {
    if (error.status === 401) {
      showToast("登录已过期，请重新登录。");
      setAuthenticated(false);
    } else if (error.status === 409) {
      appendErrorMessage("上一项研究仍在处理中，请稍后再试。", true);
    } else {
      appendErrorMessage(error.message, true);
    }
  } finally {
    setBusy(false);
    scrollMessages();
    if (!elements.appView.hidden) elements.question.focus();
  }
}

async function streamConversation(question, sourceNote = "") {
  const activeFiles = [...state.attachments];
  const response = await fetch(API.chatStream, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      attachment_ids: activeFiles.map((item) => item.attachment_id),
      rag_mode: selectedRagMode(),
    }),
  });
  if (!response.ok) {
    let message = "请求暂时未完成，请稍后重试。";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch (_error) {
      // Keep the safe fallback message.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  if (!response.body) throw new Error("浏览器不支持流式输出。请升级浏览器后重试。");

  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(
    createElement("span", "assistant-mark", "研"),
    createElement("strong", "", "智能路由中"),
  );
  if (sourceNote) meta.append(createElement("small", "source-note", sourceNote));
  const copy = createElement("div", "answer-copy natural-answer");
  copy.textContent = "正在判断最合适的处理方式…";
  article.append(meta, copy);
  elements.messages.append(article);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let rawText = "";
  let metrics = null;
  let route = "normal_chat";
  let capabilities = {};
  let ragHandled = false;
  let citationContextActive = false;
  let lastPaint = 0;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "route") {
        route = event.route;
        capabilities = event.capabilities && typeof event.capabilities === "object" ? event.capabilities : {};
        meta.querySelector("strong").textContent = event.label || "研究助手";
        meta.querySelector(".assistant-mark").textContent = route.startsWith("attachment") || route === "file_edit" ? "文" : "研";
        copy.textContent = route === "file_edit" ? "正在修改文件，完成后可下载新文件…" : "";
        if (event.reason) meta.append(createElement("small", "source-note", event.reason));
      } else if (event.type === "rag_context" && event.payload) {
        const normalized = normalizePayload(event.payload);
        citationContextActive = true;
        state.citations.clear();
        normalized.citations.forEach((citation) => {
          if (citation && citation.citation_id) state.citations.set(citation.citation_id, citation);
        });
        renderInspector(normalized);
        const note = normalized.status === "answered" ? "已参考本地论文库" : "本地论文未命中，继续综合回答";
        meta.append(createElement("small", "source-note source-note-rag", note));
      } else if (event.type === "rag_result" && event.payload) {
        article.remove();
        const retrieval = event.payload.retrieval && typeof event.payload.retrieval === "object" ? event.payload.retrieval : {};
        renderAnswer({ ...event.payload, retrieval: { ...retrieval, capability_plan: capabilities } }, true);
        ragHandled = true;
      } else if (event.type === "tool_result" && event.payload) {
        renderToolResearch(event.payload, true);
        ragHandled = true;
      } else if (event.type === "approval_required" && event.payload) {
        article.remove();
        renderToolResearch(event.payload, true);
        ragHandled = true;
      } else if (event.type === "delta" && typeof event.text === "string") {
        rawText += event.text;
        const now = performance.now();
        if (now - lastPaint >= 50) {
          if (route !== "file_edit") {
            copy.textContent = naturalText(rawText);
            scrollMessages();
          }
          lastPaint = now;
        }
      } else if (event.type === "done") {
        metrics = event.metrics || null;
      } else if (event.type === "error") {
        article.remove();
        throw new Error(event.message || "生成回答时发生错误。");
      }
    }
    if (done) break;
  }
  if (ragHandled) return;
  const editMode = route === "file_edit";
  const finalText = (editMode ? rawText : naturalText(rawText)).trim();
  if (editMode) {
    copy.textContent = "文件修改完成，可以下载新文件。";
  } else if (citationContextActive) {
    copy.replaceChildren(renderTextWithCitations(finalText));
  } else {
    copy.textContent = finalText;
  }
  if (editMode && activeFiles.length) {
    article.append(createDownloadButton(finalText, activeFiles[0].filename));
  }
  if (metrics) article.append(renderStreamMetrics(metrics));
  saveHistoryItem({ role: "assistant", text: finalText, metrics });
}

function createDownloadButton(content, originalName) {
  const button = createElement("button", "download-file", "下载修改后的文件");
  button.type = "button";
  button.addEventListener("click", () => {
    const pdf = originalName.toLowerCase().endsWith(".pdf");
    const name = pdf
      ? `${originalName.replace(/\.pdf$/i, "")}_修改版.txt`
      : originalName.replace(/(\.[^.]+)?$/, "_修改版$1");
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  return button;
}

async function uploadSelectedFile(file) {
  if (!file || state.busy) return;
  if (state.attachments.length >= 1) {
    showNotice("第一版每次修改 1 个文件，请先移除当前文件。", "error");
    return;
  }
  elements.fileButton.disabled = true;
  try {
    const response = await fetch(`${API.files}?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": file.type || "application/octet-stream" },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "文件上传失败。");
    state.attachments.push(payload);
    renderFiles();
    elements.question.placeholder = "说明你希望怎样修改这个文件…";
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    elements.fileButton.disabled = false;
    elements.fileInput.value = "";
  }
}

function renderFiles() {
  elements.fileList.replaceChildren();
  state.attachments.forEach((item) => {
    const chip = createElement("span", "file-chip");
    const remove = createElement("button", "", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `移除 ${item.filename}`);
    remove.addEventListener("click", async () => {
      try {
        await request(`${API.files}/${item.attachment_id}`, { method: "DELETE", body: "{}" });
      } catch (_error) {
        // The local reference can still be removed if cleanup is unavailable.
      }
      state.attachments = state.attachments.filter((file) => file.attachment_id !== item.attachment_id);
      renderFiles();
    });
    chip.append(createElement("span", "", item.filename), remove);
    elements.fileList.append(chip);
  });
  if (!state.attachments.length) {
    elements.question.placeholder = "输入一个关于大模型评测、RAG 或检索研究的问题…";
  }
}

function naturalText(value) {
  return String(value || "")
    .replace(/^\s*#{1,6}\s*/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/```[^\n]*\n?/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n");
}

function renderTextWithCitations(value) {
  const fragment = document.createDocumentFragment();
  const text = String(value || "");
  const pattern = /\[E[1-9]\d*\]/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > cursor) fragment.append(document.createTextNode(text.slice(cursor, index)));
    const marker = match[0];
    const citationId = marker.slice(1, -1);
    if (state.citations.has(citationId)) {
      const button = createElement("button", "citation-button inline-citation", marker);
      button.type = "button";
      button.setAttribute("aria-label", `查看引用 ${citationId}`);
      button.addEventListener("click", () => openEvidence(citationId));
      fragment.append(button);
    } else {
      fragment.append(document.createTextNode(marker));
    }
    cursor = index + marker.length;
  }
  if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
  return fragment;
}

function renderStreamMetrics(metrics) {
  const box = createElement("div", "stream-metrics");
  const seconds = (Number(metrics.elapsed_ms || 0) / 1000).toFixed(1);
  const first = (Number(metrics.first_token_ms || 0) / 1000).toFixed(1);
  box.textContent = `总耗时 ${seconds} 秒 · 首字等待（近似思考）${first} 秒 · 输入 ${Number(metrics.input_tokens || 0)} Token · 输出 ${Number(metrics.output_tokens || 0)} Token`;
  return box;
}

function renderToolResearch(payload, persist) {
  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(createElement("span", "assistant-mark", "工"), createElement("strong", "", "动态工具 Agent"));
  article.append(meta);
  const observations = Array.isArray(payload.observations) ? payload.observations : [];
  if (observations.length) {
    const list = createElement("ol", "tool-observations");
    observations.forEach((item) => {
      const row = createElement("li");
      row.append(
        createElement("strong", "", item.tool_name || "unknown_tool"),
        createElement("small", "", ` · ${item.status || "unknown"} · ${item.trust || "unclassified"}`),
        createElement("p", "", item.purpose || ""),
      );
      list.append(row);
    });
    article.append(list);
  }
  const text = payload.status === "approval_required"
    ? "敏感工具已暂停，等待你的批准或拒绝。"
    : payload.final_summary || "工具研究已完成。";
  article.append(createElement("p", payload.status === "approval_required" ? "insufficient" : "", text));
  elements.messages.append(article);
  if (persist) saveHistoryItem({ role: "assistant", text, status: payload.status });
  if (payload.status === "approval_required" && payload.pending_approval) {
    openToolApproval(payload.pending_approval);
  } else {
    state.pendingTool = null;
    showNotice("动态工具研究已完成；只有 citation_evidence 可作为论文引用依据。", "success");
  }
}

function openToolApproval(pending) {
  state.pendingTool = pending;
  elements.toolApprovalDetails.replaceChildren();
  appendDetail(elements.toolApprovalDetails, "工具", pending.tool_name);
  appendDetail(elements.toolApprovalDetails, "用途", pending.purpose);
  appendDetail(elements.toolApprovalDetails, "参数指纹", pending.arguments_sha256);
  appendDetail(elements.toolApprovalDetails, "有效期", new Date(pending.expires_at_epoch * 1000).toLocaleString());
  elements.toolApprovalDialog.showModal();
}

async function resolveToolApproval(approved) {
  if (!state.pendingTool || state.busy) return;
  setBusy(true);
  elements.toolApprovalAccept.disabled = true;
  elements.toolApprovalReject.disabled = true;
  try {
    const payload = await request(API.toolApproval, {
      method: "POST",
      body: JSON.stringify({ approved }),
    });
    elements.toolApprovalDialog.close();
    renderToolResearch(payload, true);
  } catch (error) {
    showNotice(error.message, "error");
  } finally {
    elements.toolApprovalAccept.disabled = false;
    elements.toolApprovalReject.disabled = false;
    setBusy(false);
  }
}

async function openMemories() {
  elements.memoriesButton.disabled = true;
  elements.memoriesList.replaceChildren(createElement("p", "insufficient", "正在读取长期记忆…"));
  elements.memoriesDialog.showModal();
  try {
    const payload = await request(API.memories, { method: "GET" });
    const memories = Array.isArray(payload.memories) ? payload.memories : [];
    elements.memoriesList.replaceChildren();
    if (!memories.length) {
      elements.memoriesList.append(createElement("p", "insufficient", "当前没有有效的长期记忆。"));
      return;
    }
    memories.forEach((memory) => {
      const card = createElement("article", "memory-card");
      const header = createElement("header");
      header.append(
        createElement("strong", "", memoryKindLabel(memory.kind)),
        createElement("small", "", `v${memory.version || 1}`),
      );
      card.append(
        header,
        createElement("p", "", memory.content || ""),
        createElement(
          "small",
          "",
          `更新：${formatMemoryTime(memory.updated_at)} · 来源：${(memory.source_chunk_ids || []).length} 条`,
        ),
      );
      elements.memoriesList.append(card);
    });
  } catch (error) {
    elements.memoriesList.replaceChildren(createElement("p", "insufficient", error.message));
  } finally {
    elements.memoriesButton.disabled = false;
  }
}

function memoryKindLabel(kind) {
  return {
    preference: "研究偏好",
    project_context: "项目背景",
    confirmed_conclusion: "确认结论",
  }[kind] || "长期记忆";
}

function formatMemoryTime(value) {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "未知" : parsed.toLocaleString();
}

function appendUserMessage(text, persist) {
  elements.messages.append(createElement("article", "message message-user", text));
  if (persist) saveHistoryItem({ role: "user", text, status: "sent" });
}

function appendErrorMessage(text, persist) {
  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(createElement("span", "assistant-mark", "!"), createElement("strong", "", "研究未完成"));
  const message = createElement("p", "insufficient", text);
  article.append(meta, message);
  elements.messages.append(article);
  showNotice("本次请求没有生成可验证答案，你可以修改问题后重试。", "error");
  if (persist) saveHistoryItem({ role: "assistant", text, status: "error" });
}

function normalizePayload(payload) {
  const answer = payload && typeof payload.answer === "object" ? payload.answer : payload;
  const trace = payload && typeof payload.trace === "object"
    ? payload.trace
    : payload && typeof payload.retrieval === "object"
      ? payload.retrieval
      : {};
  const context = payload && typeof payload.context === "object" ? payload.context : {};
  const claims = Array.isArray(answer.claims) ? answer.claims : [];
  const citations = Array.isArray(payload.sources)
    ? payload.sources
    : Array.isArray(answer.citations)
      ? answer.citations
      : [];
  return {
    status: answer.status || "answered",
    answerText: answer.answer_markdown || answer.answer || "",
    claims,
    citations,
    trace,
    context,
    generation: payload.generation || answer,
  };
}

function renderAnswer(payload, persist) {
  const normalized = normalizePayload(payload);
  state.citations.clear();
  normalized.citations.forEach((citation) => {
    if (citation && citation.citation_id) state.citations.set(citation.citation_id, citation);
  });

  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(
    createElement("span", "assistant-mark", "研"),
    createElement("strong", "", "论文研究 Agent"),
    createElement("small", "source-note source-note-rag", "本地论文库依据"),
  );
  article.append(meta);

  if (normalized.status === "insufficient_evidence") {
    const text = normalized.answerText || "当前检索上下文没有足够证据，无法可靠回答该问题。";
    article.append(createElement("p", "insufficient", text));
    showNotice("证据不足：系统没有调用猜测性答案补全。", "warning");
    if (persist) saveHistoryItem({ role: "assistant", text, status: "insufficient_evidence" });
  } else if (normalized.claims.length) {
    const list = createElement("ol", "claim-list");
    normalized.claims.forEach((claim) => list.append(renderClaim(claim)));
    article.append(list);
    const plainText = normalized.claims.map((claim) => String(claim.text || "")).join("\n\n");
    if (persist) saveHistoryItem({ role: "assistant", text: plainText, status: "answered" });
  } else {
    const text = normalized.answerText || "回答已完成，但没有返回可展示的 claims。";
    article.append(createElement("p", "insufficient", text));
    if (persist) saveHistoryItem({ role: "assistant", text, status: "answered" });
  }

  elements.messages.append(article);
  renderInspector(normalized);
  if (normalized.trace.degraded || normalized.trace.degraded_reason) {
    showNotice("英文查询改写已降级，本次回答使用可用的中文混合检索结果。", "warning");
  } else if (normalized.status === "answered") {
    showNotice("回答已通过本地引用白名单验证。", "success");
  }
}

function renderClaim(claim) {
  const item = createElement("li", "claim");
  item.append(renderCompactText(claim.text || ""));
  const ids = Array.isArray(claim.citation_ids) ? claim.citation_ids : [];
  if (ids.length) {
    const buttons = createElement("div", "citation-buttons");
    ids.forEach((id) => {
      const button = createElement("button", "citation-button", `[${id}]`);
      button.type = "button";
      button.setAttribute("aria-label", `查看引用 ${id}`);
      button.addEventListener("click", () => openEvidence(id));
      buttons.append(button);
    });
    item.append(buttons);
  }
  return item;
}

function renderCompactText(value) {
  const container = createElement("div", "answer-copy");
  const lines = String(value).replace(/\r\n?/g, "\n").split("\n");
  let list = null;
  let listType = "";

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      list = null;
      listType = "";
      return;
    }
    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    const nextListType = ordered ? "ol" : unordered ? "ul" : "";
    if (nextListType) {
      if (!list || listType !== nextListType) {
        list = document.createElement(nextListType);
        listType = nextListType;
        container.append(list);
      }
      list.append(createElement("li", "", (ordered || unordered)[1]));
      return;
    }
    list = null;
    listType = "";
    container.append(createElement("p", "", trimmed));
  });

  if (!container.childElementCount) container.append(createElement("p", "", "—"));
  return container;
}

function renderInspector(normalized) {
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  elements.routeMetrics.replaceChildren();
  elements.tokenMetrics.replaceChildren();
  elements.generationMetrics.replaceChildren();
  elements.citationMap.replaceChildren();
  const trace = normalized.trace || {};
  const context = normalized.context || {};
  const generation = normalized.generation || {};
  appendMetric(elements.routeMetrics, "原始问题", trace.original_question || "—");
  appendMetric(elements.routeMetrics, "解析后问题", trace.resolved_question || trace.original_question || "—");
  appendMetric(elements.routeMetrics, "独立问题", trace.standalone_question || trace.resolved_question || "—");
  appendMetric(elements.routeMetrics, "中文检索式", trace.chinese_query || trace.resolved_question || "—");
  appendMetric(elements.routeMetrics, "英文检索式", trace.english_query || "未生成");
  appendMetric(elements.routeMetrics, "会话记忆命中", trace.conversation_memory_hit_count ?? 0);
  appendMetric(elements.routeMetrics, "最近窗口轮次", trace.recent_context_turn_count ?? 0);
  appendMetric(elements.routeMetrics, "远距候选轮次", trace.recalled_candidate_count ?? 0);
  appendMetric(elements.routeMetrics, "语义解释来源", trace.interpretation_source || "未提供");
  appendMetric(elements.routeMetrics, "采用历史问题", Array.isArray(trace.selected_history_questions) && trace.selected_history_questions.length ? trace.selected_history_questions.join("；") : "未采用");
  appendMetric(elements.routeMetrics, "历史 Turn ID", Array.isArray(trace.selected_history_turn_ids) && trace.selected_history_turn_ids.length ? trace.selected_history_turn_ids.join("，") : "未采用");
  appendMetric(elements.routeMetrics, "历史相关度", Array.isArray(trace.selected_history_relevances) && trace.selected_history_relevances.length ? trace.selected_history_relevances.map((value) => Number(value).toFixed(2)).join("，") : "未采用");
  appendMetric(elements.routeMetrics, "跨路由继承", trace.inherited_across_route ? "是" : "否");
  appendMetric(elements.routeMetrics, "改写置信度", Number.isFinite(Number(trace.rewrite_confidence)) ? Number(trace.rewrite_confidence).toFixed(2) : "—");
  appendMetric(elements.routeMetrics, "需要澄清", trace.needs_clarification ? "是" : "否");
  const capabilities = trace.capability_plan || {};
  appendMetric(elements.routeMetrics, "本地论文能力", capabilities.local_papers ? "已执行" : "未执行");
  appendMetric(elements.routeMetrics, "网页研究能力", capabilities.web_research ? "已执行" : "未执行");
  appendMetric(elements.routeMetrics, "动态工具能力", capabilities.dynamic_tools ? "已执行" : "未执行");
  appendMetric(elements.routeMetrics, "附件能力", capabilities.attachments ? "已执行" : "未执行");
  appendMetric(elements.routeMetrics, "查询改写", trace.rewrite_status || trace.rewrite?.status || "未提供");
  appendMetric(elements.routeMetrics, "检索模式", trace.degraded ? "中文降级" : "中英双路");
  appendMetric(elements.routeMetrics, "索引版本", trace.index_id || "本地冻结索引");
  appendMetric(elements.routeMetrics, "检索命中", Array.isArray(trace.hits) ? trace.hits.length : normalized.citations.length);
  appendMetric(elements.tokenMetrics, "已估算 Token", context.estimated_tokens ?? "—");
  appendMetric(elements.tokenMetrics, "上下文总预算", context.token_budget ?? "—");
  appendMetric(elements.tokenMetrics, "回答预留", context.output_reserve_tokens ?? "—");
  appendMetric(elements.tokenMetrics, "纳入记忆轮次", context.included_memory_turn_count ?? "—");
  appendMetric(elements.tokenMetrics, "省略记忆轮次", context.omitted_memory_turn_count ?? "—");
  appendMetric(elements.tokenMetrics, "纳入证据", context.included_evidence_count ?? normalized.citations.length);
  appendMetric(elements.tokenMetrics, "省略证据", context.omitted_evidence_count ?? "—");
  appendMetric(elements.tokenMetrics, "证据状态", context.evidence_insufficient ? "不足" : "可回答");
  appendMetric(elements.generationMetrics, "输入 Token", generation.input_tokens ?? "—");
  appendMetric(elements.generationMetrics, "输出 Token", generation.output_tokens ?? "—");
  appendMetric(elements.generationMetrics, "生成延迟", formatLatency(generation.latency_ms));
  appendMetric(elements.generationMetrics, "模型尝试", generation.attempts ?? "—");
  appendMetric(elements.generationMetrics, "回答审计", generation.audit_persisted ? "已落盘" : "未落盘");

  normalized.citations.forEach((citation) => {
    const button = createElement("button");
    button.type = "button";
    button.append(
      createElement("span", "", `[${citation.citation_id || "E?"}]`),
      createElement("span", "", `${citation.corpus_id || "论文"} · ${formatPages(citation)}`),
    );
    button.addEventListener("click", () => openEvidence(citation.citation_id));
    elements.citationMap.append(button);
  });
  if (!normalized.citations.length) {
    elements.citationMap.append(createElement("p", "insufficient", "本轮没有可用引用。"));
  }
}

function appendMetric(list, label, value) {
  const row = createElement("div");
  row.append(createElement("dt", "", label), createElement("dd", "", value));
  list.append(row);
}

function formatLatency(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)} ms` : "—";
}

function formatPages(citation) {
  const start = citation.page_start;
  const end = citation.page_end;
  if (!start) return "页码未知";
  return start === end || !end ? `第 ${start} 页` : `第 ${start}–${end} 页`;
}

function openEvidence(citationId) {
  const citation = state.citations.get(citationId);
  if (!citation) {
    showToast("当前页面没有这条引用的详细信息。重新提问可恢复引用详情。");
    return;
  }
  elements.evidenceContent.replaceChildren();
  const details = createElement("dl");
  appendDetail(details, "引用编号", citation.citation_id);
  appendDetail(details, "论文", citation.title || citation.corpus_id || "未命名论文");
  appendDetail(details, "语料编号", citation.corpus_id || "—");
  appendDetail(details, "页码", formatPages(citation));
  appendDetail(details, "证据类型", citation.evidence_type === "figure_summary" ? "图表语义摘要" : "正文");
  appendDetail(details, "版权边界", citation.storage_class || "已在服务端校验");
  elements.evidenceContent.append(details);
  if (citation.official_url && isSafeHttpUrl(citation.official_url)) {
    const link = createElement("a", "", "打开论文官方页面");
    link.href = citation.official_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    elements.evidenceContent.append(link);
  }
  const excerpt = citation.excerpt || citation.evidence_excerpt || citation.preview;
  if (excerpt) {
    const preview = createElement("p", "evidence-preview", excerpt);
    preview.setAttribute("aria-label", "服务端提供的安全证据预览");
    elements.evidenceContent.append(preview);
  }
  if (!elements.evidenceDialog.open) elements.evidenceDialog.showModal();
}

function appendDetail(list, label, value) {
  list.append(createElement("dt", "", label), createElement("dd", "", value || "—"));
}

function isSafeHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch (_error) {
    return false;
  }
}

function showNotice(message, tone) {
  elements.notice.textContent = message;
  elements.notice.dataset.tone = tone;
  elements.notice.hidden = false;
}

function hideNotice() {
  elements.notice.hidden = true;
  elements.notice.textContent = "";
  delete elements.notice.dataset.tone;
}

function showToast(message) {
  const toast = createElement("div", "toast", message);
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function openInspector() {
  elements.inspector.classList.add("is-open");
  elements.inspectorToggle.setAttribute("aria-expanded", "true");
  elements.inspectorScrim.hidden = false;
  elements.inspectorClose.focus();
}

function closeInspector() {
  elements.inspector.classList.remove("is-open");
  elements.inspectorToggle.setAttribute("aria-expanded", "false");
  elements.inspectorScrim.hidden = true;
}

async function startNewConversation() {
  if (state.busy) return;
  elements.newConversation.disabled = true;
  try {
    archiveCurrentDialogue();
    await request(API.conversation, { method: "DELETE" });
    clearLocalDialogue();
    renderHistoryList();
    resetWorkspace();
    showToast("已开始新对话，上一轮短期记忆不再参与检索。 ");
  } catch (error) {
    if (error.status === 401) setAuthenticated(false);
    else showNotice(error.message, "error");
  } finally {
    elements.newConversation.disabled = false;
  }
}

function resetWorkspace() {
  state.citations.clear();
  elements.messages.replaceChildren(elements.emptyState);
  elements.emptyState.hidden = false;
  elements.inspectorEmpty.hidden = false;
  elements.inspectorContent.hidden = true;
  elements.conversationLabel.textContent = "新研究会话";
  hideNotice();
  closeInspector();
  elements.question.focus();
}

function saveHistoryItem(item) {
  state.history.push(item);
  state.history = state.history.slice(-24);
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state.history));
    renderHistoryList();
  } catch (_error) {
    // Browsers may deny storage in private contexts; the live view remains functional.
  }
}

function restoreHistory() {
  if (state.history.length) return;
  try {
    const stored = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "[]");
    renderHistoryList();
    if (!Array.isArray(stored) || !stored.length) return;
    elements.emptyState.hidden = true;
    stored.slice(-24).forEach((item) => {
      if (!item || typeof item.text !== "string") return;
      if (item.role === "user") appendUserMessage(item.text, false);
      else appendRestoredAssistant(item.text, item.status);
      state.history.push({ role: item.role, text: item.text, status: item.status });
    });
    elements.conversationLabel.textContent = "本标签页中的研究会话";
  } catch (_error) {
    clearLocalDialogue();
  }
}

function archiveCurrentDialogue() {
  if (!state.history.length) return;
  let archives = [];
  try {
    archives = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
    if (!Array.isArray(archives)) archives = [];
  } catch (_error) {
    archives = [];
  }
  const firstQuestion = state.history.find((item) => item.role === "user")?.text || "未命名对话";
  archives.unshift({ id: crypto.randomUUID(), title: firstQuestion, createdAt: Date.now(), items: state.history });
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(archives.slice(0, 20)));
  } catch (_error) {
    // The live conversation still works when storage is unavailable.
  }
  renderHistoryList();
}

function renderHistoryList() {
  let archives = [];
  try {
    archives = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
  } catch (_error) {
    archives = [];
  }
  elements.conversationHistory.replaceChildren();
  const currentQuestion = state.history.find((item) => item.role === "user")?.text;
  if (currentQuestion) {
    const current = createElement("button", "history-card history-card-current");
    current.type = "button";
    current.append(
      createElement("strong", "", currentQuestion),
      createElement("small", "", "当前对话"),
    );
    current.addEventListener("click", showCurrentDialogue);
    elements.conversationHistory.append(current);
  }
  if ((!Array.isArray(archives) || !archives.length) && !currentQuestion) {
    elements.conversationHistory.append(createElement("p", "history-empty", "开始新对话后，上一轮会保留在这里。"));
    return;
  }
  archives.forEach((archive) => {
    const button = createElement("button", "history-card");
    button.type = "button";
    button.append(
      createElement("strong", "", archive.title || "未命名对话"),
      createElement("small", "", new Date(archive.createdAt).toLocaleString()),
    );
    button.addEventListener("click", () => showArchivedDialogue(archive));
    elements.conversationHistory.append(button);
  });
}

function showCurrentDialogue() {
  elements.messages.replaceChildren();
  state.history.forEach((item) => {
    if (item.role === "user") appendUserMessage(item.text, false);
    else appendRestoredAssistant(item.text, item.status);
  });
  state.viewingArchive = false;
  elements.conversationLabel.textContent = "当前对话";
  hideNotice();
  scrollMessages();
}

function showArchivedDialogue(archive) {
  const items = Array.isArray(archive.items) ? archive.items : [];
  elements.messages.replaceChildren();
  items.forEach((item) => {
    if (item.role === "user") appendUserMessage(item.text, false);
    else appendRestoredAssistant(item.text, item.status);
  });
  state.viewingArchive = true;
  elements.conversationLabel.textContent = "正在查看历史对话";
  showNotice("这是历史记录；继续提问时会返回当前对话。", "warning");
  scrollMessages();
}

function appendRestoredAssistant(text, status) {
  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(createElement("span", "assistant-mark", "研"), createElement("strong", "", "论文研究 Agent"));
  article.append(meta, status === "error" ? createElement("p", "insufficient", text) : renderCompactText(text));
  elements.messages.append(article);
}

function clearLocalDialogue() {
  state.history = [];
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch (_error) {
    // No local state remains in memory even when storage APIs are unavailable.
  }
}

function scrollMessages() {
  window.requestAnimationFrame(() => elements.messages.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
}

elements.loginForm.addEventListener("submit", handleLogin);
elements.logout.addEventListener("click", handleLogout);
elements.newConversation.addEventListener("click", startNewConversation);
elements.memoriesButton.addEventListener("click", openMemories);
elements.askForm.addEventListener("submit", handleAsk);
elements.toolMode.addEventListener("change", syncRagControls);
elements.fileButton.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => uploadSelectedFile(elements.fileInput.files[0]));
elements.askForm.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.askForm.classList.add("is-dragging");
});
elements.askForm.addEventListener("dragleave", () => elements.askForm.classList.remove("is-dragging"));
elements.askForm.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.askForm.classList.remove("is-dragging");
  uploadSelectedFile(event.dataTransfer.files[0]);
});
elements.question.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    elements.askForm.requestSubmit();
  }
});
elements.inspectorToggle.addEventListener("click", openInspector);
elements.inspectorClose.addEventListener("click", closeInspector);
elements.inspectorScrim.addEventListener("click", closeInspector);
elements.evidenceClose.addEventListener("click", () => elements.evidenceDialog.close());
elements.evidenceDialog.addEventListener("click", (event) => {
  if (event.target === elements.evidenceDialog) elements.evidenceDialog.close();
});
elements.toolApprovalAccept.addEventListener("click", () => resolveToolApproval(true));
elements.toolApprovalReject.addEventListener("click", () => resolveToolApproval(false));
elements.memoriesClose.addEventListener("click", () => elements.memoriesDialog.close());
elements.memoriesDialog.addEventListener("click", (event) => {
  if (event.target === elements.memoriesDialog) elements.memoriesDialog.close();
});

checkSession();
