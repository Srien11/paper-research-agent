"use strict";

const API = Object.freeze({
  session: "api/session",
  login: "api/login",
  logout: "api/logout",
  conversation: "api/conversation",
  conversations: "api/conversations",
  activateConversation: (conversationId) => `api/conversations/${encodeURIComponent(conversationId)}/activate`,
  agentRuns: "api/agent/runs",
  agentApproval: (requestId) => `api/agent/runs/${encodeURIComponent(requestId)}/approval`,
  agentPlan: (requestId) => `api/agent/runs/${encodeURIComponent(requestId)}/plan`,
  agentControl: (requestId) => `api/agent/runs/${encodeURIComponent(requestId)}/control`,
  agentExplanation: (requestId, taskId) => `api/agent/runs/${encodeURIComponent(requestId)}/tasks/${encodeURIComponent(taskId)}/explanation`,
  files: "api/files",
  fileDownload: (attachmentId) => `api/files/${encodeURIComponent(attachmentId)}/download`,
  memories: "api/memories",
});

const STORAGE_KEY = "paper-research.current-dialogue.v1";
const HISTORY_KEY = "paper-research.dialogue-history.v1";
const PENDING_REQUEST_KEY = "paper-research.pending-request.v1";
const state = {
  busy: false,
  citations: new Map(),
  history: [],
  phaseTimer: null,
  pendingTool: null,
  pendingRequest: null,
  viewingArchive: false,
  attachments: [],
  activeRunId: null,
  activePlan: null,
  controlRevision: 0,
  planPollTimer: null,
  runStartedAt: null,
  runMetrics: {},
  currentConversationId: null,
  serverConversations: [],
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
  planControl: document.querySelector("#plan-control"),
  planControlStatus: document.querySelector("#plan-control-status"),
  planPause: document.querySelector("#plan-pause"),
  planResume: document.querySelector("#plan-resume"),
  planCancel: document.querySelector("#plan-cancel"),
  planRefresh: document.querySelector("#plan-refresh"),
  planObjective: document.querySelector("#plan-objective"),
  planSaveObjective: document.querySelector("#plan-save-objective"),
  planTaskList: document.querySelector("#plan-task-list"),
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

function createRequestId() {
  return `web_${crypto.randomUUID().replaceAll("-", "")}`;
}

function createPendingRequest(question) {
  return {
    requestId: createRequestId(),
    question,
    ragMode: selectedRagMode(),
    attachmentIds: state.attachments.map((item) => item.attachment_id),
    conversationId: state.currentConversationId,
  };
}

function savePendingRequest(pendingRequest) {
  state.pendingRequest = pendingRequest;
  try {
    localStorage.setItem(PENDING_REQUEST_KEY, JSON.stringify(pendingRequest));
  } catch (_error) {
    // In-memory idempotency remains available when persistent storage is denied.
  }
}

function loadPendingRequest() {
  if (state.pendingRequest) return state.pendingRequest;
  try {
    const value = JSON.parse(localStorage.getItem(PENDING_REQUEST_KEY) || "null");
    const valid = value
      && typeof value.question === "string"
      && /^[A-Za-z0-9_-]{16,128}$/.test(value.requestId)
      && ["disabled", "preferred", "required"].includes(value.ragMode)
      && Array.isArray(value.attachmentIds)
      && value.attachmentIds.length <= 5
      && value.attachmentIds.every((item) => /^[0-9a-f]{32}$/.test(item))
      && (value.conversationId === undefined || typeof value.conversationId === "string");
    if (!valid) return null;
    state.pendingRequest = value;
    return value;
  } catch (_error) {
    return null;
  }
}

function clearPendingRequest() {
  state.pendingRequest = null;
  state.pendingTool = null;
  try {
    localStorage.removeItem(PENDING_REQUEST_KEY);
  } catch (_error) {
    // In-memory state is already cleared.
  }
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

function setAuthenticated(authenticated, session = null) {
  elements.loginView.hidden = authenticated;
  elements.appView.hidden = !authenticated;
  if (authenticated) {
    if (session?.conversation_id) state.currentConversationId = session.conversation_id;
    restoreServerHistory();
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
    setAuthenticated(payload.authenticated !== false, payload);
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
    const session = await request(API.login, { method: "POST", body: JSON.stringify({ username, password }) });
    setAuthenticated(true, session);
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
  clearPendingRequest();
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
  const existingRequest = loadPendingRequest();
  if (existingRequest && existingRequest.question !== question) {
    showNotice("上一条请求尚未确认完成，请先输入原问题重试。", "warning");
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
  const pendingRequest = existingRequest || createPendingRequest(question);
  savePendingRequest(pendingRequest);
  elements.emptyState.hidden = true;
  if (!existingRequest) appendUserMessage(question, true);
  elements.question.value = "";
  setBusy(true);
  scrollMessages();
  try {
    await streamConversation(pendingRequest);
  } catch (error) {
    if (error.status === 401) {
      showToast("登录已过期，请重新登录。");
      setAuthenticated(false);
    } else if (error.status === 409) {
      appendErrorMessage("上一项研究仍在处理中，请稍后重试。", true);
    } else {
      if (error.status >= 400 && error.status < 500) clearPendingRequest();
      appendErrorMessage(error.message, true);
    }
  } finally {
    setBusy(false);
    await refreshPlanControl();
    await restoreServerHistory(false);
    scrollMessages();
    if (!elements.appView.hidden) elements.question.focus();
  }
}

async function streamConversation(pendingRequest, sourceNote = "") {
  activatePlanControl(pendingRequest.requestId);
  const responsePromise = fetch(API.agentRuns, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      request_id: pendingRequest.requestId,
      message: pendingRequest.question,
      attachment_ids: pendingRequest.attachmentIds,
      rag_mode: pendingRequest.ragMode,
    }),
  });
  const response = await responsePromise;
  return consumeAgentStream(response, pendingRequest, sourceNote);
}

async function consumeAgentStream(response, pendingRequest, sourceNote = "") {
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
  state.citations.clear();
  elements.routeMetrics.replaceChildren();
  elements.tokenMetrics.replaceChildren();
  elements.generationMetrics.replaceChildren();
  elements.citationMap.replaceChildren();
  renderRunMetrics();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let rawText = "";
  let route = "direct_chat";
  let finalStatus = "running";
  let waitingApproval = false;
  const outputAttachmentIds = [];
  let lastPaint = 0;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "run_reused") {
        meta.append(createElement("small", "source-note", "已复用原请求结果"));
      } else if (event.type === "task_started") {
        renderAgentInspectorEvent("节点执行中", event);
      } else if (event.type === "route_selected") {
        route = event.capability || route;
        const labels = {
          direct_chat: "普通交流",
          local_rag: "本地论文研究",
          dynamic_tools: "动态工具研究",
          attachment_qa: "附件问答",
          file_edit: "文件编辑",
        };
        meta.querySelector("strong").textContent = labels[route] || "研究助手";
        meta.querySelector(".assistant-mark").textContent = route === "attachment_qa" || route === "file_edit" ? "文" : "研";
        copy.textContent = route === "file_edit" ? "正在修改文件，完成后可下载新文件…" : "";
      } else if (event.type === "rag_result") {
        const sourceCount = Array.isArray(event.source_ids) ? event.source_ids.length : 0;
        meta.append(createElement("small", "source-note source-note-rag", `本地论文来源 ${sourceCount} 条`));
        renderAgentInspectorEvent("本地论文检索", event);
      } else if (event.type === "tool_result") {
        const toolNames = Array.isArray(event.tool_names) ? event.tool_names : [];
        meta.append(createElement("small", "source-note", toolNames.length ? `工具：${toolNames.join("、")}` : "动态工具已完成"));
        renderAgentInspectorEvent("动态工具", event);
      } else if (event.type === "attachment_result") {
        meta.append(createElement("small", "source-note", "已读取当前会话附件"));
        renderAgentInspectorEvent("附件问答", event);
      } else if (event.type === "file_result") {
        const ids = Array.isArray(event.output_attachment_ids) ? event.output_attachment_ids : [];
        ids.forEach((id) => {
          if (/^[0-9a-f]{32}$/.test(id) && !outputAttachmentIds.includes(id)) outputAttachmentIds.push(id);
        });
        renderAgentInspectorEvent("文件输出", event);
      } else if (event.type === "task_completed") {
        mergeRunMetrics(event.counts);
        renderRunMetrics(state.activePlan);
      } else if (event.type === "approval_required" && event.pending_approval) {
        waitingApproval = true;
        openToolApproval(event.pending_approval, pendingRequest.requestId);
        copy.textContent = "敏感工具已暂停，等待你的批准或拒绝。";
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
        finalStatus = event.status || "failed";
        stopPlanRefresh();
      } else if (event.type === "error") {
        finalStatus = "failed";
        stopPlanRefresh();
      }
    }
    if (done) break;
  }
  if (["completed", "failed", "cancelled", "conflict"].includes(finalStatus)) {
    clearPendingRequest();
  }
  if (finalStatus === "failed" || finalStatus === "cancelled" || finalStatus === "conflict") {
    article.remove();
    throw new Error("主 Agent 未能完成这次请求，请检查输入后重试。");
  }
  if (waitingApproval || finalStatus === "waiting_approval") {
    showNotice("敏感工具等待审批；批准或拒绝后将使用同一请求继续。", "warning");
    return;
  }
  if (finalStatus === "paused") {
    showNotice("运行已暂停，已完成步骤不会重跑。你可以编辑计划后继续。", "warning");
    await refreshPlanControl();
    return;
  }
  const editMode = route === "file_edit";
  const finalText = (editMode ? rawText : naturalText(rawText)).trim();
  if (editMode) {
    copy.textContent = "文件修改完成，可以下载新文件。";
  } else copy.replaceChildren(renderTextWithCitations(finalText));
  outputAttachmentIds.forEach((attachmentId) => {
    article.append(createServerDownloadButton(attachmentId));
  });
  saveHistoryItem({ role: "assistant", text: finalText, status: finalStatus });
}

function activatePlanControl(requestId) {
  stopPlanRefresh();
  state.activeRunId = requestId;
  state.activePlan = null;
  state.controlRevision = 0;
  state.runStartedAt = performance.now();
  state.runMetrics = {};
  elements.planControl.hidden = false;
  elements.planControlStatus.textContent = "启动中";
  elements.planTaskList.replaceChildren(createElement("p", "history-empty", "正在生成可编辑计划…"));
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  elements.routeMetrics.replaceChildren();
  appendMetric(elements.routeMetrics, "节点状态", "正在生成计划");
  renderRunMetrics();
  syncPlanButtons("running");
  schedulePlanRefresh(250);
}

function schedulePlanRefresh(delay = 800) {
  if (state.planPollTimer) window.clearTimeout(state.planPollTimer);
  state.planPollTimer = window.setTimeout(() => refreshPlanControl(), delay);
}

function stopPlanRefresh() {
  if (state.planPollTimer) window.clearTimeout(state.planPollTimer);
  state.planPollTimer = null;
}

async function refreshPlanControl() {
  if (!state.activeRunId) return;
  try {
    const plan = await request(API.agentPlan(state.activeRunId));
    state.activePlan = plan;
    state.controlRevision = plan.control.revision;
    renderPlanControl(plan);
    if (["running", "pause_requested", "resuming", "cancel_requested"].includes(plan.control.status)) {
      schedulePlanRefresh(700);
    } else {
      stopPlanRefresh();
    }
  } catch (error) {
    if (error.status === 404 || error.status === 409) {
      if (state.busy) schedulePlanRefresh(500);
      else {
        stopPlanRefresh();
        elements.planControlStatus.textContent = "未生成计划";
        elements.planTaskList.replaceChildren(createElement("p", "history-empty", "本次运行在计划生成前结束。"));
      }
      return;
    }
    showToast(error.message);
  }
}

function syncPlanButtons(status) {
  const paused = status === "paused";
  elements.planPause.disabled = !["running", "resuming"].includes(status);
  elements.planResume.disabled = !paused;
  elements.planCancel.disabled = ["cancelled", "completed", "failed"].includes(status);
  elements.planSaveObjective.disabled = !paused || !state.activePlan;
  elements.planObjective.disabled = !paused || !state.activePlan;
}

function planStatusLabel(status) {
  return ({
    running: "执行中",
    pause_requested: "正在暂停",
    paused: "已暂停",
    resuming: "正在继续",
    cancel_requested: "正在取消",
    cancelled: "已取消",
    completed: "已完成",
    failed: "失败",
    waiting_approval: "等待审批",
  })[status] || status;
}

function renderPlanControl(plan) {
  const status = plan.control.status;
  elements.planControlStatus.textContent = planStatusLabel(status);
  elements.planObjective.value = plan.objective || "";
  syncPlanButtons(status);
  elements.planTaskList.replaceChildren();
  renderPlanNodeMetrics(plan);
  renderRunMetrics(plan);
  if (!plan.tasks.length) {
    elements.planTaskList.append(createElement("p", "history-empty", "正在生成任务节点…"));
  }
  plan.tasks.forEach((task, index) => {
    const card = createElement("article", "plan-task");
    const header = createElement("header");
    header.append(
      createElement("strong", "", `${index + 1}. ${task.title}`),
      createElement("small", "", planStatusLabel(task.status)),
    );
    const reason = createElement("p", "", task.execution_reason);
    const usage = createElement(
      "small",
      "",
      `调用 ${task.call_count}${task.max_calls ? `/${task.max_calls}` : ""} · ${Number(task.elapsed_seconds || 0).toFixed(1)} 秒 · $${Number(task.cost_usd || 0).toFixed(4)}`,
    );
    const budgets = createElement("div", "plan-task-budget");
    const budgetFields = [
      ["max_seconds", "秒"],
      ["max_calls", "调用"],
      ["max_cost_usd", "美元"],
    ];
    const inputs = {};
    budgetFields.forEach(([field, label]) => {
      const wrapper = createElement("label", "", label);
      const input = createElement("input");
      input.type = "number";
      input.min = field === "max_cost_usd" ? "0" : "1";
      input.step = field === "max_cost_usd" ? "0.01" : "1";
      input.value = task[field] ?? "";
      input.disabled = status !== "paused" || task.status === "completed";
      inputs[field] = input;
      wrapper.append(input);
      budgets.append(wrapper);
    });
    const actions = createElement("div", "plan-task-actions");
    const explain = createElement("button", "button button-quiet", "为什么");
    explain.type = "button";
    explain.addEventListener("click", async () => {
      try {
        const payload = await request(API.agentExplanation(state.activeRunId, task.task_id));
        showToast(payload.explanation);
      } catch (error) {
        showToast(error.message);
      }
    });
    actions.append(explain);
    if (status === "paused" && task.status !== "completed") {
      const saveBudget = createElement("button", "button button-secondary", "保存预算");
      saveBudget.type = "button";
      saveBudget.addEventListener("click", () => updateTaskBudget(task.task_id, inputs));
      actions.append(saveBudget);
      if (!["skipped", "cancelled"].includes(task.status)) {
        const skip = createElement("button", "button button-quiet", "跳过");
        skip.type = "button";
        skip.addEventListener("click", () => editActivePlan({ skip_task_ids: [task.task_id] }));
        actions.append(skip);
      }
      if (task.status === "failed") {
        const retry = createElement("button", "button button-secondary", "只重试此步");
        retry.type = "button";
        retry.addEventListener("click", () => editActivePlan({ retry_task_ids: [task.task_id] }));
        actions.append(retry);
      }
      if (index > 0) {
        const up = createElement("button", "button button-quiet", "上移");
        up.type = "button";
        up.addEventListener("click", () => movePlanTask(index, -1));
        actions.append(up);
      }
      if (index < plan.tasks.length - 1) {
        const down = createElement("button", "button button-quiet", "下移");
        down.type = "button";
        down.addEventListener("click", () => movePlanTask(index, 1));
        actions.append(down);
      }
    }
    card.append(header, reason, usage, budgets, actions);
    elements.planTaskList.append(card);
  });
}

function renderPlanNodeMetrics(plan) {
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  elements.routeMetrics.replaceChildren();
  appendMetric(elements.routeMetrics, "运行状态", planStatusLabel(plan.control.status));
  plan.tasks.forEach((task, index) => {
    appendMetric(
      elements.routeMetrics,
      `${index + 1}. ${task.title}`,
      `${planStatusLabel(task.status)} · ${task.capability}`,
    );
  });
}

function mergeRunMetrics(counts) {
  if (!counts || typeof counts !== "object") return;
  [
    "elapsed_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_context_tokens",
    "token_budget",
    "output_reserve_tokens",
  ].forEach((key) => {
    const value = Number(counts[key]);
    if (Number.isFinite(value) && value >= 0) {
      if (["token_budget", "output_reserve_tokens"].includes(key)) {
        state.runMetrics[key] = Math.max(Number(state.runMetrics[key] || 0), value);
      } else state.runMetrics[key] = Number(state.runMetrics[key] || 0) + value;
    }
  });
  if (!state.runMetrics.total_tokens) {
    state.runMetrics.total_tokens = Number(state.runMetrics.input_tokens || 0)
      + Number(state.runMetrics.output_tokens || 0);
  }
}

function renderRunMetrics(plan = null) {
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  const metrics = state.runMetrics || {};
  const taskElapsedMs = plan && Array.isArray(plan.tasks)
    ? plan.tasks.reduce((total, task) => total + Number(task.elapsed_seconds || 0) * 1000, 0)
    : 0;
  const wallElapsedMs = state.runStartedAt === null ? 0 : performance.now() - state.runStartedAt;
  const elapsedMs = Math.max(Number(metrics.elapsed_ms || 0), taskElapsedMs, wallElapsedMs);
  elements.tokenMetrics.replaceChildren();
  appendMetric(elements.tokenMetrics, "已估算 Token", metrics.estimated_context_tokens ?? "—");
  appendMetric(elements.tokenMetrics, "上下文总预算", metrics.token_budget ?? "—");
  appendMetric(elements.tokenMetrics, "回答预留", metrics.output_reserve_tokens ?? "—");
  elements.generationMetrics.replaceChildren();
  appendMetric(elements.generationMetrics, "运行耗时", formatLatency(elapsedMs));
  appendMetric(elements.generationMetrics, "输入 Token", metrics.input_tokens ?? 0);
  appendMetric(elements.generationMetrics, "输出 Token", metrics.output_tokens ?? 0);
  appendMetric(elements.generationMetrics, "总 Token", metrics.total_tokens ?? 0);
}

async function sendRunControl(action) {
  if (!state.activeRunId) return;
  try {
    const control = await request(API.agentControl(state.activeRunId), {
      method: "POST",
      body: JSON.stringify({ action, expected_revision: state.controlRevision }),
    });
    state.controlRevision = control.revision;
    elements.planControlStatus.textContent = planStatusLabel(control.status);
    syncPlanButtons(control.status);
    schedulePlanRefresh(250);
  } catch (error) {
    showToast(error.message);
    refreshPlanControl();
  }
}

async function editActivePlan(changes) {
  if (!state.activeRunId || !state.activePlan) return;
  try {
    const plan = await request(API.agentPlan(state.activeRunId), {
      method: "PATCH",
      body: JSON.stringify({ expected_revision: state.activePlan.plan_revision, ...changes }),
    });
    state.activePlan = plan;
    renderPlanControl(plan);
    showToast("计划已更新，已完成结果保持不变。" );
  } catch (error) {
    showToast(error.message);
    refreshPlanControl();
  }
}

function updateTaskBudget(taskId, inputs) {
  const budget = {};
  Object.entries(inputs).forEach(([field, input]) => {
    if (input.value !== "") budget[field] = Number(input.value);
  });
  editActivePlan({ task_edits: [{ task_id: taskId, budget }] });
}

function movePlanTask(index, offset) {
  const ids = state.activePlan.tasks.map((task) => task.task_id);
  const target = index + offset;
  [ids[index], ids[target]] = [ids[target], ids[index]];
  editActivePlan({ ordered_task_ids: ids });
}

function renderAgentInspectorEvent(label, event) {
  elements.inspectorEmpty.hidden = true;
  elements.inspectorContent.hidden = false;
  appendMetric(elements.routeMetrics, label, `${event.counts?.source_count || 0} 条来源`);
}

function createServerDownloadButton(attachmentId) {
  const link = createElement("a", "download-file", "下载生成文件");
  link.href = API.fileDownload(attachmentId);
  link.setAttribute("download", "");
  return link;
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

function openToolApproval(pending, requestId) {
  state.pendingTool = { requestId, pending };
  elements.toolApprovalDetails.replaceChildren();
  appendDetail(elements.toolApprovalDetails, "工具", pending.tool_name);
  appendDetail(elements.toolApprovalDetails, "用途", pending.purpose);
  appendDetail(elements.toolApprovalDetails, "参数指纹", pending.arguments_sha256);
  appendDetail(elements.toolApprovalDetails, "有效期", new Date(pending.expires_at_epoch * 1000).toLocaleString());
  elements.toolApprovalDialog.showModal();
}

async function resolveToolApproval(approved) {
  if (!state.pendingTool || state.busy) return;
  const pendingTool = state.pendingTool;
  const pendingRequest = loadPendingRequest();
  if (!pendingRequest || pendingRequest.requestId !== pendingTool.requestId) {
    clearPendingRequest();
    elements.toolApprovalDialog.close();
    showNotice("审批对应的请求已失效，请重新发起。", "error");
    return;
  }
  setBusy(true);
  elements.toolApprovalAccept.disabled = true;
  elements.toolApprovalReject.disabled = true;
  try {
    const response = await fetch(API.agentApproval(state.pendingTool.requestId), {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
    });
    elements.toolApprovalDialog.close();
    await consumeAgentStream(response, pendingRequest, "审批恢复");
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
  if (loadPendingRequest()) {
    const retry = createElement("button", "download-file", "重试原请求");
    retry.type = "button";
    retry.addEventListener("click", retryPendingRequest);
    article.append(retry);
  }
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

async function restoreServerHistory(renderCurrent = true) {
  try {
    const payload = await request(API.conversations);
    state.currentConversationId = payload.current_conversation_id || state.currentConversationId;
    state.serverConversations = Array.isArray(payload.conversations) ? payload.conversations : [];
    const current = state.serverConversations.find(
      (item) => item.conversation_id === state.currentConversationId,
    );
    if (current) {
      state.history = current.messages.map((item) => ({
        role: item.role,
        text: item.text,
        status: item.status,
      }));
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state.history));
      } catch (_error) {
        // The server remains authoritative when browser caching is unavailable.
      }
      if (renderCurrent && !state.viewingArchive) showCurrentDialogue();
    } else if (renderCurrent && !state.viewingArchive) {
      state.history = [];
      renderDialogueMessages([], "新研究会话");
    }
    renderHistoryList();
    const pendingRequest = loadPendingRequest();
    if (pendingRequest) {
      elements.question.value = pendingRequest.question;
      showNotice(`已恢复未完成的原问题：“${pendingRequest.question}”。可直接点击发送重试。`, "warning");
    }
  } catch (_error) {
    restoreHistory();
  }
}

async function activatePersistedConversation(dialogue) {
  if (state.busy) return;
  try {
    const session = await request(API.activateConversation(dialogue.conversation_id), {
      method: "POST",
      body: "{}",
    });
    state.currentConversationId = session.conversation_id;
    state.history = dialogue.messages.map((item) => ({
      role: item.role,
      text: item.text,
      status: item.status,
    }));
    const pendingRequest = loadPendingRequest();
    if (pendingRequest?.conversationId && pendingRequest.conversationId !== session.conversation_id) {
      elements.question.value = "";
      hideNotice();
    } else if (pendingRequest) {
      elements.question.value = pendingRequest.question;
      showNotice(`已恢复未完成的原问题：“${pendingRequest.question}”。可直接点击发送重试。`, "warning");
    }
    state.viewingArchive = false;
    showCurrentDialogue();
    renderHistoryList();
  } catch (error) {
    showNotice(error.message, "error");
  }
}

async function startNewConversation() {
  if (state.busy) return;
  elements.newConversation.disabled = true;
  try {
    const session = await request(API.conversation, { method: "DELETE" });
    state.currentConversationId = session.conversation_id;
    clearPendingRequest();
    clearLocalDialogue();
    resetWorkspace();
    await restoreServerHistory(false);
    showToast("已开始新对话，旧对话已保存在左侧。 ");
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
  if (state.planPollTimer) window.clearTimeout(state.planPollTimer);
  state.planPollTimer = null;
  state.activeRunId = null;
  state.activePlan = null;
  state.controlRevision = 0;
  elements.planControl.hidden = true;
  elements.conversationLabel.textContent = "新研究会话";
  hideNotice();
  closeInspector();
  elements.question.focus();
}

function saveHistoryItem(item) {
  state.history.push(item);
  state.history = state.history.slice(-24);
  if (state.currentConversationId) {
    const now = new Date().toISOString();
    const title = state.history.find((entry) => entry.role === "user")?.text || "未命名对话";
    const persisted = {
      conversation_id: state.currentConversationId,
      title,
      created_at: now,
      updated_at: now,
      messages: state.history.map((entry) => ({ ...entry, created_at: now })),
    };
    const index = state.serverConversations.findIndex(
      (entry) => entry.conversation_id === state.currentConversationId,
    );
    if (index >= 0) state.serverConversations[index] = persisted;
    else state.serverConversations.unshift(persisted);
  }
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
  const archives = [...state.serverConversations].sort((left, right) => {
    if (left.conversation_id === state.currentConversationId) return -1;
    if (right.conversation_id === state.currentConversationId) return 1;
    return String(right.updated_at).localeCompare(String(left.updated_at));
  });
  elements.conversationHistory.replaceChildren();
  if (!archives.length) {
    elements.conversationHistory.append(createElement("p", "history-empty", "完成首轮提问后，对话会永久保存在这里。"));
    return;
  }
  archives.forEach((archive) => {
    const isCurrent = archive.conversation_id === state.currentConversationId;
    const button = createElement("button", `history-card${isCurrent ? " history-card-current" : ""}`);
    button.type = "button";
    button.append(
      createElement("strong", "", archive.title || "未命名对话"),
      createElement("small", "", isCurrent ? "当前对话" : new Date(archive.updated_at).toLocaleString()),
    );
    button.addEventListener("click", () => {
      if (isCurrent) showCurrentDialogue();
      else activatePersistedConversation(archive);
    });
    elements.conversationHistory.append(button);
  });
}

function showCurrentDialogue() {
  renderDialogueMessages(state.history, "当前对话");
  state.viewingArchive = false;
  hideNotice();
  scrollMessages();
}

function showArchivedDialogue(archive) {
  const items = Array.isArray(archive.messages) ? archive.messages : [];
  renderDialogueMessages(items, "正在查看历史对话");
  state.viewingArchive = true;
  showNotice("这是服务端持久化的历史记录。", "warning");
  scrollMessages();
}

function renderDialogueMessages(items, label) {
  elements.messages.replaceChildren();
  if (!items.length) {
    elements.messages.append(elements.emptyState);
    elements.emptyState.hidden = false;
    elements.conversationLabel.textContent = label;
    return;
  }
  elements.emptyState.hidden = true;
  items.forEach((item) => {
    if (item.role === "user") appendUserMessage(item.text, false);
    else appendRestoredAssistant(item.text, item.status);
  });
  elements.conversationLabel.textContent = label;
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

async function retryPendingRequest() {
  const pendingRequest = loadPendingRequest();
  if (!pendingRequest || state.busy) return;
  hideNotice();
  setBusy(true);
  try {
    await streamConversation(pendingRequest, "断线重试，复用原请求标识");
  } catch (error) {
    if (error.status >= 400 && error.status < 500 && error.status !== 409) {
      clearPendingRequest();
    }
    appendErrorMessage(error.message, true);
    } finally {
      setBusy(false);
      await refreshPlanControl();
      await restoreServerHistory(false);
      scrollMessages();
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
elements.planPause.addEventListener("click", () => sendRunControl("pause"));
elements.planResume.addEventListener("click", () => sendRunControl("resume"));
elements.planCancel.addEventListener("click", () => sendRunControl("cancel"));
elements.planRefresh.addEventListener("click", refreshPlanControl);
elements.planSaveObjective.addEventListener("click", () => {
  const objective = elements.planObjective.value.trim();
  if (objective) editActivePlan({ objective });
});

checkSession();
