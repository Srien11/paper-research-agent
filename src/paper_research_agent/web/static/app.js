"use strict";

const API = Object.freeze({
  session: "api/session",
  login: "api/login",
  logout: "api/logout",
  conversation: "api/conversation",
  ask: "api/ask",
  recommendations: "api/recommended-questions",
});

const STORAGE_KEY = "paper-research.current-dialogue.v1";
const state = {
  busy: false,
  citations: new Map(),
  history: [],
  phaseTimer: null,
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
  questionList: document.querySelector("#recommended-questions"),
  askForm: document.querySelector("#ask-form"),
  question: document.querySelector("#question"),
  askButton: document.querySelector("#ask-button"),
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
  toastRegion: document.querySelector("#toast-region"),
  conversationLabel: document.querySelector("#conversation-label"),
};

const fallbackQuestions = [
  { category: "RAG 可靠性", title: "降低检索增强生成的幻觉", prompt: "现有研究中，哪些方法能降低 RAG 系统的幻觉，并如何评估效果？" },
  { category: "评测方法", title: "大模型评测的可信度", prompt: "大语言模型评测中常见的可靠性威胁有哪些？" },
  { category: "检索基准", title: "BEIR 与 MTEB 的区别", prompt: "BEIR 和 MTEB 分别覆盖哪些任务，它们的设计目标有什么区别？" },
];

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
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
    loadRecommendedQuestions();
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

async function loadRecommendedQuestions() {
  let questions = fallbackQuestions;
  try {
    const payload = await request(API.recommendations, { method: "GET" });
    if (Array.isArray(payload) && payload.length) questions = payload;
    else if (Array.isArray(payload.questions) && payload.questions.length) questions = payload.questions;
  } catch (_error) {
    // The static fallback keeps the interface useful during API degradation.
  }
  renderRecommendedQuestions(questions);
}

function renderRecommendedQuestions(questions) {
  elements.questionList.replaceChildren();
  questions.slice(0, 8).forEach((item) => {
    const button = createElement("button", "question-card");
    button.type = "button";
    button.dataset.prompt = String(item.prompt || item.title || "");
    button.append(
      createElement("small", "", item.category || "推荐问题"),
      createElement("strong", "", item.title || item.prompt || "开始研究"),
    );
    button.addEventListener("click", () => {
      elements.question.value = button.dataset.prompt;
      elements.question.focus();
    });
    elements.questionList.append(button);
  });
}

function setBusy(busy) {
  state.busy = busy;
  elements.askButton.disabled = busy;
  elements.newConversation.disabled = busy;
  elements.question.disabled = busy;
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
  elements.emptyState.hidden = true;
  appendUserMessage(question, true);
  const loadingMessage = elements.loadingTemplate.content.cloneNode(true).firstElementChild;
  elements.messages.append(loadingMessage);
  elements.question.value = "";
  setBusy(true);
  scrollMessages();
  try {
    const payload = await request(API.ask, {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    loadingMessage.remove();
    renderAnswer(payload, true);
  } catch (error) {
    loadingMessage.remove();
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
  meta.append(createElement("span", "assistant-mark", "研"), createElement("strong", "", "论文研究 Agent"));
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
  item.append(createElement("p", "", claim.text || ""));
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
  appendMetric(elements.routeMetrics, "英文检索式", trace.english_query || "未生成");
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
  elements.evidenceDialog.showModal();
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
    await request(API.conversation, { method: "DELETE" });
    clearLocalDialogue();
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
  } catch (_error) {
    // Browsers may deny storage in private contexts; the live view remains functional.
  }
}

function restoreHistory() {
  if (state.history.length) return;
  try {
    const stored = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "[]");
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

function appendRestoredAssistant(text, status) {
  const article = createElement("article", "message message-assistant");
  const meta = createElement("div", "message-meta");
  meta.append(createElement("span", "assistant-mark", "研"), createElement("strong", "", "论文研究 Agent"));
  article.append(meta, createElement("p", status === "error" ? "insufficient" : "", text));
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
elements.askForm.addEventListener("submit", handleAsk);
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

checkSession();
