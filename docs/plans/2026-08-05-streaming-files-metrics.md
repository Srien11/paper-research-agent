# Streaming Chat, Files, and Metrics Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add safe research-file uploads, real streaming chat, plain-language output, and a compact post-response usage card.

**Architecture:** Add authenticated raw-body upload endpoints with bounded per-session storage under `data/runtime/uploads`, avoiding multipart parser dependencies. Add an NDJSON streaming chat endpoint that proxies provider deltas and finishes with trusted timing/token metrics. The browser streams text into a plain-text answer node and renders an immutable metrics card after completion.

**Tech Stack:** FastAPI, httpx streaming, vanilla JavaScript, Nginx, pytest.

---

### Task 1: Safe file upload boundary

**Files:** Create `web/files.py`; modify `web/app.py`, `web/models.py`, `deploy/nginx-paper-research-locations.conf`; test `tests/web/test_files.py`, `tests/web/test_app.py`.

1. Test filename, type, size, session isolation, and path traversal rejection.
2. Implement raw-body upload limited to 10 MiB and allow PDF/TXT/MD/CSV/JSON.
3. Return opaque attachment IDs; never expose server paths.
4. Raise Nginx request limit to 12 MiB only for the authenticated upload route.

### Task 2: Provider streaming and metrics

**Files:** Modify `web/chat_runtime.py`, `web/app.py`; test `tests/web/test_chat_runtime.py`.

1. Test provider SSE delta parsing and usage extraction.
2. Emit NDJSON `delta`, then `done` with elapsed milliseconds, first-token wait, and token counts.
3. Include extracted attachment text in the user message with strict size budgets.

### Task 3: Streaming interface and natural-language formatting

**Files:** Modify `web/static/index.html`, `app.js`, `app.css`; test `tests/web/test_static.py`.

1. Add attachment picker, removable file chips, and upload progress.
2. Stream escaped text with no Markdown rendering.
3. Add a compact metrics card after the final token.
4. Keep RAG-only mode on the existing verified non-streaming endpoint.
