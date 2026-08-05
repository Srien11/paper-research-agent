# Unified Agent Mode Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make one switch mean “RAG-only”; when it is off, provide normal conversation with optional tools and reliable approval continuation.

**Architecture:** Keep the existing strict RAG `/ask` endpoint and dynamic Agent `/tools/run` endpoint, but reverse and rename the UI switch so its product meaning is unambiguous. Strengthen the Agent router so ordinary conversation finishes without tools, and enforce explicit user intent before any approval-gated workspace write. Preserve the checkpoint-based approval resume path and expose successful execution clearly in the UI.

**Tech Stack:** Vanilla JavaScript/CSS, FastAPI, LangGraph, Pydantic, pytest.

---

### Task 1: Correct the mode switch contract

**Files:**
- Modify: `src/paper_research_agent/web/static/index.html`
- Modify: `src/paper_research_agent/web/static/app.js`
- Test: `tests/web/test_static.py`

1. Add failing assertions for the RAG-only label and routing direction.
2. Route checked mode to `/ask`; route unchecked mode to `/tools/run`.
3. Remove the client-only greeting shortcut so normal conversation is an Agent capability.
4. Run the static tests.

### Task 2: Prevent accidental sensitive-tool requests

**Files:**
- Modify: `src/paper_research_agent/agent/dynamic/router.py`
- Modify: `src/paper_research_agent/agent/dynamic/graph.py`
- Test: `tests/agent/test_dynamic_tools.py`

1. Add a failing test where a router selects a write tool for “你好”.
2. Add an explicit-write-intent guard for note/report writes.
3. Clarify the router system contract: greetings and ordinary conversation finish directly; tools are optional.
4. Run the dynamic Agent tests.

### Task 3: Make approval completion visible

**Files:**
- Modify: `src/paper_research_agent/web/static/app.js`
- Test: `tests/web/test_static.py`

1. Render approved tool status, purpose, and the final response in one continued Agent message.
2. Distinguish approval completion from denial.
3. Run Web and Agent test suites.
