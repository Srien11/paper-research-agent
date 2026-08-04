# Agent Runtime 闭环实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将现有只读研究工具和 LangGraph 编排接入可运行、可恢复、可审计的本地问答 Runtime，并继续复用严格引用验证。

**Architecture:** 新增框架边界清晰的 `ResearchAgentRuntime`，负责调用已编译 Graph、映射 `session_id/thread_id`、施加总超时并把最终 State 重新校验为本地不可变证据。现有 `RAGRuntime` 仅在显式启用时走 Agent 路径，Graph 证据继续进入原有 `assemble_context -> answer_context` 可信边界；默认路径保持兼容。

**Tech Stack:** Python 3.11+、Pydantic、LangChain、LangGraph、LangGraph SQLite Checkpointer、FastAPI、pytest

---

### Task 1: 固定工具策略与 Runtime 输出契约

**Files:**
- Create: `src/paper_research_agent/agent/policy.py`
- Create: `src/paper_research_agent/agent/runtime.py`
- Test: `tests/agent/test_runtime.py`

**Step 1: Write the failing test**

覆盖未知工具拒绝、调用预算拒绝、Graph 最终证据与不可变 chunk 再校验、总超时和 `thread_id` 传递。

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_runtime.py -q`

Expected: FAIL because runtime modules do not exist.

**Step 3: Write minimal implementation**

实现只允许 `search_corpus` 与 `get_evidence` 的冻结策略；Runtime 用 `asyncio.timeout` 调用 `graph.ainvoke`，并将状态中的证据 ID、哈希、页码、版权分类与本地 chunk 逐项比对后生成 `ContextEvidence`。

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_runtime.py -q`

Expected: PASS.

### Task 2: 在 Graph 内执行工具预算

**Files:**
- Modify: `src/paper_research_agent/agent/graph.py`
- Modify: `tests/agent/test_graph.py`

**Step 1: Write the failing test**

验证 State 包含 `tool_call_count`，每次搜索/取证前检查工具白名单和总调用数，恢复后的调用数继续生效。

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_graph.py -q`

Expected: FAIL on missing policy and call count.

**Step 3: Write minimal implementation**

Graph 仅保留固定节点；Planner 不获得通用工具调用权。每一步最多产生一次搜索和一次显式 ID 取证，超过预算时关闭失败。

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_graph.py -q`

Expected: PASS.

### Task 3: 可选接入现有 Web Runtime

**Files:**
- Modify: `src/paper_research_agent/web/runtime.py`
- Modify: `tests/web/test_runtime.py`

**Step 1: Write the failing test**

验证启用 Agent 时使用 `session_id` 作为 Graph `thread_id`、不再执行原单次检索、证据仍经过原回答与引用验证；未启用时行为不变。

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/web/test_runtime.py -q`

Expected: FAIL on missing Agent dependency.

**Step 3: Write minimal implementation**

为依赖容器增加可选 `research_agent`。Agent 结果转成兼容的安全检索轨迹，完整 State、证据正文、提示词和本地路径不跨 Web 边界。

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/web/test_runtime.py -q`

Expected: PASS.

### Task 4: 生产 Planner 与 SQLite Checkpoint 工厂

**Files:**
- Create: `src/paper_research_agent/agent/factory.py`
- Modify: `src/paper_research_agent/agent/planner.py`
- Modify: `src/paper_research_agent/web/runtime.py`
- Modify: `pyproject.toml`
- Test: `tests/agent/test_factory.py`

**Step 1: Write the failing test**

验证只有 `PRA_RESEARCH_AGENT_ENABLED=true` 时才加载 Agent 可选依赖；Planner 使用现有 DashScope 模型与端点；checkpoint 文件固定在项目 runtime 数据目录。

**Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_factory.py -q`

Expected: FAIL on missing factory.

**Step 3: Write minimal implementation**

使用 `ChatOpenAI` 连接 DashScope OpenAI-compatible endpoint，用 `LangChainResearchPlanner` 生成严格 `ResearchPlan`；使用官方 SQLite checkpointer。API Key 仅从环境读取，不写入 State、日志或 checkpoint。

**Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/agent/test_factory.py -q`

Expected: PASS.

### Task 5: 文档与完整验证

**Files:**
- Modify: `README.md`
- Modify: `docs/系统架构.md`

**Step 1: Update documentation**

记录启用开关、两工具白名单、State/checkpoint 边界、总超时和未开放的敏感能力。

**Step 2: Run focused checks**

Run: `.\.venv\Scripts\python.exe -m ruff check src/paper_research_agent/agent tests/agent src/paper_research_agent/web/runtime.py tests/web/test_runtime.py`

Expected: PASS.

**Step 3: Run full regression suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

**Step 4: Check patch hygiene**

Run: `git diff --check`

Expected: no whitespace errors and no runtime databases, corpus text, keys, or local paths staged.
