# 最小证据编译契约 v2 实施计划

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将比较证据编译改为“模型只提交最小事实单元、代码确定性投影、失败单元事务式重试”，并把编译失败与证据不足彻底分离。

**Architecture:** 比较任务使用只含 `requirement_id` 与事实字段的批次传输契约；批次返回后逐单元执行严格 schema、计划 ID、论文归属、事实意图、限定条件和可见 chunk 校验。合法单元立即提交并在重试中保持不变，代码从已提交单元生成完整 `EvidenceAssessment`；仍失败的单元标记为 `compiler_failed`，不得转换为空台账或 `insufficient_evidence`。

**Tech Stack:** Python 3.12、Pydantic v2、LangChain structured output、unittest/pytest、Ruff、mypy。

---

### Task 1: 建立最小模型契约与失败状态

**Files:**
- Modify: `src/paper_research_agent/agent/models.py`
- Test: `tests/agent/test_models.py`

**Step 1: 写失败测试**

验证 `EvidenceCellCompilation` 只接受 `requirement_id` 和最小事实数组，不包含 coverage、status、missing IDs、sufficiency、follow-up 或模型生成的 `fact_id`；验证 `compiler_failed` 是合法 assessment/termination 状态。

**Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_models -v`

Expected: FAIL，提示最小编译模型或新状态尚不存在。

**Step 3: 实现最小契约**

新增以下不可变模型：

```python
class EvidenceFactCompilation(FrozenContract):
    statement: str
    chunk_ids: tuple[str, ...]
    fact_requirement_ids: tuple[str, ...]
    qualifiers: tuple[EvidenceQualifier, ...] = ()

class EvidenceCellCompilation(FrozenContract):
    requirement_id: str
    facts: tuple[EvidenceFactCompilation, ...] = ()

class EvidenceCompilationBatch(FrozenContract):
    cells: tuple[dict[str, object], ...]
```

批次只负责稳定接收每个原始单元，严格校验由逐单元提交器完成。扩展审计记录失败 requirement ID，并加入 `compiler_failed` 状态与终止原因。

**Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_models -v`

Expected: PASS。

### Task 2: 实现确定性单元校验与台账投影

**Files:**
- Modify: `src/paper_research_agent/agent/coverage.py`
- Test: `tests/agent/test_coverage.py`

**Step 1: 写失败测试**

覆盖合法单元投影、确定性 `fact_id`、缺失事实 ID、coverage chunk 投影、限定条件、跨论文 chunk、未知 requirement/fact ID、重复单元，以及“部分单元失败仍保留其他单元”。

**Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_coverage -v`

Expected: FAIL，提示投影与单元校验函数尚不存在。

**Step 3: 实现提交器与投影器**

提供清晰边界：

```python
def validate_evidence_compilation_cell(...) -> EvidenceCellCompilation: ...
def project_evidence_compilation(...) -> EvidenceAssessment: ...
```

提交器只接受当前计划中的 requirement/fact ID，以及该 requirement 对应论文中本轮编译器可见的 chunk；投影器按计划顺序生成所有 ledger cell、coverage、missing IDs、status、sufficiency 和确定性 follow-up。存在重试后仍失败的单元时，assessment 使用 `compiler_failed` 且保留所有已提交事实。

**Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_coverage -v`

Expected: PASS。

### Task 3: 切换比较 reasoner 并实现事务式重试

**Files:**
- Modify: `src/paper_research_agent/agent/reasoner.py`
- Modify: `src/paper_research_agent/agent/graph.py`
- Test: `tests/agent/test_reasoner.py`
- Test: `tests/agent/test_graph.py`

**Step 1: 写失败测试**

验证比较模型 schema 不再包含派生字段；第一次合法单元会跨重试保留；第二次请求只包含失败 requirement 及稳定错误码；两次失败返回 `compiler_failed`；图在该状态下以独立终止原因结束。保留 direct 任务原行为。

**Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_reasoner tests.agent.test_graph -v`

Expected: FAIL，比较 reasoner 仍绑定完整 `EvidenceAssessment`。

**Step 3: 实现比较编译流程**

初始化 direct assessment 与 comparison batch 两个 structured model。比较路径首轮请求全部单元，逐单元提交后冻结成功结果；若存在失败，第二轮 payload 只含失败单元、允许 ID、可见 chunk 和逐单元错误码。最终完全由投影器构造 assessment，不调用空台账修复器。

**Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m unittest tests.agent.test_reasoner tests.agent.test_graph -v`

Expected: PASS。

### Task 4: 升级安全轨迹与评测指标

**Files:**
- Modify: `src/paper_research_agent/evaluation/comparison_end_to_end.py`
- Modify: `scripts/evaluate_comparison_end_to_end.py`
- Modify: `src/paper_research_agent/web/runtime.py`
- Test: `tests/evaluation/test_comparison_end_to_end.py`
- Test: `tests/web/test_runtime.py`

**Step 1: 写失败测试**

验证轨迹记录提交/失败单元数、exact gold chunk 命中、同论文替代 chunk 命中、最终保留事实数，并确保不记录证据正文或原始模型响应。

**Step 2: 运行测试确认失败**

Run: `.venv/Scripts/pytest.exe tests/evaluation/test_comparison_end_to_end.py -q && .venv/Scripts/python.exe -m unittest tests.web.test_runtime -v`

Expected: FAIL，新指标尚未投影。

**Step 3: 实现指标投影**

从安全审计与现有 lineage 数据确定性计算单元失败计数、保留率和 exact/alternative chunk 指标；运行产物继续留在 Git 之外。

**Step 4: 运行测试确认通过**

Run: `.venv/Scripts/pytest.exe tests/evaluation/test_comparison_end_to_end.py -q && .venv/Scripts/python.exe -m unittest tests.web.test_runtime -v`

Expected: PASS。

### Task 5: 全量回归与真实 smoke

**Files:**
- Modify only if failures expose an implementation defect.

**Step 1: 运行本地全量测试**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

Expected: 全部 PASS。

**Step 2: 运行函数式评测测试**

Run: `.venv/Scripts/pytest.exe tests/evaluation/test_comparison_end_to_end.py -q`

Expected: 全部 PASS。

**Step 3: 运行静态检查**

Run: `.venv/Scripts/ruff.exe check src tests scripts/evaluate_comparison_end_to_end.py`

Run: `.venv/Scripts/mypy.exe --python-version 3.12 src/paper_research_agent`

Expected: 全部 PASS。

**Step 4: 在凭据与冻结语料可用时运行 5 题 smoke**

复用现有 `comparison-end-to-end-compiler-audit-smoke5-v1` 配置生成新运行文件，确认不存在 schema 失败导致的整题 0 事实，且编译失败与证据不足可区分。

**Step 5: 在 smoke 通过后运行 30 题评测**

比较必要事实召回、完整率、检索纯度、引用正确率和超时率；不提交运行 JSON、论文正文、索引或密钥。

