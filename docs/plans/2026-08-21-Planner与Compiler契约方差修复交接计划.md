# Planner 与 Compiler 契约方差修复交接 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不修改固定 Top 4、检索器、重排器和 query rewrite 的前提下，消除 Planner 偶发整题契约失败和 Compiler 因限定词元数据缺项而丢弃已可见正确证据的问题，并用修正版金标完成可复现门禁。

**Architecture:** 保留模型负责语义选择，但把失败原因、重试指令和事务边界变为确定性控制逻辑。Planner 先获得字段级、安全且不含正文的失败码，并把对应修复要求反馈给第二次调用；Compiler 按事实而不是整格保留已验证结果，只重试尚未满足的事实，并显式传递每个事实要求的限定词类型。若聚焦重复门禁仍不能稳定通过，再把 Planner 的 ID、网格、步骤和语料范围改为本地确定性构造。

**Tech Stack:** Python 3.12、Pydantic v2、LangChain structured output、LangGraph、SQLite checkpointer、pytest、Ruff、mypy、现有私有固定 Top 4 端到端评测。

---

## 一、交接时现状

### 1. 仓库状态

- 仓库：`D:\agent-study\paper-research-agent`
- 分支：`main`
- 远端：`origin/main`
- 已推送基线提交：
  - `73562ce 功能: 建立事实级查询执行门控与排名诊断`
  - `3a17239 评测: 修正端到端金标范围门禁`
- 交接前全量验证：875 个测试、121 个子测试通过；2 条既有依赖告警，无失败。
- Ruff 与 mypy 通过。
- 私有金标、运行 JSON、SQLite checkpoint 和论文内容受 `.gitignore` 保护，不得强制提交。

### 2. 已经解决或被证伪的问题

旧金标将补充细节、重复事实和题目未要求内容统一列为必答事实，造成系统性假阴性：

- 30 题旧必答事实：140 条。
- v2 明确必答事实：93 条。
- 降为可选：47 条。
  - 题目未明确要求：11 条。
  - 与其他强制事实重复：16 条。
  - 仅作支撑细节：20 条。
- 旧诊断追踪的 8 条“缺失事实”全部被审定为可选。
- 五题固定 Top 4 复测中，已实际执行的 12 条明确必答事实全部进入 Top 4，`not_hydrated` 从旧金标的 5 条降为 v2 的 0 条。

因此当前没有证据支持“Planner 加细后仍未真正查询”或“明确必答事实稳定进不了 Top 4”。不要继续通过扩大 Top K、关闭 reranker、修改融合参数或 query rewrite 处理本轮问题。

### 3. 当前仍存在的两个生产问题

#### 问题 A：Planner 契约方差导致整题在检索前失败

固定 Top 4 五题首次运行：

- `CPG013` 抛出 `ComparisonTargetResolutionError`。
- `error_reason_code=planner_contract_invalid`。
- `error_stage=before_research_result`。
- 候选论文 Top 8 正确包含目标 `C013` 和 `T009`，因此候选检索不是根因。
- 使用相同候选集合和新 checkpoint 单题复跑后成功，最终选择 `C013,T009`，生成 3 个维度并完成 3/3 强制事实覆盖。

历史同题相关运行也表现出交替成功和失败，说明是结构化规划输出方差，不是稳定的数据或检索故障。

当前 Planner 一次模型输出必须同时满足：

- `task_type=comparison`；
- 至少两个且不重复的目标，目标 corpus 必须位于已解析候选集合；
- 一个或多个不重复维度；
- 完整的 target × dimension requirement 网格；
- 全局唯一 requirement/fact/step ID；
- 正确的 step corpus、target、dimension 和 fact 绑定；
- 总 step/fact 数不超过预算；
- 事实锚点来自用户问题目录。

任一条件失败都会进入同一 `except ValueError`。除锚点和事实预算外，多数错误被压缩为 `planner_contract_invalid`；失败的模型输出和字段路径没有进入安全审计。第二次调用只收到通用“违反契约”提示，没有收到第一次失败的具体字段或失败码，因此无法定向纠正。

相关代码：

- `src/paper_research_agent/agent/planner.py:43-49`
- `src/paper_research_agent/agent/planner.py:218-278`
- `src/paper_research_agent/agent/models.py:675-779`
- `scripts/evaluate_comparison_end_to_end.py:429-482`

#### 问题 B：Compiler 因限定词结构字段缺项丢弃 rank 2 证据

首次五题运行中的 `CPG020:claim_1`：

- 已检索、已水化、对 Compiler 可见。
- 最终 rank 2。
- 最早损失阶段为 `not_compiled`。
- 相关 requirement 为 `req_ineffective_claim`。
- 唯一计划事实 `fact_ineff_no_external` 要求限定词类型 `condition` 和 `method`。
- Compiler 第一次返回该格 2 个事实，失败码 `required_qualifier_missing`。
- Compiler 第二次仍返回 2 个事实，仍为相同失败码。
- 该格最终未提交，因此正确证据没有进入 generation input。

现有安全审计没有保留失败事实的 qualifier 数组，因此不能从历史记录断言具体缺了 `condition` 还是 `method`；可以确定至少有一个映射到该事实的输出对象未同时携带两种类型。

独立复跑时，同一语义事实生成的两个编译事实均带有 `condition + method`，该题 3/3 语义事实覆盖并正确引用。这证明证据和正文语义并未缺失，失败发生在结构化限定词元数据层。

当前 `_validate_compilation_batch` 以 requirement cell 为事务单位：格内任一事实违反 qualifier、映射或 chunk scope 约束，整个格都不进入 `committed`。虽然不同格之间已经可以保留有效 sibling，但同一格内的有效事实仍会与无效事实一起丢失。

相关代码：

- `src/paper_research_agent/agent/coverage.py:26-57`
- `src/paper_research_agent/agent/coverage.py:61-148`
- `src/paper_research_agent/agent/reasoner.py:232-335`
- `src/paper_research_agent/agent/reasoner.py:339-411`
- `src/paper_research_agent/agent/reasoner.py:414-484`

### 4. 证据边界

- v2 是 Agent 辅助范围审定，尚未完成人工双审或人工仲裁。
- 当前只完成五题固定 Top 4 门禁以及 `CPG013`、`CPG020` 单题复跑，没有使用 v2 完整重跑 30 题。
- “明确必答事实稳定进入 Top 4”目前只在已执行的五题样本中成立，不能外推为全量生产结论。
- 精确金标 chunk 排在 Top 4 外但同论文替代证据支持相同事实时，应计入语义覆盖，不得把唯一 chunk ID 当作唯一答案。

## 二、实施不变量

- 固定每步骤水化 Top 4，自适应水化关闭。
- 不修改 retriever、reranker、候选池、融合权重或 RAG query rewrite。
- 不增加 Planner 或 Compiler 的最大模型调用次数；仍最多两次。
- 不用金标文本、目标 chunk 或论文正文生成生产查询、规划或重试提示。
- 审计只记录安全 ID、枚举失败码、字段路径和计数，不记录问题正文、查询正文、事实正文或论文正文。
- 保持 fail closed：不能通过自动伪造 qualifier value、事实或引用来让验证通过。
- 允许保留已独立验证的事实；不允许一个无效事实连带删除同格中的有效事实。
- 保留旧 checkpoint 兼容性。
- 每个任务先写失败测试，聚焦测试通过后再提交。
- 不覆盖或强制添加受忽略的私有评测文件。

## 三、实施任务

### Task 1: 增加 Planner 字段级安全失败码

**Files:**
- Modify: `src/paper_research_agent/agent/models.py`
- Modify: `src/paper_research_agent/agent/planner.py`
- Modify: `tests/agent/test_models.py`
- Modify: `tests/agent/test_planner.py`

**Step 1: 写失败测试**

新增表驱动测试，分别构造下列无效计划并断言稳定失败码：

- `planner_task_type_invalid`
- `planner_target_count_invalid`
- `planner_target_duplicate`
- `planner_target_outside_candidate_set`
- `planner_dimension_invalid`
- `planner_grid_incomplete`
- `planner_requirement_reference_invalid`
- `planner_step_scope_invalid`
- `planner_step_grid_incomplete`
- `planner_id_duplicate`
- `planner_step_budget_invalid`
- `planner_fact_budget_invalid`
- `planner_anchor_selection_invalid`
- 无法分类的 Pydantic 错误回退为 `planner_schema_invalid`

失败码必须不包含模型生成正文、问题正文、论文 ID 以外的自由文本。

**Step 2: 运行测试确认失败**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\agent\test_models.py tests\agent\test_planner.py -q
```

Expected: FAIL；现有 `_planner_contract_reason` 只区分锚点、事实预算和通用错误。

**Step 3: 增加审计契约**

在 `models.py` 增加本地审计模型，并从 provider schema 排除：

```python
class PlannerAttemptAudit(FrozenContract):
    attempt: int = Field(ge=1, le=2)
    outcome: Literal["validated", "schema_invalid", "contract_invalid"]
    failure_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,95}$",
    )


class ResearchPlan(FrozenContract):
    # existing fields...
    planner_attempts: SkipJsonSchema[tuple[PlannerAttemptAudit, ...]] = ()
```

为 `PlannerAttemptAudit` 增加约束：成功时不得有失败码，失败时必须有失败码，attempt 必须从 1 连续递增。

**Step 4: 实现确定性错误分类**

将 `_planner_contract_reason` 改为检查 `ValidationError.errors()` 的 `msg/type/loc`，以及 Planner 自己抛出的稳定错误。只返回固定枚举，不拼接 `str(error)` 到外部诊断。

建议接口：

```python
def _planner_failure_code(error: ValueError) -> str:
    messages = (
        [str(item.get("msg", "")) for item in error.errors()]
        if isinstance(error, ValidationError)
        else [str(error)]
    )
    # ordered fragment-to-code mapping; final fallback is planner_schema_invalid
```

**Step 5: 运行聚焦测试**

Expected: PASS。

**Step 6: 提交**

```powershell
git add src/paper_research_agent/agent/models.py src/paper_research_agent/agent/planner.py tests/agent/test_models.py tests/agent/test_planner.py
git commit -m "诊断: 细分Planner结构化契约失败码"
```

### Task 2: 把第一次 Planner 失败精确反馈给第二次调用

**Files:**
- Modify: `src/paper_research_agent/agent/planner.py`
- Modify: `tests/agent/test_planner.py`

**Step 1: 写失败测试**

覆盖两条路径：

1. 第一次返回目标超出候选集合，第二次返回有效计划；断言第二次消息包含 `FAILURE_CODE=planner_target_outside_candidate_set` 及对应修复指令，最终计划审计为“失败、成功”。
2. 两次返回不完整网格；断言异常携带两个 body-free attempt audit，均为 `planner_grid_incomplete`。

不得断言或保存原始模型正文。

**Step 2: 运行测试确认失败**

```powershell
.venv\Scripts\python.exe -m pytest tests\agent\test_planner.py -q
```

Expected: FAIL；当前第二次提示只有通用契约说明。

**Step 3: 写最小实现**

增加固定映射，不把原始异常插入提示：

```python
_PLANNER_REPAIR_INSTRUCTIONS = {
    "planner_grid_incomplete": (
        "Return exactly one requirement and one discovery step for every "
        "resolved target-by-dimension pair."
    ),
    "planner_target_outside_candidate_set": (
        "Use corpus IDs only from LOCAL_CORPUS_CATALOG_JSON."
    ),
    # cover every stable failure code
}
```

第一次失败后追加：

```python
failure_code = _planner_failure_code(exc)
messages.append(
    HumanMessage(
        content=(
            f"FAILURE_CODE={failure_code}. "
            f"{_PLANNER_REPAIR_INSTRUCTIONS[failure_code]} "
            "Return the full corrected plan."
        )
    )
)
```

成功返回时把 audit 附加到 `ResearchPlan.planner_attempts`；最终失败时把 audit 附加到 `ComparisonTargetResolutionError`。不得增加第三次调用。

**Step 4: 运行测试确认通过**

Expected: PASS，`structured.ainvoke.await_count` 仍不超过 2。

**Step 5: 提交**

```powershell
git add src/paper_research_agent/agent/planner.py tests/agent/test_planner.py
git commit -m "修复: 按失败类型定向重试Planner"
```

### Task 3: 将 Planner 尝试审计接入评测但不泄露正文

**Files:**
- Modify: `src/paper_research_agent/evaluation/comparison_end_to_end.py`
- Modify: `scripts/evaluate_comparison_end_to_end.py`
- Modify: `tests/evaluation/test_comparison_end_to_end.py`
- Modify: `tests/evaluation/test_comparison_end_to_end_runner.py`

**Step 1: 写失败测试**

断言成功和失败 case 都能输出：

```json
"planner_attempts": [
  {"attempt": 1, "outcome": "contract_invalid", "failure_code": "planner_grid_incomplete"},
  {"attempt": 2, "outcome": "validated", "failure_code": null}
]
```

同时断言序列化结果不含 question、query、description、statement、evidence body 或模型原始输出。

**Step 2: 运行测试确认失败**

```powershell
.venv\Scripts\python.exe -m pytest tests\evaluation\test_comparison_end_to_end.py tests\evaluation\test_comparison_end_to_end_runner.py -q
```

**Step 3: 写最小实现**

- `ComparisonCaseDiagnostic` 增加 `planner_attempts`。
- 成功时从 `research.plan.planner_attempts` 读取。
- 失败时从 `ComparisonTargetResolutionError.attempts` 读取。
- 汇总增加按 `failure_code` 计数，但不记录自由文本。

**Step 4: 运行测试确认通过**

Expected: PASS。

**Step 5: 提交**

```powershell
git add src/paper_research_agent/evaluation/comparison_end_to_end.py scripts/evaluate_comparison_end_to_end.py tests/evaluation/test_comparison_end_to_end.py tests/evaluation/test_comparison_end_to_end_runner.py
git commit -m "评测: 记录Planner逐次契约失败审计"
```

### Task 4: 给 Compiler 限定词失败提供精确、逐事实修复输入

**Files:**
- Modify: `src/paper_research_agent/agent/reasoner.py`
- Modify: `tests/agent/test_reasoner.py`

**Step 1: 写 CPG020 形状的失败测试**

构造一个 requirement，其中唯一事实要求：

```python
required_qualifier_kinds=("condition", "method")
```

第一次模型输出映射正确、chunk scope 正确，但只返回 `method` qualifier；第二次返回 `condition + method`。断言：

- 第一次失败码为 `required_qualifier_missing`。
- 第二次 payload 只包含失败 requirement。
- 第二次 payload 明确包含该 fact 的完整 `required_qualifier_kinds`。
- 第二次 system message 明确说明 qualifier 是 per-fact 约束，不是 per-cell 汇总约束。
- 第二次通过并提交事实。

**Step 2: 运行测试确认失败**

```powershell
.venv\Scripts\python.exe -m pytest tests\agent\test_reasoner.py -q
```

**Step 3: 写最小实现**

将 retry payload 从单个字符串错误升级为固定结构：

```python
"repair_errors": {
    requirement_id: {
        "code": "required_qualifier_missing",
        "required_qualifiers_by_fact": {
            fact.fact_requirement_id: list(fact.required_qualifier_kinds)
            for fact in requirement.fact_requirements
        },
    }
}
```

Compiler system message增加确定性要求：映射到一个 fact requirement 的每个返回事实，都必须在自己的 `qualifiers` 数组中包含该 requirement 声明的全部 kind；不能依赖 statement 中出现限定词，也不能依赖同格另一个 fact 的 qualifiers。

不得自动生成 qualifier value；模型无法从可见证据提取时，应返回空事实而不是伪造限定条件。

**Step 4: 运行测试确认通过**

Expected: PASS；Compiler 最大调用次数仍为 2。

**Step 5: 提交**

```powershell
git add src/paper_research_agent/agent/reasoner.py tests/agent/test_reasoner.py
git commit -m "修复: 精确重试Compiler限定词缺项"
```

### Task 5: 将 Compiler 事务边界从整格缩小到事实

**Files:**
- Modify: `src/paper_research_agent/agent/coverage.py`
- Modify: `src/paper_research_agent/agent/reasoner.py`
- Modify: `tests/agent/test_coverage.py`
- Modify: `tests/agent/test_reasoner.py`

**Step 1: 写失败测试**

覆盖三种情况：

1. 同一格两个事实映射到同一 fact requirement：一个 qualifier 完整、一个缺项。保留完整事实，丢弃无效事实；只要计划 fact 已被有效事实满足，该格应提交。
2. 同一格包含两个不同 fact requirement：一个事实有效、一个事实缺 qualifier。保留有效事实，只重试未满足的 fact requirement。
3. 第二次仍失败：最终 assessment 为 `compiler_failed`，但第一次已验证事实仍在 ledger 和 generation input 中；不得因同格另一个事实失败而删除。

**Step 2: 运行测试确认失败**

```powershell
.venv\Scripts\python.exe -m pytest tests\agent\test_coverage.py tests\agent\test_reasoner.py -q
```

**Step 3: 提取单事实验证函数**

在 `coverage.py` 增加纯函数：

```python
def validate_evidence_compilation_fact(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    requirement: EvidenceRequirement,
    fact: EvidenceFactCompilation,
) -> EvidenceFactCompilation:
    """Validate scope, mapping and required qualifiers for one fact."""
```

`validate_evidence_compilation_cell` 调用该函数，保持旧 API 行为兼容。

**Step 4: 在批次校验中保留有效事实**

修改 `_validate_compilation_batch`：

- 先独立解析和验证每个 fact。
- 收集有效 facts 与每个 fact requirement 的满足状态。
- 无效额外 fact 不得使已满足的 planned fact 失败。
- 对多事实 requirement，只把尚未被有效事实覆盖的 fact ID 加入 retry。
- 第二次响应与第一次保留事实按 `fact_requirement_id + chunk_ids` 稳定去重后合并。
- 如果第二次仍失败，ledger 保留第一次有效事实，同时 assessment 标记 `compiler_failed`。

不得将无效事实的 statement、chunk 或 qualifier 自动复制到有效事实。

**Step 5: 扩展安全审计**

`EvidenceCompilationAttemptAudit` 增加计数型字段：

- `accepted_fact_count`
- `rejected_fact_count`
- `unresolved_fact_requirement_count`

只记计数，不记录正文。

**Step 6: 运行测试确认通过**

Expected: PASS；现有 sibling cell 事务测试继续通过。

**Step 7: 提交**

```powershell
git add src/paper_research_agent/agent/coverage.py src/paper_research_agent/agent/reasoner.py tests/agent/test_coverage.py tests/agent/test_reasoner.py
git commit -m "修复: 按事实保留Compiler有效编译结果"
```

### Task 6: 运行聚焦重复门禁并决定是否需要 Planner 确定性骨架

**Files:**
- Modify only if needed: `src/paper_research_agent/agent/models.py`
- Modify only if needed: `src/paper_research_agent/agent/planner.py`
- Modify only if needed: `src/paper_research_agent/agent/fact_queries.py`
- Modify only if needed: `tests/agent/test_planner.py`
- Modify only if needed: `tests/agent/test_fact_queries.py`

**Step 1: 运行纯测试门禁**

```powershell
.venv\Scripts\python.exe -m pytest tests\agent\test_models.py tests\agent\test_fact_queries.py tests\agent\test_planner.py tests\agent\test_coverage.py tests\agent\test_reasoner.py tests\evaluation\test_comparison_end_to_end.py tests\evaluation\test_comparison_end_to_end_runner.py -q
```

Expected: PASS。

**Step 2: 使用全新 checkpoint 重复运行 CPG013 五次**

沿用 `scripts/evaluate_comparison_end_to_end.py` 现有参数、当前 reranker、固定水化 4、自适应水化关闭。每次必须使用不同 checkpoint，禁止 resume。

门禁：

- 5/5 Planner 成功。
- 每次目标均为 `C013,T009`。
- 每次完整形成 2×3 网格。
- 3/3 v2 强制事实完成语义编译、表达和正确引用。
- 不得出现 `planner_contract_invalid` 或宽泛的 `planner_schema_invalid`。

**Step 3: 使用全新 checkpoint 重复运行 CPG020 五次**

门禁：

- 5/5 Planner 成功。
- 5/5 无最终 `required_qualifier_missing` 单元。
- 3/3 v2 强制事实达到语义覆盖并正确引用。
- 精确 chunk 可由同论文语义等价证据替代；不得要求唯一 chunk ID。

**Step 4: 执行停止/升级判断**

若 CPG013 和 CPG020 均 5/5 通过，停止 Planner 架构扩张，进入 Task 7。

若 CPG013 仍有任一次 Planner 契约失败，才实施确定性骨架：

- provider schema 不再让模型生成 requirement/step/target/dimension ID。
- 模型只返回已解析候选中的 corpus ID、维度标签和逐格事实语义提案。
- 本地代码按稳定顺序生成 target ID、dimension ID、完整 target × dimension requirements、step ID、corpus scope 和预算。
- `materialize_atomic_fact_steps` 继续负责事实级执行步骤。
- 缺格、越界候选或事实超预算仍 fail closed，不用宽查询或完整问题伪造缺失事实。

建议 provider-only 提案契约：

```python
class ComparisonFactProposal(FrozenContract):
    corpus_id: str
    dimension_index: int
    description: str
    protected_anchor_ids: tuple[int, ...]
    retrieval_expansions: tuple[str, ...] = ()
    required_qualifier_kinds: tuple[EvidenceQualifierKind, ...] = ()


class ComparisonPlanProposal(FrozenContract):
    selected_corpus_ids: tuple[str, ...]
    dimension_labels: tuple[str, ...]
    facts: tuple[ComparisonFactProposal, ...]
```

本地 builder 必须重新验证 candidate membership、完整网格、全局预算和锚点，再构造正式 `ResearchPlan`。该条件分支需要独立测试和独立提交：

```powershell
git commit -m "重构: 确定性构造比较规划控制骨架"
```

### Task 7: 运行修正版固定 Top 4 五题门禁

**Files:**
- Do not modify production code during the run.
- Local ignored output: `data/evaluations/runs/`
- Local ignored checkpoint: `data/runtime/`
- Create report after passing: `reports/Planner与Compiler契约方差修复复测-v1.md`

**Step 1: 确认实验不变量**

- 当前 reranker 未变。
- 每步骤水化数为 4。
- 自适应水化关闭。
- 使用 v2 金标范围。
- 全新 checkpoint，不 resume。
- 运行题目：`CPG007,CPG013,CPG020,CPG024,CPG026`。

**Step 2: 运行五题**

使用当前评测脚本的既有 CLI 参数；先执行 `--help` 核对参数名，不得凭计划中的旧命令猜测。

**Step 3: 检查门禁**

- 5/5 无 Planner 失败。
- `not_hydrated=0`，仅按 v2 明确必答事实计算。
- Compiler 最终失败单元为 0。
- 已编译强制事实全部进入 generation input。
- 已表达强制事实引用正确。
- 审计中没有问题正文、查询正文、事实正文或论文正文。

任一项失败时停止，不运行全量 30 题；按新的细分失败码归因，不修改检索参数掩盖问题。

**Step 4: 写复测报告并提交**

报告必须列出代码版本、检索配置哈希、checkpoint ID、题目数、Planner attempt 汇总、Compiler fact 接受/拒绝计数、事实链路损失阶段和证据边界。

```powershell
git add reports/Planner与Compiler契约方差修复复测-v1.md
git commit -m "报告: 记录Planner与Compiler方差复测结果"
```

### Task 8: 运行 v2 全量 30 题和工程质量门禁

**Files:**
- Local ignored output only during evaluation.
- Update: `reports/Planner与Compiler契约方差修复复测-v1.md`

**Step 1: 运行全量单元测试**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected: 不少于交接基线的 875 个测试和 121 个子测试通过，无新增告警。

**Step 2: 运行静态检查**

```powershell
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\mypy.exe
```

Expected: PASS。

**Step 3: 使用 v2 金标全量运行 30 题**

必须使用固定 Top 4、全新 checkpoint、当前 reranker、关闭自适应水化，不得 resume 五题运行。

全量门禁：

- 30/30 无 `planner_contract_invalid`、`planner_schema_invalid` 或其他 Planner 终止错误。
- 最终 Compiler failed unit 为 0。
- v2 强制事实 `not_hydrated=0`；若不为 0，必须提供事实级排名诊断后再决定是否属于新的检索问题。
- 语义事实编译、生成输入、表达和正确引用阶段不得发生无解释损失。
- 精确 chunk recall 与语义事实 recall 分开报告。

**Step 4: 更新报告**

报告不得宣称 v2 是最终人工金标。明确列出：

- 自动范围审定仍待人工双审；
- 全量结果只证明当前开发集上的稳定性；
- 任何同论文替代证据必须经过语义支持和引用正确性检查。

**Step 5: 检查提交内容**

```powershell
git status --short
git diff --check
git diff --cached --check
```

确认 `.env`、私有金标、运行 JSON、SQLite、日志和论文正文未进入暂存区。

**Step 6: 提交最终报告**

```powershell
git add reports/Planner与Compiler契约方差修复复测-v1.md
git commit -m "评测: 完成修正版金标全量稳定性门禁"
```

## 四、禁止的捷径

- 不得把 Top 4 改为 Top 8/Top 20 以掩盖 Planner 或 Compiler 失败。
- 不得关闭 reranker 或针对单题手工调权。
- 不得把可选事实重新放回主门禁证明“修复有效”。
- 不得把 Compiler 缺少的 qualifier value 从 Planner 描述机械复制过去。
- 不得在验证失败后直接把 `evidence_sufficient=True`。
- 不得为通过测试删除 fail-closed 校验。
- 不得在报告中只给总体百分比而隐藏 Planner 整题失败或 Compiler 单元失败。
- 不得覆盖 v1 金标或提交私有 v2 金标。

## 五、新任务启动提示

新任务开始时先执行：

```powershell
git status --short
git log -5 --oneline
.venv\Scripts\python.exe -m pytest tests\agent\test_planner.py tests\agent\test_reasoner.py tests\agent\test_coverage.py -q
```

然后读取：

- `reports/端到端金标范围审计与Top4复测-v2.md`
- `reports/事实查询语义锚点与执行门控诊断-v1.md`
- `reports/当前重排器误排原因诊断-v1.md`
- 本计划全文。

从 Task 1 开始，不重新讨论或修改已经通过门禁的 Top 4 检索链路。
