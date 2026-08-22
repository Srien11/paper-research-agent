# 主 Agent 自适应编排与比较规划维度修复复测 v1

## 阶段结论

本轮修复通过工程门禁、CPG020 三次全新会话门禁和固定 Top 4 五题 v2 门禁。

- CPG020 三次均走 `fast_path`，不再出现 `planner_dimension_invalid`；主路由加快路物化的 p50 约为 15.00 ms，加上上下文水化后的主编排前奏 p50 约为 62.61 ms，低于 500 ms 目标和 1 秒硬上限。
- CPG020 三次总耗时为 45.68、29.51、28.51 秒，p50 为 29.51 秒；Research Planner、六路 search batch、Compiler 和完整 research child 分开计时，答案 Provider 三次均为 0 attempts / 0 ms。
- 固定五题 v2 评测 5/5 成功，五题均由本地 parser 按原顺序物化 3 个维度；每题 2 targets × 3 dimensions 形成 6 格完整网格。
- 15/15 v2 强制事实全部完成检索、水化、Compiler 可见、编译、进入确定性答案、表达和引用正确性链路；Compiler 30/30 请求单元通过，失败单元为 0。
- 五题真实主 Agent 轨迹中，CPG020、CPG024 命中快路；CPG007、CPG013、CPG026 因 `complex_or_ambiguous` 保守进入完整规划。分类器没有为了性能强制快路，5/5 均成功完成。

本报告不包含问题、查询、事实、答案、模型输出或论文正文。v2 金标仍属于内部评测资产，不随报告提交。

## 实验血缘与固定配置

- 代码版本：`0d3244133341832815ac48335f1f633d55e96c3a`。
- 五题 v2 结果：`data/evaluations/runs/main-adaptive-five-v2.json`（本地忽略文件）。
- 五题 checkpoint：`data/runtime/main-adaptive-five-v2.sqlite3`（本地忽略文件）。
- checkpoint ID：`881e529b825efa05883722000b4e974e087aff952fd7a5a5849ee2bb421c0bbd`。
- retrieval config SHA256：`e36d5fa4bedcf250eef09103459a52a1efc0561f7b7f4d313b8af4a68dbfb188`。
- experiment fingerprint：`60bdfd768987dff73312188d2176e5d47afb1379596c2a6c4cfde95a6937f6b7`。
- 题目：`CPG007, CPG013, CPG020, CPG024, CPG026`。
- 评测配置：v2 金标、当前 reranker、固定 Top 4、comparison concurrency=6、local workers=6、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、自适应证据水化关闭、fast path enabled、全新 checkpoint、不 resume。
- 本轮未修改 Top 4、六路并发、retriever、reranker、融合权重、query rewrite、Compiler 或确定性答案契约。

首次五题命令误用了脚本默认的 v1 答案金标，结果仅作诊断并已明确弃用；最终门禁使用独立全新 checkpoint 和显式 `comparison-end-to-end-gold-v2.jsonl` 重新运行，以下五题结论全部来自 v2 结果。

## 工程门禁

| 门禁 | 结果 |
| --- | ---: |
| 广覆盖聚焦测试 | 348 passed，92 subtests |
| 全量 pytest | 922 passed，162 subtests |
| Ruff | 通过 |
| mypy | 159 source files，通过 |
| Task 7 观测聚焦测试 | 70 passed，18 subtests |

两条 pytest warning 分别来自 Starlette 的 `httpx` 弃用提示和 Pydantic Settings 的未解析前向引用提示，均非本轮回归失败。

## CPG020 修复前后对比

修复前最近三次失败的最终安全码均为 `planner_dimension_invalid`。代表性失败运行总耗时约 34.13 秒：上下文水化约 48.5 ms，固定主 Agent 模型前奏约 13.54 秒，Research Planner 约 20.47 秒，随后在进入 search、Compiler 和 answer 前终止。

修复后 CPG020 三次全新主 Agent 会话结果：

| 指标 | 运行 1 | 运行 2 | 运行 3 | p50 |
| --- | ---: | ---: | ---: | ---: |
| 主请求总耗时 | 45.68 s | 29.51 s | 28.51 s | 29.51 s |
| 上下文水化 | 51.03 ms | 47.61 ms | 44.43 ms | 47.61 ms |
| 确定性主路由 | 0.08 ms | 0.38 ms | 0.07 ms | 0.08 ms |
| 快路 goal/task 物化 | 14.92 ms | 15.51 ms | 13.91 ms | 14.92 ms |
| 完整 research child | 45.36 s | 29.23 s | 28.23 s | 29.23 s |
| Research Planner | 13.37 s | 5.30 s | 4.63 s | 5.30 s |
| 六路 search batch | 20.61 s | 12.73 s | 12.72 s | 12.73 s |
| Compiler | 11.18 s | 11.05 s | 10.69 s | 11.05 s |
| Answer Provider | 0 ms / 0 attempts | 0 ms / 0 attempts | 0 ms / 0 attempts | 0 |

三次均满足：

- `planning_route=fast_path`，原因码为 `clear_single_local_rag`。
- 主 Agent 的 TurnInterpreter、GoalReconciler、TaskPlanner Provider 调用均为 0。
- `planner_fallback_reason=null`，没有维度错误或比较规划终止。
- 本地维度为 3，2 targets × 3 dimensions 为 6 格完整网格。
- Compiler 失败单元为 0；答案 Provider 的 attempts、input tokens、output tokens 和 latency 均为 0。

运行 1 的 Research Planner 和 search 延迟较高，但主编排仍保持约 15 ms；这证明主编排快路与外部 Provider/检索方差已经被正确分离。

## 固定五题 v2 正确性门禁

| 门禁 | 结果 |
| --- | ---: |
| 成功题目 | 5/5 |
| 最终目标准确 | 5/5 |
| 结构门禁通过 | 5/5 |
| 本地维度数量与顺序匹配 | 5/5 |
| 完整 target×dimension 网格 | 5 × 6 格 |
| Planner 尝试 | 5，均首次验证成功 |
| Planner schema/contract 失败 | 0/0 |
| target resolution/planning failure | 0/0 |
| deterministic fact fallback | 0 |
| Compiler 请求/接受单元 | 30/30 |
| Compiler 最终失败单元 | 0 |
| Compiler 接受/拒绝事实 | 17/0 |
| 未解析事实要求 | 0 |
| 确定性答案 Provider 输入/输出 token | 0/0 |

五题每题均为 2 个目标、3 个本地维度和 6 个唯一网格单元。Provider 没有维度增删、重排或改名的控制权。

## v2 强制事实链路

本轮共评估 15 条 v2 强制事实：

| 阶段 | 通过率 | 最早损失数 |
| --- | ---: | ---: |
| retrieved | 100% | `not_retrieved=0` |
| hydrated | 100% | `not_hydrated=0` |
| visible to Compiler | 100% | `not_visible_to_compiler=0` |
| exact gold chunk compiled | 100% | — |
| semantic fact compiled | 100% | `not_compiled=0` |
| deterministic answer input | 100% | `not_in_generation_input=0` |
| expressed | 100% | `not_expressed=0` |
| citation correct | 100% | `citation_incorrect=0` |
| 完整链路 | 15/15 | `complete=15` |

精确金标 chunk recall 和语义事实 recall 均为 100%。本轮没有使用同论文替代证据；若后续出现替代证据，仍必须单独通过语义支持与引用正确性判断，不能仅凭 corpus 相同视为通过。

## 主 Agent 自适应路由与耗时

下表来自五题各一次真实主 Agent 全新会话；它与上面的“强制比较 v2 正确性评测”是两类证据，不能混为同一耗时口径。

| 题号 | 路由 | 路由原因 | 路由 | 快路物化 | 完整规划 | child | 比较 Planner / search / Compiler | Answer Provider | 总耗时 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| CPG007 | full planner | complex_or_ambiguous | 0.05 ms | — | 20.23 s | 22.25 s | 非强制比较路径 | 13.76 s / 3 attempts | 42.82 s |
| CPG013 | full planner | complex_or_ambiguous | 0.08 ms | — | 18.91 s | 17.30 s | 非强制比较路径 | 10.56 s / 2 attempts | 68.50 s |
| CPG020 | fast path | clear_single_local_rag | 0.14 ms | 15.34 ms | — | 41.23 s | 13.71 / 15.75 / 11.53 s | 0 / 0 attempts | 41.59 s |
| CPG024 | fast path | clear_single_local_rag | 0.08 ms | 17.64 ms | — | 39.32 s | 14.12 / 15.44 / 9.61 s | 0 / 0 attempts | 39.69 s |
| CPG026 | full planner | complex_or_ambiguous | 0.27 ms | — | 16.57 s | 17.50 s | 非强制比较路径 | 10.91 s / 2 attempts | 67.88 s |

解释：

- CPG020、CPG024 能被无模型允许清单证明为明确单一私有论文比较，因此固定构造一个 `local_rag` 任务，并继续复用路由、child、评估、workspace 更新、提交校验与幂等链路。
- CPG007、CPG013、CPG026 无法被允许清单证明为明确单一比较，按设计进入完整规划。其 Answer Provider 次数属于完整规划后选择的非强制比较 RAG 回答，不是 Research Planner 的重试次数。
- Research Planner 的 structured-output 尝试上限仍为 2；五题 v2 强制比较评测均在第 1 次验证成功。

## fallback 与失败分类

- 五题生产形态和 v2 评测均未触发 fallback：`fallback_count=0`，无 fallback reason code。
- “未使用 fallback”样本：五题均在第一次事实提案中验证成功，计划记录没有回退原因。
- “使用 fallback”路径由 `test_invalid_fact_repairs_use_question_derived_fallback` 固定验证：连续两次非法事实提案后，维度骨架保持不变，事实来源标记为 `planned_fallback`，安全原因码为 `fact_proposal_repair_exhausted`，且事件不复制问题或模型输出正文。
- 目标选择非法时仍抛 `ComparisonPlanningError` 或 `ComparisonTargetResolutionError` 的准确分类，不允许 fallback 猜测候选论文。

## 模型裁判方差

确定性事实血缘显示 15/15 强制事实表达且引用正确。聚合模型裁判给出 must-have claim recall 93.3%、citation correctness 78.6%，低于确定性追踪；模型裁判失败率为 0，dimension coverage 为 100%。

本门禁对“精确金标 chunk 是否随对应强制事实进入确定性答案并被正确引用”采用可直接核验的事实血缘结果，同时披露模型裁判方差。对于非精确 chunk 的替代证据，不适用这一确定性结论，仍需语义判断与人工复核。

## 安全、回滚与结论边界

- SQLite 观测升级到 `user_version=4`，仅新增固定 `planning_route` 和安全码 `fallback_reason`；不保存原始问题、query、fact、answer、Provider output 或论文正文。
- 冒烟摘要只读取当前 event cursor 之后的事件，不跨运行累计。
- 快路回滚只需设置 `PRA_MAIN_AGENT_FAST_PATH_ENABLED=false`；这会让新请求重新进入原完整主规划，不回滚六路检索、Compiler、固定维度骨架或确定性答案。
- 本报告只证明计划指定的 CPG020 三次与固定五题门禁。按计划停止于五题，不运行 30 题，不调整模型、retriever、reranker 或评测集。
