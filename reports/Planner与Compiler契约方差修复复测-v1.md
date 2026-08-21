# Planner 与 Compiler 契约方差修复复测 v1

## 阶段结论

Planner（规划器）与 Compiler（证据编译器）的契约方差修复已通过固定 Top 4 五题门禁。五题均在第一次 Planner 调用中形成有效计划，Compiler 对 30 个请求单元全部完成验证，15 条 v2 强制事实均完成检索、水化、编译、生成输入、表达和引用正确性链路，没有事实损失。

本阶段结果仅证明当前开发集五题样本上的稳定性。v2 金标是 Agent 辅助的范围审定结果，尚未完成人工双审或仲裁；全量 30 题结论将在工程门禁和独立全新 checkpoint 运行完成后补充。

## 实验血缘与不变量

- 代码版本：`3bbf32d083b744cf60f345e802d07e8f1b9a0dc1`。
- 运行结果：`data/evaluations/runs/planner-compiler-five-top4-v1.json`（本地忽略文件）。
- checkpoint 文件：`data/runtime/planner-compiler-five-top4-v1.sqlite3`（本地忽略文件）。
- checkpoint ID：`543b9c4425022b361ffa634ee844079b5d239f470e42ddd3d21cdd4cca285fc8`。
- retrieval config SHA256：`e36d5fa4bedcf250eef09103459a52a1efc0561f7b7f4d313b8af4a68dbfb188`。
- experiment fingerprint：`8a80b03a703df16d99944fc0f52e42c74c669aacbb047d556bd06ffd968d2737`。
- 题目：`CPG007,CPG013,CPG020,CPG024,CPG026`。
- 配置：当前 reranker、每步骤固定水化 4、自适应水化关闭、v2 金标、全新 checkpoint、不 resume。
- 本轮未修改 Top 4、retriever、reranker、融合权重或 query rewrite。

## 五题门禁结果

| 门禁 | 结果 |
| --- | ---: |
| 成功题目 | 5/5 |
| 最终目标准确 | 5/5 |
| 结构门禁通过 | 5/5 |
| Planner 尝试 | 5 |
| Planner 首次验证成功 | 5/5 |
| Planner schema/contract 失败 | 0/0 |
| Compiler 请求/接受单元 | 30/30 |
| Compiler 最终失败单元 | 0 |
| Compiler 接受/拒绝事实 | 21/0 |
| 未解析事实要求 | 0 |

Planner 没有产生失败码。五题均为一次调用成功，因此没有触发第二次定向修复。Compiler 也均在第一次尝试通过，没有触发限定词或映射重试。

## v2 强制事实链路

本轮共评估 15 条 v2 强制事实：

| 阶段 | 通过率 | 最早损失数 |
| --- | ---: | ---: |
| retrieved | 100% | `not_retrieved=0` |
| hydrated | 100% | `not_hydrated=0` |
| visible to Compiler | 100% | `not_visible_to_compiler=0` |
| exact gold chunk compiled | 100% | — |
| semantic fact compiled | 100% | `not_compiled=0` |
| generation input | 100% | `not_in_generation_input=0` |
| expressed | 100% | `not_expressed=0` |
| citation correct | 100% | `citation_incorrect=0` |
| 完整链路 | 15/15 | `complete=15` |

精确金标 chunk recall 与语义事实 recall 本轮均为 100%，但两者仍作为不同指标报告。同论文替代证据不能仅凭 corpus 相同而视为通过；后续如出现替代证据，仍必须同时通过语义支持与引用正确性检查。

## 模型裁判方差

聚合模型裁判给出 must-have claim recall 93.3%、citation correctness 92.9%，低于确定性事实链路的 100%。差异集中在 `CPG024`：模型裁判记为 must-have 3/4、citation 2/4；确定性追踪则显示该题 4/4 强制事实均编译、进入生成输入、被表达且引用正确，并且引用了精确金标 chunk。

因此本门禁对“精确金标 chunk 是否随对应强制事实进入答案”采用确定性判定，保留模型裁判结果作为方差披露，而不让一次语义裁判否定可直接核验的精确引用关系。对于非精确 chunk 的同论文替代证据，仍要求模型语义判定和引用正确性检查，不能套用这一确定性豁免。

## 安全与证据边界

- 运行审计只保留安全 ID、固定失败码、计数、哈希与排名诊断；报告不包含问题正文、查询正文、事实正文或论文正文。
- v2 是 Agent 辅助范围审定，仍待人工双审；分歧项需要人工仲裁后才能称为最终人工金标。
- 五题结果不能外推为生产分布或全量数据稳定性结论。
- 本轮证明的是固定 Top 4 下的端到端事实链路，不证明其他检索配置或其他数据集同样稳定。

## 后续门禁

全量单元测试、Ruff、mypy 与 v2 全量 30 题尚待执行。全量运行必须使用独立全新 checkpoint，并继续保持固定 Top 4、当前 reranker、自适应水化关闭和不 resume。
