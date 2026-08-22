# 30 题全量评测 Compiler 终态统计修正

## 修正对象与可追溯信息

- 原结果：`data/evaluations/runs/comparison-current-full30-top4-v2-20260822-r1.json`
- 修正结果：`data/evaluations/runs/comparison-current-full30-top4-v2-20260822-r1-corrected.json`
- 原结果 SHA-256：`78e98853b6dbd39a2b1acaa5aa4ca8c1aafe86809f54db98f3566a99de877df7`
- 原运行代码版本：`adbcde5e663493cfa404f938aae636e413435dea`
- 实验指纹：`192268bf982e146605b9ce332423ad3a8c5a95d20b276142462e85a59f609494`
- 修正结果格式：`comparison-e2e-run-v2`

本次修正通过离线重算完成。工具仅读取原 run JSON 中的 body-free（不含正文）诊断，未加载 `.env`，未访问网络，也未调用检索、Planner、Compiler、回答模型或裁判模型。原结果未被覆盖；修正结果另存，并通过 `source_run_sha256` 关联上述原结果哈希。

## 缺陷原因

原聚合器把每次 Compiler 尝试中的请求、接受和失败单元直接跨尝试求和。该统计可以描述完整重试过程，却被字段名和门禁报告误作每题事务完成后的最终状态。因此，第一次失败但第二次定向重试成功的单元仍被误报为最终失败。

修正后的聚合明确拆分为两层：

- `attempts`：保留所有历史尝试，完整呈现重试成本和曾发生的失败；
- `final`：按题、按 requirement 合并每个单元的最新状态，表示最终事务结果。

## 修正结果

Compiler 尝试：33 次，3 题触发重试；历史请求 168 单元、接受 163 单元、失败 5 单元。3 题中的 5 个单元均在第一次尝试失败，第二次定向重试全部恢复。

Compiler 终态：163/163 单元通过，最终失败 0，最终未解析事实要求 0，最终保留事实 107 条。

分区结果如下：

| 分区 | 题数 | 尝试数 | 重试题数 | 历史失败单元 | 终态通过 | 最终失败 | 最终未解析 | 保留事实 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dev | 20 | 22 | 2 | 4 | 111/111 | 0 | 0 | 73 |
| sealed_test | 10 | 11 | 1 | 1 | 52/52 | 0 | 0 | 34 |
| 合计 | 30 | 33 | 3 | 5 | 163/163 | 0 | 0 | 107 |

## 不变指标与边界

逐字段核对确认，修正前后的 `experiment_fingerprint`、`question_count`、`summary`、`answer_scores`、`fact_lineage`、`planner` 和 `cases` 完全相同；各分区除 `compilation` 外的所有汇总也完全相同。允许变化的字段仅为：

- `schema_version`；
- `source_run_sha256`；
- 顶层 `compilation`；
- `split_summaries.<split>.compilation`。

因此，本修正不改变以下既有结论：

- 运行成功率仍为 30/30；
- 严格全条件通过率仍为 20/30；
- 目标选择、事实血缘和模型裁判指标均保持不变；
- 答案内容及其确定性评分均保持不变。

本报告不包含问题、查询、事实、答案、论文正文、chunk ID 或 Provider 原始输出；两个 run JSON 均继续由 `.gitignore` 排除，不进入版本控制。
