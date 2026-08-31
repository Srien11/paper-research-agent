# Ragas 银标质量审计 v1

## 审计范围

- 私有源数据题数：80
- 本轮冻结范围：GQ001～GQ020，共 20 题
- 默认保留：13 题
- 隔离：7 题
- 审计方式：本地只读检查问题、must-have claim、citation relation、证据归属和 OCR
  质量；未调用外部模型
- 标注状态：全部仍为 `silver_generated`，不是人工仲裁金标

## 隔离决定

| 题目 ID | 原因码 | 复核状态 |
|---|---|---|
| GQ003 | `cross_paper_attribution` | confirmed |
| GQ009 | `question_reference_mismatch` | high_risk |
| GQ011 | `degraded_ocr_evidence` | confirmed |
| GQ012 | `incomplete_reference` | confirmed |
| GQ013 | `question_reference_mismatch` | confirmed |
| GQ014 | `non_atomic_reference` | confirmed |
| GQ017 | `incomplete_reference` | confirmed |

隔离清单只保存稳定 ID、原因码和复核状态，不保存问题、答案、claim 或证据正文。
原始私有 JSONL 保持不变，后续完成双人复核和仲裁后才能恢复题目。

## 对旧结果的敏感性分析

| 统计范围 | 评分题数 | Selected Context Precision | Selected Context Recall |
|---|---:|---:|---:|
| 原始结果 | 15 | 0.499 | 0.700 |
| 剔除 GQ012、GQ014、GQ017 | 12 | 0.624 | 0.875 |
| 再剔除高风险 GQ009 | 11 | 0.680 | 0.909 |

该分析是事后诊断，不是新的正式基线。它表明旧召回分数明显受银标缺陷影响；精确率在
较可信子集上仍偏低，Agent 的上下文去冗余仍需单独优化。

## 已实施门禁

- Ragas 默认入口先应用隔离清单，再执行题型、offset 和 limit 选择。
- 清单同时冻结本轮20题范围，排除后不会自动补入后续未审计题。
- 运行结果记录隔离清单哈希、隔离数量和题目 ID，便于复现。
- 银标生成阶段新增跨论文归属、损坏证据和占位式 claim 的确定性拒绝规则。
- 未进入 citation relation 的候选证据不再标记为 `required`。

## 隔离后重跑结果

- 重跑题数：13
- 运行成功：13/13
- Ragas 实际评分：11/13；GQ018、GQ019 仍为 `insufficient_evidence`
- Selected Context Precision：0.6798
- Selected Context Recall：0.9091
- Response Faithfulness：1.0000
- Answer Relevancy：0.8045
- Citation Faithfulness：0.9091
- Paper ID Recall：1.0000

完整安全报告：`reports/Ragas论文研究Agent评测-reviewed13-v1.md`。该结果仍基于
`silver_generated`，只能作为清洗后的诊断基线，不能替代人工双标与仲裁。
