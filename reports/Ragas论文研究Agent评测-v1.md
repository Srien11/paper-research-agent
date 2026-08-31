# Ragas 论文研究 Agent 评测

> 状态：该20题结果已因银标质量审计而停止作为当前基线。GQ003、GQ009、GQ011、
> GQ012、GQ013、GQ014、GQ017 已进入隔离清单。修正后的13题结果见
> `reports/Ragas论文研究Agent评测-reviewed13-v1.md`，审计依据见
> `reports/Ragas银标质量审计-v1.md`。

- 实验指纹：`9e7eed589758ad3df7552a2d3efae24547f177ef1d957afc557e2234f488e862`
- 问题数：20
- Ragas 实际评分率：75.00%
- 回答状态正确率：75.00%
- 发布门禁：未启用，仅作诊断

| 指标 | 平均分 | 诊断阈值 |
|---|---:|---:|
| selected_context_precision | 0.4990（待改进） | 0.8 |
| selected_context_recall | 0.7000（待改进） | 0.8 |
| response_faithfulness | 0.8994（待改进） | 0.9 |
| answer_relevancy | 0.8334（达标） | 0.8 |
| citation_faithfulness | 0.9139（达标） | 0.9 |
| citation_validity | 1.0000 | N/A |
| paper_id_precision | 0.5767 | N/A |
| paper_id_recall | 0.9000（达标） | 0.8 |

## 解释边界

- Ragas 指标只对 answered 且存在引用来源的结果评分
- Ragas 评分使用实际入选 chunk 的完整正文；正文只驻留内存且不写入结果
- selected_context 指标评估最终入选引用，不等同于原始召回排序
- paper_id 指标的金标仅来自 must-have claim 的 supports 关系
- LLM 裁判分数必须用人工双标样本校准，当前诊断阈值未启用发布门禁
- 结果不保存问题、回答、claim、证据摘录、Provider 原始响应或本地路径
- 评测按独立进程分批运行，以便在批次之间释放本地模型内存
