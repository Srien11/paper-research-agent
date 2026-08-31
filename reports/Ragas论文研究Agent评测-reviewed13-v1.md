# Ragas 论文研究 Agent 评测

- 实验指纹：`dbe7571c16908c47378fe50928d88cd6227b1d2cd127b0b490da15d05422687f`
- 问题数：13
- Ragas 实际评分率：84.62%
- 回答状态正确率：84.62%
- 发布门禁：未启用，仅作诊断

| 指标 | 平均分 | 诊断阈值 |
|---|---:|---:|
| selected_context_precision | 0.6798（待改进） | 0.8 |
| selected_context_recall | 0.9091（达标） | 0.8 |
| response_faithfulness | 1.0000（达标） | 0.9 |
| answer_relevancy | 0.8045（达标） | 0.8 |
| citation_faithfulness | 0.9091（达标） | 0.9 |
| citation_validity | 1.0000 | N/A |
| paper_id_precision | 0.4682 | N/A |
| paper_id_recall | 1.0000（达标） | 0.8 |

## 解释边界

- Ragas 指标只对 answered 且存在引用来源的结果评分
- Ragas 评分使用实际入选 chunk 的完整正文；正文只驻留内存且不写入结果
- selected_context 指标评估最终入选引用，不等同于原始召回排序
- paper_id 指标的金标仅来自 must-have claim 的 supports 关系
- LLM 裁判分数必须用人工双标样本校准，当前诊断阈值未启用发布门禁
- 结果不保存问题、回答、claim、证据摘录、Provider 原始响应或本地路径
- 评测按独立进程分批运行，以便在批次之间释放本地模型内存
