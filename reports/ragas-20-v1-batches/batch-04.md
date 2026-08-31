# Ragas 论文研究 Agent 评测

- 实验指纹：`902a141b965672ab6403469b622f3435f2bc299b8d0395bb4a5172ed695bfa1b`
- 问题数：4
- Ragas 实际评分率：75.00%
- 回答状态正确率：75.00%
- 发布门禁：未启用，仅作诊断

| 指标 | 平均分 | 诊断阈值 |
|---|---:|---:|
| selected_context_precision | 0.4917（待改进） | 0.8 |
| selected_context_recall | 0.3333（待改进） | 0.8 |
| response_faithfulness | 0.6111（待改进） | 0.9 |
| answer_relevancy | 0.8942（达标） | 0.8 |
| citation_faithfulness | 0.7361（待改进） | 0.9 |
| citation_validity | 1.0000 | N/A |
| paper_id_precision | 0.7778 | N/A |
| paper_id_recall | 0.8333（达标） | 0.8 |

## 解释边界

- Ragas 指标只对 answered 且存在引用来源的结果评分
- Ragas 评分使用实际入选 chunk 的完整正文；正文只驻留内存且不写入结果
- selected_context 指标评估最终入选引用，不等同于原始召回排序
- paper_id 指标的金标仅来自 must-have claim 的 supports 关系
- LLM 裁判分数必须用人工双标样本校准，当前诊断阈值未启用发布门禁
- 结果不保存问题、回答、claim、证据摘录、Provider 原始响应或本地路径
