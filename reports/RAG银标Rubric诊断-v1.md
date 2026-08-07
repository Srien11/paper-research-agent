# RAG 银标 Rubric 诊断

- 实验指纹：`fdbf9a7c95f85a3f0bb43a0f515dce87848b90abc5713b4b32b376cf89f46eb4`
- 题目：80（可回答 60 / 不可回答 20）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 98.75% |
| 回答状态准确率 | 68.75% |
| Must-have Claim F1 | 42.36% |
| Citation F1 | 34.73% |
| Forbidden Claim Rate | 4.65% |
| Unsupported Claim Rate | 56.03% |
| 拒答 F1 | 52.00% |
| 错误拒答率 | 28.33% |
| Span Recall | 41.30% |
| 必要证据组 Recall | 39.29% |
| 论文 Recall | 71.43% |
| 端到端延迟 P50 | 8998.95 ms |
| 端到端延迟 P95 | 14154.39 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `conflicting_evidence` | 66.67% | 61.43% | 50.00% |
| `definition_scope` | 90.91% | 52.17% | 37.50% |
| `experimental_result` | 63.64% | 53.44% | 45.45% |
| `figure_table_explanation` | 88.89% | 37.58% | 77.78% |
| `method_mechanism` | 70.59% | 35.83% | 50.00% |
| `multi_hop_synthesis` | 50.00% | 48.00% | 35.71% |
| `multi_paper_comparison` | 50.00% | 17.36% | 16.67% |

## 安全错误类型

{"AnswerGenerationError": 1}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
