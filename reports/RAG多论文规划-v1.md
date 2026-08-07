# RAG 银标 Rubric 诊断

- 实验指纹：`3e6a57d63d23bd7c0b74c9c8a83fe2e2c955c58043fcd49fdc23369b0c947cc2`
- 题目：10（可回答 10 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 70.00% |
| 回答状态准确率 | 40.00% |
| Must-have Claim F1 | 18.07% |
| Citation F1 | 12.50% |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | 58.33% |
| 拒答 F1 | N/A |
| 错误拒答率 | 30.00% |
| Span Recall | 14.29% |
| 必要证据组 Recall | 18.75% |
| 论文 Recall | 71.43% |
| 端到端延迟 P50 | 8681.93 ms |
| 端到端延迟 P95 | 34656.37 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `multi_paper_comparison` | 40.00% | 18.07% | 14.29% |

## 安全错误类型

{"AnswerGenerationError": 1, "AnswerValidationError": 1, "ValueError": 1}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
