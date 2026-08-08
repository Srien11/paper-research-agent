# RAG 银标 Rubric 诊断

- 实验指纹：`946b8442d90ba5667124f1c89cce95f4e7f3ee17eedb8b93f41f17c6ef9ac112`
- 题目：5（可回答 5 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 60.00% |
| 回答状态准确率 | 60.00% |
| Must-have Claim F1 | 47.37% |
| Citation F1 | 47.37% |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | 18.18% |
| 拒答 F1 | N/A |
| 错误拒答率 | 0.00% |
| Span Recall | 0.00% |
| 必要证据组 Recall | 0.00% |
| 论文 Recall | 100.00% |
| 端到端延迟 P50 | 88083.95 ms |
| 端到端延迟 P95 | 93766.11 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `multi_paper_comparison` | 60.00% | 47.37% | 0.00% |

## 安全错误类型

{"ValueError": 2}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
