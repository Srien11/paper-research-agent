# RAG 银标 Rubric 诊断

- 实验指纹：`eced4fae2bc739698c7be8ff80ec4dfc6facbb532b8d75c8c6226428f6a8d0a1`
- 题目：5（可回答 5 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 80.00% |
| 回答状态准确率 | 40.00% |
| Must-have Claim F1 | 25.81% |
| Citation F1 | 25.81% |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | 42.86% |
| 拒答 F1 | N/A |
| 错误拒答率 | 40.00% |
| Span Recall | 11.11% |
| 必要证据组 Recall | 22.22% |
| 论文 Recall | 75.00% |
| 端到端延迟 P50 | 82723.14 ms |
| 端到端延迟 P95 | 110300.00 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `multi_paper_comparison` | 40.00% | 25.81% | 11.11% |

## 安全错误类型

{"ComparisonTargetResolutionError": 1}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
