# RAG 银标 Rubric 诊断

- 实验指纹：`568acaec773eb8ebcc187e1a9d5d8958bc91223cc3cb2f87717e0e63758c9fd5`
- 题目：5（可回答 5 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 100.00% |
| 回答状态准确率 | 80.00% |
| Must-have Claim F1 | 40.91% |
| Citation F1 | 40.91% |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | 47.06% |
| 拒答 F1 | N/A |
| 错误拒答率 | 20.00% |
| Span Recall | 9.09% |
| 必要证据组 Recall | 8.33% |
| 论文 Recall | 60.00% |
| 端到端延迟 P50 | 69703.03 ms |
| 端到端延迟 P95 | 76907.75 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `multi_paper_comparison` | 80.00% | 40.91% | 9.09% |

## 安全错误类型

{}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
