# RAG 银标 Rubric 诊断

- 实验指纹：`25f546951b7e724deb1457b6e393039218ea9b4444b9b47bd87aaac7c6c00718`
- 题目：1（可回答 1 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 100.00% |
| 回答状态准确率 | 100.00% |
| Must-have Claim F1 | 100.00% |
| Citation F1 | 100.00% |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | 0.00% |
| 拒答 F1 | N/A |
| 错误拒答率 | 0.00% |
| Span Recall | 66.67% |
| 必要证据组 Recall | 50.00% |
| 论文 Recall | 100.00% |
| 端到端延迟 P50 | 38437.53 ms |
| 端到端延迟 P95 | 38437.53 ms |

## 题型切片

| 题型 | 状态准确率 | Claim F1 | Span Recall |
|---|---:|---:|---:|
| `multi_paper_comparison` | 100.00% | 100.00% | 66.67% |

## 安全错误类型

{}

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
