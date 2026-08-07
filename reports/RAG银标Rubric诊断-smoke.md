# RAG 银标 Rubric 诊断

- 实验指纹：`00ba43212bec64a438ce27de2d027dea6dacfc04c653c690a543661d2da5e88c`
- 题目：1（可回答 1 / 不可回答 0）

| 指标 | 结果 |
|---|---:|
| 运行成功率 | 100.00% |
| 回答状态准确率 | 0.00% |
| Must-have Claim F1 | N/A |
| Citation F1 | N/A |
| Forbidden Claim Rate | 0.00% |
| Unsupported Claim Rate | N/A |
| 拒答 F1 | N/A |
| 错误拒答率 | 100.00% |
| Span Recall | 0.00% |
| 必要证据组 Recall | 0.00% |
| 论文 Recall | 0.00% |
| 端到端延迟 P50 | 6591.90 ms |
| 端到端延迟 P95 | 6591.90 ms |

## 解释边界

- 当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分
- rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正
- 结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷
