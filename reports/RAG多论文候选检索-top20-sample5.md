# RAG 多论文候选检索小样本诊断

- Candidate Paper Recall@20（macro）：70.00%
- 显式编号解析准确率：100.00%
- 摘要缺失回退论文数：2

| 问题 | 标注论文 | 候选论文 | Recall |
|---|---|---|---:|
| `GQ009` | C017, T008 | T012, T007, C042, C011, T019, C017, C023, C004, C044, C035, T001, C018, C021, C009, C005, C057, T009, C058, C050, C059 | 50.00% |
| `GQ016` | C012, T017 | C012, C032, T013, C004, C038, C002, T005, C005, C009, C024, T011, C035, C030, C010, C059, C023, C037, C021, C044, T018 | 50.00% |
| `GQ017` | C044, T006 | T013, C037, C054, T012, C025, T011, T010, T005, T018, C038, C012, C057, C049, T016, C017, C033, C039, C001, C044, T014 | 50.00% |
| `GQ032` | C007, C012 | C007, C003, C047, C010, C037, C012, T008, C059, T017, C004, C016, C035, T012, C043, T004, C019, C060, C038, C036, C034 | 100.00% |
| `GQ035` | C014, C049 | C014, C049, C019, C037, T012, C003, C048, T008, C007, C035, T006, C059, C021, T019, T004, C043, C017, C060, C002, C004 | 100.00% |

## 解释边界

- Silver labels are diagnostic only. GQ016 is closer to single-paper identification, while GQ017 is closer to cross-paper multi-hop QA; neither receives hard-coded IDs.
