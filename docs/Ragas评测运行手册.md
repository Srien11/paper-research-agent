# Ragas 论文研究 Agent 评测运行手册

## 当前基线

- Ragas 固定为 `0.4.3`，避免升级后指标提示词和分数漂移。
- 使用 `ragas.metrics.collections` 新接口；不使用计划在 Ragas 1.0 删除的旧接口。
- `langchain-community` 固定为 `>=0.3.31,<0.4`。Ragas 0.4.3 仍导入旧
  VertexAI 模块，而 `langchain-community` 0.4.x 已删除该模块。
- 当前阈值只用于诊断，`release_gate_enabled=false`，尚未作为发布门禁。
- 默认应用 `configs/evaluation/ragas-silver-quarantine-v1.json`：冻结本轮原始
  GQ001～GQ020 范围，并隔离其中 7 道已确认或高风险银标。原始私有 JSONL 不删除。

## 安装

在项目虚拟环境中安装论文 Agent 与 Ragas 评测依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval,web,agent,ragas-eval]"
```

## 配置

默认配置位于 `configs/evaluation/ragas-v1.json`。API Key 只通过环境变量读取：

```powershell
$env:PRA_RAGAS_API_KEY = '<评测专用 Key>'
$env:PRA_RAGAS_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:PRA_RAGAS_JUDGE_MODEL = 'qwen3.7-plus-2026-05-26'
$env:PRA_RAGAS_EMBEDDING_MODEL = 'text-embedding-v4'
```

`PRA_RAGAS_API_KEY` 留空时回退到 `DASHSCOPE_API_KEY`。建议长期使用独立裁判 Key 和独立
用量统计，并让裁判模型与被测回答模型不同；当前同源模型配置只能作为第一版基线。

## 准备检查

先运行不调用 Agent、不发送模型请求的准备检查：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_with_ragas.py --check
```

检查内容包括：

- Ragas 是否确为 0.4.3，且 collections 指标可导入；
- 配置文件是否符合严格 Schema；
- 本地金标集能否加载，以及可回答/不可回答样本数量；
- 裁判凭据是否已配置。检查输出不包含问题或证据正文。
- 银标冻结范围、隔离数量、稳定题目 ID 和原因码；当前默认应保留 13 题。

如果显式传入其他 `--dataset`，默认隔离清单不会自动套用；可同时使用
`--quarantine-manifest <path>` 指定对应清单。`--include-quarantined` 只供诊断旧结果，
会重新纳入已知问题题目，不应用于正式比较。

## 小规模冒烟

先对两条可回答样本运行，验证本地索引、回答模型、裁判模型和嵌入模型链路：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_with_ragas.py `
  --answerable-only `
  --limit 2
```

确认成功后再运行开发集：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_with_ragas.py
```

该命令当前运行隔离后的 13 题，而不是从后续未审计的 60 道银标中补足题数。

默认产物：

- `data/evaluations/runs/ragas-gold-v1.json`：逐案例安全分数和聚合结果，已被 Git 忽略；
- `reports/Ragas论文研究Agent评测-v1.md`：聚合中文报告，不包含正文。

## 指标含义

| 指标 | 本项目中的含义 |
|---|---|
| `selected_context_precision` | 最终入选引用中，与参考要点相关的上下文排序质量 |
| `selected_context_recall` | 最终引用上下文覆盖参考要点的程度 |
| `response_faithfulness` | 整体回答中的论断能否由全部入选引用支持 |
| `answer_relevancy` | 回答是否直接覆盖用户问题，需要嵌入模型 |
| `citation_faithfulness` | 每个 claim 只对它实际引用的完整 chunk 评分，检查引用错绑 |
| `citation_validity` | claim 中的引用 ID 是否存在于安全来源白名单 |
| `paper_id_precision/recall` | 最终引用论文与 must-have claim 的 `supports` 关系所关联论文集合的确定性重合度 |

这里的 `selected_context` 是生成前最终入选引用所对应的完整实际 chunk，不是 Web 展示用短摘录，也不是原始 BM25/向量召回列表。完整 chunk 根据 `chunk_id` 从运行时使用的本地 `chunks.jsonl` 水化，只在评分进程内存中存在。

评测默认复用一个本地 Agent runtime，并以最多 4 题为一批并发执行 Ragas 评分。全局裁判并发同样限制为 4；批次完成后才加载下一批评分输入，避免题目数增加时正文在内存中无界累积。运行长评测时还应将 `PRA_LOCAL_RETRIEVAL_WORKERS` 与 `PRA_COMPARISON_SEARCH_CONCURRENCY` 同时设为 4，避免复杂题内部六路 ONNX 重排占满系统提交内存。每批结束后命令行只输出安全的完成数量，不输出正文。

在物理内存较小的 Windows 机器上，ONNX Runtime 可能跨批保留内存池。超过 4 题的评测应使用 `--offset` 和 `--limit 4` 将每批放在独立进程中运行，再用 `scripts/merge_ragas_results.py` 合并安全结果；批次 JSON 和合并过程都不包含正文。
原始检索排序仍使用现有 Recall@k、MRR 和 nDCG 评测。

## 隐私与复现

- 问题、回答、claim、完整引用 chunk 仅在进程内存中进入 Ragas，不写入评测结果和报告。
- 输出只保存问题 ID、切片标签、数字分数、耗时、状态和异常类型名。
- 配置、数据集、代码提交号均记录哈希或版本，便于复现实验。
- `internal_research_only` 摘录仍只能发送给数据政策允许的已配置模型端点。
- 正式比较两个版本时，应固定数据集、索引、Ragas 版本、裁判模型和嵌入模型。

## 上线门禁前的工作

当前诊断阈值尚未经过裁判校准。启用 `release_gate_enabled` 前至少完成：

1. 从开发集抽取 30～50 条，由两名审阅者独立评 Faithfulness 和引用正确性。
2. 对分歧案例进行仲裁，比较人工标签与 Ragas 分数。
3. 根据误放率和误杀率重新确定阈值，而不是直接采用配置中的初始值。
4. 使用不同裁判模型复评一轮，报告同源裁判偏差。
5. 对相同版本至少重复三次，报告均值和波动范围。

## 银标生成门禁

`scripts/prepare_gold_candidates.py` 生成的记录仍然只能标记为 `silver_generated`。
生成阶段会确定性拒绝以下已知缺陷：

- claim 提到某篇论文的独有标题标记，却只引用另一篇论文的证据；
- claim 的全部支持证据都是明显损坏的 OCR 或 URL 编码文本；
- claim 使用 `something`、`implied`、“具体指标或维度”等占位式措辞；
- 未被任何 must-have claim 引用的候选 span 不再标为 `required`，而是标为
  `distractor`。

这些门禁只能拦截确定性缺陷，不能替代两名标注者复核和第三人仲裁。
