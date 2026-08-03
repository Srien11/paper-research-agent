# RAG 可视化真实测试 v1

## 测试范围

- 日期：2026-08-03
- 入口：`RAGRuntime.from_environment()`，与 Web API 使用同一常驻运行时
- 索引：`idx_04be970e1e15cf435e744200`
- 语料：80 篇论文、6,252 个 chunks
- 模型：`qwen3.7-plus-2026-05-26`
- 参数：`temperature=0.1`、`top_p=0.7`、输出上限与上下文预留均为 1,200 Token
- 安全：测试输出不包含 Key、证据正文、图片路径、PDF 路径或 Provider 原始响应

## 启动验证

真实 chunks、FAISS、BGE 与 Reranker 在当前设备完成常驻加载，运行时返回：

```json
{"ready": true, "chunks": 6252}
```

启动过程同时校验了索引 chunk 数、Embedding 模型与 revision，以及 manifest 中记录的
chunks、FAISS 和 SQLite 文件 SHA-256。没有重新分块或生成向量。

## 第一轮真实问题

问题：

> 现有研究中，哪些方法用于评估 RAG 回答的引用正确性与证据忠实度？

英文科研检索式：

> methods for evaluating citation correctness and evidence faithfulness in RAG

回答状态为 `answered`，改写状态为 `success`，没有降级。回答形成 6 条带引用 claim，涉及：

- RAGAS 的忠实度、答案相关性和上下文相关性评估；
- ARES 的轻量评判模型、预测能力推断与少量人工标注；
- RAGVUE 的原子声明、关键实体和时间表达一致性检查；
- RAGChecker、RAGTruth、MEMERAG、HoH、Unanswerability-Eval；
- G-Eval 与 AutoCalib 等 LLM-as-a-judge 方法。

本轮主要来源为：

- `C029`：RAGAs，EACL 2024，页 2–3、7；
- `C015`：ARES，NAACL 2024，页 1–2；
- `C059`：RAGVUE，EACL 2026，页 1–4。

上下文与调用统计：

| 指标 | 结果 |
| --- | ---: |
| 纳入证据 | 8 |
| 因预算省略证据 | 2 |
| 纳入记忆轮次 | 0 |
| 上下文估算 Token | 6,685 / 8,192 |
| 回答预留 Token | 1,200 |
| Provider 输入 / 输出 Token | 5,427 / 409 |
| 生成延迟 | 8,485 ms |
| 调用尝试 | 1 |
| 查询 / 回答审计 | 均成功落盘 |

## 第二轮连续追问

问题：

> 它们分别需要人工标注吗？

短期记忆没有直接充当证据，而是把检索问题解析为：

> 当前问题：它们分别需要人工标注吗？<br>
> 上一轮研究问题：现有研究中，哪些方法用于评估 RAG 回答的引用正确性与证据忠实度？

英文科研检索式：

> RAG evaluation methods citation correctness evidence faithfulness manual annotation required

回答状态仍为 `answered`，改写成功且无降级。系统区分了 RAGAS 的 reference-free 设计、
ARES 所需的少量人工偏好验证数据、WikiEval 的人类判断，以及 RAGVUE 的自动化定位。

上下文与调用统计：

| 指标 | 结果 |
| --- | ---: |
| 纳入证据 | 7 |
| 因预算省略证据 | 3 |
| 纳入记忆轮次 | 1 |
| 省略记忆轮次 | 0 |
| 上下文估算 Token | 6,906 / 8,192 |
| 回答预留 Token | 1,200 |
| Provider 输入 / 输出 Token | 5,714 / 304 |
| 生成延迟 | 7,522 ms |
| 调用尝试 | 1 |
| 查询 / 回答审计 | 均成功落盘 |

## 结论

真实链路已验证：中文问题、Qwen 英文科研改写、中英双路检索、RRF、英文重排、上下文预算、
引用白名单、中文结构化回答和短期记忆可以在同一个常驻 Web 运行时连续工作。第二轮明确证明
“上下文”是每次临时组装的快照，而短期记忆只提供追问指代所需的上一轮信息；事实仍由第二轮
重新检索的论文证据支持。
