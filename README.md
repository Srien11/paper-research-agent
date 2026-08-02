# 论文研究 Agent

一个面向小型、高质量论文集合的可复现 RAG 与研究 Agent 工程。

当前阶段聚焦于把冻结论文语料转换成可追溯、可评测的检索基线。Agent 编排、反思和产品界面将在检索基线稳定后迭代，避免把解析、检索和推理误差混在一起。

## 当前里程碑

- [x] 冻结语料论文级数据契约
- [x] 语料清单校验器
- [x] 全 2286 页确定性结构化解析与元素 ID
- [x] 可追溯语义片段与稳定 chunk ID
- [x] 基线 A：纯向量检索
- [x] 30 条单审阅者银标诊断查询
- [x] 基线 B：BM25 + 向量混合检索
- [x] 基线 C：混合检索 + 重排模型
- [x] 分层、预算受控且可引用的 RAG 上下文组装
- [x] 图片信息契约、806 张图表裁剪与图文混合检索链路
- [x] 806 张图表的视觉模型摘要与图文混合索引重建
- [x] 中文问题到英文科研检索式的双路在线检索编排
- [x] 私人研究模式下的结构化回答生成与严格引用验证
- [x] 24 小时、按 session 隔离的本地短期记忆与多轮追问上下文
- [ ] 多步研究 Agent

详细安排见[RAG 检索基线实施计划](docs/plans/2026-07-26-RAG检索基线实施计划.md)。

## 数据边界

原始 PDF、生成索引、运行数据库和密钥不进入 Git。仓库通过环境变量引用本地冻结语料：

```powershell
Copy-Item .env.example .env
$env:PRA_CORPUS_DIR = 'D:\path\to\research_collection'
```

当前冻结版本：

```text
llm-eval-reliability-v1.0.0-2026-07-26
80 篇论文：核心集 60，挑战集 20
```

## 本地验证

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/validate_corpus.py --corpus-dir D:\path\to\research_collection
python scripts/parse_corpus.py --corpus-dir D:\path\to\research_collection

# 以下 BUILD_DIR 替换为本地最新解析产物目录
python scripts/build_chunks.py `
  --elements "$env:BUILD_DIR\elements.jsonl" `
  --sections "$env:BUILD_DIR\sections.jsonl"
python scripts/build_retrieval_index.py `
  --chunks data/processed/chunks/chunks.jsonl
python scripts/search.py "How is hallucination evaluated?" `
  --variant C --chunks data/processed/chunks/chunks.jsonl
python scripts/evaluate_retrieval.py `
  --chunks data/processed/chunks/chunks.jsonl
```

中文生产查询使用独立的双路入口，不改变原有 A/B/C 基线，也不需要重新分块或生成向量：

```powershell
$env:DASHSCOPE_API_KEY = '<本机 Key>'
$env:PRA_CORPUS_DIR = 'D:\path\to\research_collection'
python scripts/search.py "哪些方法能降低 RAG 在 TruthfulQA 上的幻觉？" `
  --bilingual `
  --chunks data/processed/chunks/chunks.jsonl
```

完整问答可以用一条命令串起查询改写、双路检索、上下文组装、回答生成和引用验证：

```powershell
python scripts/rag.py "哪些方法能降低 RAG 在 TruthfulQA 上的幻觉？" `
  --chunks data/processed/chunks/chunks.jsonl `
  --corpus-dir "$env:PRA_CORPUS_DIR" `
  --output data/runtime/example-answer.json
```

连续追问需要显式复用同一个安全 `session_id`；不传该参数时仍是完全无状态的单轮问答：

```powershell
python scripts/rag.py "BEIR 基准包含哪些检索任务？" `
  --session-id beir-review-01 `
  --chunks data/processed/chunks/chunks.jsonl `
  --corpus-dir "$env:PRA_CORPUS_DIR"

python scripts/rag.py "它和 MTEB 有什么区别？" `
  --session-id beir-review-01 `
  --chunks data/processed/chunks/chunks.jsonl `
  --corpus-dir "$env:PRA_CORPUS_DIR"
```

短期记忆不是整份模型上下文。它只在本地 SQLite 中保存最近问题、通过引用验证的
`claims.text`、来源 chunk ID 和版权分类；不会保存论文证据正文、旧的 `[E1]` 标签、完整
回答、模型请求或原始响应。默认 24 小时过期、每个 session 最多 20 轮，每次最多选 6 轮
且不超过 1200 个保守估算 Token。配置见 `configs/memory/short-term-v1.json`。

模型上下文是每次调用临时重新组装的快照：可信 system 规则、筛选后的低信任记忆、当前
问题、本轮新检索证据和输出约束。记忆只帮助解析“它、上述方法、前者”等指代，不能充当
事实证据；当前回答仍必须引用本轮重新检索到的论文 chunk。memory-aware 查询的改写缓存和
查询审计明文也被限制为最多 1 天，并在读取或打开数据库时物理清理过期记录。

收到查询后，中文 `BM25 + BGE` 召回会与 Qwen 英文科研检索式改写并行启动；中英文
各自先做一次路内 RRF，再做跨语言 RRF，最后只使用英文改写查询执行一次英文
Reranker。改写总截止时间默认 2 秒，超时、网络错误、限流或无效 JSON 均返回中文混合
召回结果，不经过英文 Reranker；没有配置 Key 时同样安全降级。配置见
`configs/retrieval/bilingual-qwen-v1.json`。

改写缓存和查询审计分别保存在 `data/runtime/query-rewrite-v2.sqlite3` 与
`data/runtime/query-audit-v1.sqlite3`，不与索引元数据共库。成功改写 90 天内直接命中，
365 天内只在 API 失败时作为过期缓存降级；审计中的中英文查询明文保留 30 天，之后仅
保留查询哈希、模型/提示词版本、各阶段排名、分数和延迟。审计不写证据正文、图注、
`figure_json`、绝对路径、请求/响应体或密钥。

查询改写请求只包含固定改写提示词和当前用户问题，不上传论文正文、图片摘要或命中证据。
传入 `--corpus-dir`（或设置 `PRA_CORPUS_DIR`）后，结果会附带命中论文的
`storage_class`。双路结果若没有加载版权映射，可以在本地查看排名，但后续上下文组装会
拒绝继续，避免把 `internal_research_only` 边界默认为可公开。

图片语义入库分为三个阶段。裁剪阶段不调用视觉模型，也不会上传图片：

```powershell
python scripts/crop_figures.py `
  --elements "$env:BUILD_DIR\elements.jsonl" `
  --corpus-dir "$env:PRA_CORPUS_DIR" `
  --output data/processed/figure-semantics-v1
```

视觉摘要支持百炼实时 API 和 `z-ai` 命令行。百炼接入使用创建 API Key 时一并显示的
业务空间专属 API Host；密钥只通过环境变量读取，不能写入命令、日志或仓库：

```powershell
$env:DASHSCOPE_API_KEY = '<本机新建或重置后的 Key>'
$env:DASHSCOPE_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
python scripts/summarize_figures.py `
  --provider dashscope `
  --candidates data/processed/figure-semantics-v1/figure_candidates.jsonl `
  --model-id qwen3.7-plus `
  --model-id qwen3.7-plus-2026-05-26 `
  --workers 4
```

多个 `--model-id` 按顺序使用。仅当平台明确返回
`AllocationQuota.FreeTierOnly` 时才切换下一个模型；普通网络错误、限流和无效 JSON
会在当前模型重试。若允许免费额度用尽后按量付费，应在百炼控制台关闭“免费额度用完
即停”，平台会自动按“免费额度 → 资源包 → 节省计划 → 按量付费”扣减，此时通常无需
配置第二个模型。命令结束会输出各模型的实际输入、输出和总 Token；每成功一张即原子
更新 `figures.jsonl`，重跑可断点续传。`--workers 4` 会并发处理 4 张图片；若限流频繁，
可改为 `--workers 2`，程序会遵循 `Retry-After` 并指数退避。

未配置 `DASHSCOPE_BASE_URL` 时默认使用上面的北京共享端点；若能从控制台取得业务空间
专属 API Host，仍优先使用专属端点。JSON 模式默认不设置输出 Token 硬上限，避免在
JSON 对象中途截断；摘要长度由提示词约束。

也可以使用原有 `z-ai` 命令行；`model-id` 必须填写实际视觉模型及版本，不能用占位名称：

```powershell
python scripts/summarize_figures.py `
  --provider zai-cli `
  --candidates data/processed/figure-semantics-v1/figure_candidates.jsonl `
  --model-id "<实际视觉模型及版本>"
```

生成 `figures.jsonl` 后，将图片摘要作为“一图一块”合并进文本块，再重建本地索引：

```powershell
python scripts/build_chunks.py `
  --elements "$env:BUILD_DIR\elements.jsonl" `
  --sections "$env:BUILD_DIR\sections.jsonl" `
  --figures data/processed/figure-semantics-v1/figures.jsonl
python scripts/build_retrieval_index.py `
  --chunks data/processed/chunks/chunks.jsonl
```

图片向量文本只包含图名、图片类型、原始图注、视觉摘要和关键发现。坐标、相对路径、
置信度、模型与提示词版本作为检索元数据保存，不参与语义相似度计算。当前实际状态和
全量审计结果见[图片语义入库状态](reports/图片语义入库状态-v1.md)。

检索运行结果可以在调用大模型前组装为安全的上下文预览：

```powershell
python scripts/assemble_context.py `
  --run data/evaluations/example-retrieval-run.json `
  --chunks data/processed/chunks/chunks.jsonl `
  --token-budget 8192 --output-reserve 1200 `
  --output data/runtime/example-context.json
```

组装顺序固定为可信系统规则、必要对话历史、低信任短期记忆、当前用户问题/任务状态、
检索证据。历史、记忆、任务状态和论文正文均属于低信任数据；记忆与证据分别使用
canonical JSON 封装，不能覆盖系统规则或触发工具。预算计算覆盖最终消息模板并预留输出
空间；如果空间冲突，先丢弃最旧记忆，至少保护可容纳的最高排名新证据。

`scripts/answer.py` 是调试生成边界的低层入口；它可以把已经组装好的上下文送入私人研究
回答模块：

```powershell
python scripts/answer.py `
  --context data/runtime/example-context.json `
  --output data/runtime/example-answer.json
```

回答模型固定为 `qwen3.7-plus-2026-05-26`，使用 `temperature=0.1`、`top_p=0.7`、
`max_tokens=1200` 并关闭思考。模型只返回结构化 `claims + citation_ids`；本地验证器依据
`AssembledContext.citations` 白名单校验引用并追加 `[E1]` 标记，未知引用、截断输出或非法
JSON 均关闭失败。没有可用证据时在本地直接返回“证据不足”，不调用 API。

`private_research` 是当前唯一回答输出模式。它允许将已经加载版权分类的少量命中片段发送
给百炼，但不会上传整篇 PDF，也不会把上下文、证据正文、模型原始响应或答案正文写入审计
库。`data/runtime/answer-audit-v1.sqlite3` 只保存模型、Token、延迟、引用/chunk ID、版权分类
和答案哈希。最终回答 JSON 只包含中文答案、结构化 claims、最小引用元数据和调用统计；不
提供公开导出模式。v1 审计只记录已验证回答和本地证据不足结果；Provider 超时、非法 JSON
或引用校验失败不会写库，也不会伪装成“证据不足”。

## 工程原则

- 每个结论最终绑定 `paper_id / element_id / page / evidence_span`。
- 所有实验记录语料、解析器、chunk、embedding 和检索配置版本。
- 先评测检索，再引入生成与 Agent。
- 索引可重建，数据导入幂等。
- 受限论文只用于本地研究，不能由仓库重新分发。
- 冻结清单中的选题理由、挑战提示和问题创意不得进入索引。

全量文本解析已覆盖 2286/2286 页，并通过跨记录完整性审计和 14 篇、42 页
视觉抽检。806 个图注对应区域已裁剪并由 `qwen3.7-plus` 生成结构化视觉摘要；当前统一
索引包含 5,446 个正文块和 806 个图片语义块，共 6,252 个 384 维向量。图片采用“视觉
模型转语义文本，再由文本 Embedding 入库”的方案，并非直接对图片像素生成向量。

检索评测使用开发期单审阅者银标集，不是密封测试集。下一步应由第二名审阅者独立标注
论文相关性与证据片段，解决分歧后冻结人工金标测试集；在此之前不得据此宣称泛化效果。
模型、正文、片段和索引只保存在本地忽略目录，发布要求见
[检索基线发布门禁](docs/发布门禁.md)。
