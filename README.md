# 论文研究 Agent

一个面向小型、高质量论文集合的可复现、证据可追溯 RAG 与研究 Agent 工程。

项目重点不是让模型自由拼接论文结论，而是把论文发现、分格检索、事实编译、引用验证、
状态恢复和高风险工具审批收敛到可测试的确定性边界。当前已经形成从冻结语料、可追溯检索、
上下文与记忆，到三图分层 Agent 编排、事实级证据台账和私有研究界面的本地闭环。

## 核心工程亮点

- **可信检索与引用**：围绕 80 篇、2,286 页英文论文形成 6,252 个检索块和 806 张图表语义，融合中文 BM25 / BGE 与英文科研检索式，经路内和跨语言 RRF、CrossEncoder 重排后生成回答；引用必须命中本轮真实 chunk 白名单。
- **事实级证据编译**：多论文比较由模型输出最小事实单元，代码确定性生成事实 ID、覆盖状态、缺失项和最终证据台账；每个“论文 × 维度 × 原子事实”单元独立提交和重试，成功事实不会因其他单元失败而清空，编译失败也不会伪装成证据不足。
- **三图分层编排**：在原有固定证据图与动态工具图之上新增跨轮次主 Agent 图；主图统一恢复上下文、对齐目标、拆分任务、选择子图、评估结果并原子提交会话状态，两类子图只执行受控任务，不修改主目标与计划。
- **受控 Agent Runtime**：LangGraph ReAct 工作流覆盖规划、工具执行、证据充分性反思、提前终止和有限重规划；18 项扩展研究工具全部受严格 Schema、总时长、调用次数、重复调用和信任等级约束，不注册任意 Shell、Python 或文件系统能力。
- **状态、安全与审批**：SQLite Checkpoint 恢复 thread；笔记、报告和长期记忆写入使用与工具名、参数哈希绑定的一次性审批令牌，避免旧确认复用到新参数。
- **后端智能路由**：前端提交问题、附件和明确的 `rag_mode`（`disabled` / `preferred` / `required`）；关闭时禁止本地检索，优先模式允许本地 RAG、普通聊天和联网研究动态分流，仅本地模式则强制使用论文库。策略层继续校验路由合法性，高风险覆盖、删除和外发仍要求确认。
- **隐私可观测性**：事件日志仅记录指纹、耗时、计数、路由和原因码，不记录问题、证据正文或 Provider 原始载荷；论文原文、本地索引、密钥、运行数据库和内部资料均不进入 Git。
- **资源受控部署**：针对 2 核 2G 环境采用单 worker、串行重任务和 850MB systemd 内存上限，将模型推理与重排交给外部服务；发布门禁要求自动测试不少于 596 项且全部通过。

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
- [x] 框架无关的只读论文搜索与证据读取工具
- [x] LangChain 工具适配与受控 LangGraph ReAct 研究工作流
- [x] 可选启用、带 SQLite State Checkpoint 的受控多步研究 Agent Runtime
- [x] 证据充分性反思、提前终止、重复查询防护与有界重新规划
- [x] 不保存问题、查询和证据正文的 Agent 节点、工具、耗时与 Runtime 拦截日志
- [x] 多论文比较最小事实契约、确定性证据投影与逐单元事务式重试
- [x] 精确金标块、同论文替代块和语义事实三级证据血缘归因
- [x] 20 题开发集与 10 题封存集的完整多论文比较端到端评测

详细安排见[RAG 检索基线实施计划](docs/plans/2026-07-26-RAG检索基线实施计划.md)。

## 最新评测快照

在代码版本 `66810d5` 上，30 题真实多论文比较评测取得 27/30 成功、0 超时；
Top 8 候选论文召回率为 100%，最终目标论文准确率为 90%，比较维度覆盖率为 84.3%，
必要事实覆盖率为 84.5%，已表达事实的引用正确率为 99.1%。要求目标、维度、必要事实、
引用、禁用事实和答案完整性同时满分的严格全条件通过率为 33.3%。

这组结果证明了论文级候选召回、证据隔离和引用校验的稳定性，也暴露出规划契约、
Top 4 正文水化深度和最终覆盖完整性仍是主要瓶颈。完整聚合口径、事实损失漏斗与限制见
[多论文比较最小证据编译器 v2 评测摘要](reports/多论文比较最小证据编译器-v2-评测摘要.md)。

## 主 Agent 统一编排

生产启动首先进入跨轮次主 Agent 主体图（`paper_research_main_v1`），不是先
进入某个执行子图。主图先恢复最近历史、滚动摘要、活动目标、会话任务计划、
远距会话召回与长期记忆，再解释本轮变化、对齐目标和规划任务；只有完成这些
步骤后，才按能力把受控任务派发给普通聊天、本地论文、动态工具、附件问答或
文件编辑执行器。本地论文研究图与动态工具图都是 child executor（子执行器），
不能自行修改主目标、任务计划或会话版本。

浏览器只使用以下统一接口：

- `POST /paper-research/api/agent/runs`：提交带稳定 `request_id`、`message`、
  `rag_mode` 和附件 ID 的请求，返回统一 NDJSON 事件流。
- `GET /paper-research/api/agent/runs/{request_id}`：断流后查询终态或等待状态。
- `POST /paper-research/api/agent/runs/{request_id}/approval`：批准或拒绝原写任务，
  只恢复暂停的 child，不重新解释、对齐目标或规划。

同一 `request_id` 重试会返回持久化结果和 `run_reused`，不会新增 turn 或重复副作用。
旧 `/ask`、`/chat/stream`、`/tools/run` 与 `/tools/approval` 在 `primary` 模式只作为
兼容代理保留，浏览器不得调用；`legacy` 模式则可在紧急回滚时恢复旧实现。

生产命令 `scripts/serve_web.py` 默认使用 `PRA_MAIN_AGENT_MODE=primary`。显式配置
`PRA_MAIN_AGENT_MODE=legacy` 并重启即可回滚；已废弃的
`PRA_MAIN_AGENT_ENABLED` 只为旧部署读取兼容，不应继续写入新环境文件。

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

Web 研究台另由编排层维护统一的 Conversation Store：普通聊天、本地 RAG、附件、网页研究
和动态工具共享同一 conversation ID。每轮无条件准备最近窗口、主题 Episode 与远距用户问题
候选，再由一次结构化 Turn Interpreter 同时判断历史依赖、生成独立问题并规划研究能力；规则
只校验真实 turn ID、会话隔离、能力开关和低置信度澄清。只有独立问题进入中英文论文查询改写。
“使用本地论文知识库”表示本地论文必须参与且可与动态研究组合，“仅依据本地论文回答”则禁止
外部研究。检查器显示最近窗口、远距候选、采用 turn ID/相关度、能力计划和实际检索路径。
清空会话会同时清除共享账本及其 Episode 索引。

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

首版研究工具位于 `paper_research_agent.agent`。`ResearchToolService` 提供框架无关的
`search_corpus` 和 `get_evidence`：前者只返回排名、稳定 ID、页码、哈希和版权分类，后者
只按显式 chunk ID 从本地不可变语料读取正文。搜索结果会再次经过现有血缘适配器校验；
检索元数据与源 chunk 不一致、版权映射缺失或请求越界时关闭失败。

安装 `agent` 可选依赖后，`build_langchain_tools` 将这两个方法包装为固定名称和 Pydantic
参数契约的 LangChain 工具；`build_research_graph` 通过这些工具执行受控
`plan -> reason -> execute_tools -> assess_evidence -> reason/finalize` 循环。Planner 最多产生
6 个只读检索子问题；结构化 Evidence Reasoner 可在证据充分时提前结束，或在证据不足时
提出一条新的本地检索，但不能扩展工具白名单。Graph 同时限制每步证据数、总工具调用次数、
重规划次数和重复查询，并按 chunk ID 去重。

`ResearchAgentRuntime` 负责总超时、`session_id -> thread_id` 映射，并重新核对终态中的动作
序列、工具计数、重规划计数、终止原因，以及正文、哈希、页码和版权分类。通过后，证据才会进入现有
`assemble_context -> answer_context`、引用白名单和回答验证器。SQLite Checkpoint 只保存在
`data/runtime/`，其中可能包含研究 State 和内部证据正文，禁止提交或公开。

```powershell
python -m pip install -e ".[retrieval,web,agent]"
$env:PRA_RESEARCH_AGENT_ENABLED = 'true'
$env:PRA_RESEARCH_AGENT_MAX_STEPS = '4'
$env:PRA_RESEARCH_AGENT_EVIDENCE_PER_STEP = '3'
$env:PRA_RESEARCH_AGENT_MAX_TOOL_CALLS = '12'
$env:PRA_RESEARCH_AGENT_TIMEOUT_SECONDS = '90'
# 可选：让目录、表格、图片和公式工具读取当前解析产物
$env:PRA_SECTIONS_PATH = 'data/processed/<current-build>/sections.jsonl'
$env:PRA_ELEMENTS_PATH = 'data/processed/<current-build>/elements.jsonl'
```

默认不开启 Agent，原单次 RAG 路径保持不变。完整 Agent 架构由一层主编排图和两条执行子图
组成：主图统一维护跨轮次目标、任务计划和路由；固定证据子图只用
`search_corpus / get_evidence` 形成可验证引用；动态工具子图才允许模型在固定 18 项扩展工具中
逐步选择。两条执行子图都不是自由工具 Agent；Shell、任意 Python 和任意网络/文件能力均未注册，
写工具必须经过 Runtime 审批。

Agent 启用后，结构化事件默认写入与 State Checkpoint 同目录的
`data/runtime/agent-events-v1.sqlite3`。每次运行生成独立 `run_id`，覆盖运行开始/完成/失败、
每个 Graph 节点开始/完成/失败、工具调用开始/完成/失败、总超时、工具白名单/预算拦截和
终态输出拒绝。事件只保存组件名、状态、耗时、数量、预算、错误类型、固定原因码，以及问题、
查询、步骤和 thread 的 SHA-256 指纹；不保存 API Key、问题正文、查询正文、chunk ID、证据
正文、Provider 请求/响应或自由文本异常消息。日志写入采用 best-effort，SQLite 暂时不可写
不会改变研究结果或放宽 fail-closed 安全策略。

可按一次运行查看安全轨迹：

```sql
SELECT event_id, event_type, component, name, status, duration_ms, reason_code
FROM agent_events
WHERE run_id = '<run_id>'
ORDER BY event_id;
```

### 扩展研究工具平台

生产 Agent Runtime 现在除核心 `search_corpus / get_evidence` 外，还注册 18 项严格 Schema
工具。核心可引用证据循环仍用原两工具关闭失败；扩展工具通过
`ResearchAgentRuntime.execute_tool` 或完整 LangChain registry 调用，结果不会绕过现有引用验证。

- 本地证据：`get_adjacent_chunks`、`get_paper_metadata`、`trace_evidence_source`、
  `get_paper_outline`
- 学术网络：`search_scholarly_sources`、`resolve_paper_identifier`、`get_citation_graph`、
  `check_paper_status`
- 非文本理解：`extract_table`、`inspect_figure`、`extract_equation`
- 计算核验：`calculate`、`analyze_experiment_data`、`verify_claim`、
  `check_reproducibility`
- 成果管理：`save_research_note`、`export_research_report`、`manage_long_term_memory`

多论文比较不属于动态工具。主路由会把带有多个本地论文 ID 的学术比较交给固定研究图；规划器将
每篇论文与比较维度拆成独立步骤，`search_corpus` 用对应 `corpus_id` 同时约束 BM25 和向量召回。
证据编译模型只返回最小事实、片段 ID、事实需求 ID 和限定词；本地代码再校验论文范围、事实需求、
必要限定词和覆盖状态，确定性生成事实 ID 与最终台账。单元失败时只重试失败 requirement，已提交
单元继续保留；连续编译失败以独立 `compiler_failed` 状态终止，不回退为空台账或证据不足。
最终回答上下文不再携带原始证据正文，而是按计划 requirement 顺序投影经过验证的事实台账与引用。
当前编译器可见性严格隔离不同论文，但同一论文内的证据仍可被该论文的多个维度复用；这是后续按
缺失事实自适应水化并收紧到同维度范围的明确优化项。普通单篇问答仍保持原有全局检索 rank 顺序。
升级前若有尚未完成、下一步仍指向 `compare_papers` 的动态 Graph checkpoint，应新建任务重新执行；
该旧工具名已从严格目录移除，恢复时会按未知工具关闭失败。

工具目录为每项能力固定风险级别、默认超时和最大结果数。本地读取直接执行；学术网络只向
Semantic Scholar 或 Crossref 发送当前检索式/论文标识符，不发送本地 chunk 或论文正文；
计算器只解释算术 AST，实验分析只支持最多 1000 行、20 列和固定统计白名单；笔记、报告、
长期记忆新增、更新、删除必须使用与工具名、参数哈希绑定的 5 分钟一次性审批令牌；搜索和
列举是只读操作，无需审批。记录按作用域隔离，支持内容去重、版本替代、软删除和可选过期；
所有文件和 SQLite 只写入 `data/runtime/`，默认拒绝路径逃逸与静默覆盖。

动态工具 Graph 会先执行只读记忆召回，再进入
`recall_memory -> route -> execute -> route/propose_memory/finalize`；记忆写入路径为
`propose_memory -> execute(仅生成请求) -> interrupt -> 用户批准/拒绝 -> execute/finalize`。只有
用户明确表达“记住、更新记忆、忘记”等意图时才会生成结构化候选，普通问答不会自动写入。
模型只负责
返回结构化 `ToolDecision`，Runtime 再校验工具名、参数 Schema、风险策略、超时、重复调用和
初始计划、重复调用、连续无新证据、工具次数与总时长预算。观察结果额外标注四类信任等级：`citation_evidence`、`research_context`、
`computed_result`、`side_effect`，长期记忆始终属于低信任 `research_context`，只有第一类可作为
论文事实的引用候选；确认结论还必须绑定当前语料中真实存在的不可变 chunk ID。

Owner Web API 提供 `POST /paper-research/api/tools/run` 与
`POST /paper-research/api/tools/approval`。两者都要求登录 Cookie 和允许的 `Origin`；待审批响应
只返回工具名、用途、参数 SHA-256 与过期时间，不返回完整参数、正文、一次性令牌或内部审批 ID。
登录后的 `GET /paper-research/api/memories` 只投影当前有效记录的内容、类型、来源 ID、版本和
生命周期字段；页面可查看记忆，修改和删除仍需通过动态工具审批。
路由冒烟基线位于 `evaluation/datasets/tool-routing-v1.jsonl`，覆盖本地读取、网络读取、受限计算
和三类写入审批场景。

```python
result = await runtime.execute_tool("calculate", {"expression": "(2 + 3) * 4"})

pending = await runtime.execute_tool(
    "save_research_note",
    {"title": "结论", "content": "已确认的研究结论"},
)
token = runtime.approve_tool_request(pending.summary["approval_request_id"])
saved = await runtime.execute_tool(
    "save_research_note",
    {"title": "结论", "content": "已确认的研究结论", "approval_token": token},
)
```

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

## 站长私有可视化研究台

`paper_research_agent.web` 将现有 RAG 包装为常驻 FastAPI 服务：启动时只加载一次
6,252 个 chunks、BM25、FAISS、BGE 和 Reranker，后续问题复用同一运行时。生产环境只
运行一个 Uvicorn worker，并用单查询锁保护本地 CPU 模型；服务仅监听回环地址，由 Nginx
在 `/paper-research/` 下同源代理。

因为完整索引包含 30 篇 `internal_research_only` 论文，首页项目卡可以公开，但真实问答、
引用摘录、英文改写、短期记忆和上下文检查器全部要求站长登录。登录后由服务端生成不可猜测
的 conversation ID，并通过 `HttpOnly + Secure + SameSite=Strict` 签名 Cookie 绑定；
浏览器不能自选 session ID，也不会保存站长密码。POST 请求还会校验精确同源 Origin。

安装 Web 依赖后，以环境变量提供私有路径与凭据并启动：

```powershell
python -m pip install -e ".[retrieval,web]"
$env:PRA_MAIN_AGENT_MODE = 'primary'
python scripts/serve_web.py --host 127.0.0.1 --port 8092
```

启动日志必须显示 `主 Agent Web 启动模式：primary`。若统一入口出现数据撕裂、
重复副作用、引用丢失或持续提交拒绝，执行以下回滚并重启同一服务：

```powershell
$env:PRA_MAIN_AGENT_MODE = 'legacy'
python scripts/serve_web.py --host 127.0.0.1 --port 8092
```

回滚不删除主 Agent checkpoint、Conversation Store（会话存储）或事件库，也不得在
生产 SQLite 中手工改状态。问题修复后复用原数据验证，再显式切回 `primary`。

生产凭据优先复用个人站现有 `ZHIMO_ADMIN_USER`、`ZHIMO_ADMIN_SALT`、
`ZHIMO_ADMIN_HASH` 和 `ZHIMO_PBKDF2_ITERATIONS`；另需独立设置至少 32 字节的
`PRA_WEB_SESSION_SECRET`。本地开发也可改用 `PRA_WEB_USER` 与 `PRA_WEB_PASSWORD`。
这些变量、DashScope Key、论文派生数据、模型缓存与运行数据库均不得进入 Git 或部署代码包。

研究台返回安全字段白名单：中文回答与 claims、引用编号、论文标题、官方链接、页码、版权
分类、短摘录，以及改写状态、入选证据/记忆数量和 Token 预算。它不会返回 PDF 路径、图片
路径、完整 figure JSON、系统提示词、Provider 原始响应或未选中证据全文。部署模板见
`deploy/`，推荐问题见 `configs/web/recommended-questions-v1.json`。

## 可选 MCP 只读工具

MCP（Model Context Protocol，模型上下文协议）只作为现有 Dynamic Tools 子图的工具
provider（提供方），不新增子图，也不进入固定 Local RAG 证据闭环。默认安装和默认运行均
关闭 MCP；安装可选依赖后，由管理员静态配置文件决定准入的 Server 与工具：

```powershell
python -m pip install -e ".[agent,mcp]"
$env:PRA_MCP_ENABLED = 'true'
$env:PRA_MCP_CONFIG_PATH = 'D:\secure-config\mcp-servers.json'
python scripts/serve_web.py
```

示例见 `deploy/mcp-servers.example.json`。示例不含真实路径或密钥；生产配置必须把 Zotero
命令指向本项目固定 Python 环境，把 GitHub 命令指向已审查且固定版本的官方二进制。
GitHub PAT 只通过 `GITHUB_PERSONAL_ACCESS_TOKEN` 继承，禁止写入 JSON、日志或 Git。

所有 MCP 结果都是低信任 `research_context`，不能直接成为论文引用。Server 离线时应用
继续启动，相关工具不进入当次不可变 Registry snapshot（注册快照）；安全事件只记录
Server ID、`ready/degraded`、工具数和 reason code。紧急关闭无需迁移或删除状态：

```powershell
$env:PRA_MCP_ENABLED = 'false'
python scripts/serve_web.py
```

完整安装、升级审查、隐私、监控与回滚步骤见
[MCP 工具接入与运维](docs/MCP工具接入与运维.md)。

## 可中断、可编辑的计划执行

主 Agent 支持暂停、继续、取消、跳过步骤、调整目标和任务顺序、只重试失败步骤，以及单任务时间、调用次数和费用预算。所有编辑都保留已完成步骤及其产物；完整状态机、API 和恢复不变量见 [可中断可编辑计划执行](docs/可中断可编辑计划执行.md)。

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

基础检索评测仍包含开发期单审阅者银标集；论文级候选发现已经补充人工金标基线，多论文比较则
使用 20 题开发集和 10 题封存集执行完整生产链路评测。当前端到端样本仍只有 30 题，生成模型与
模型 Judge（裁判）同源，也尚未进行多次随机重复和第二审阅者盲审，因此这些结果用于内部归因和
版本比较，不应外推为跨语料、跨领域的泛化能力。
模型、正文、片段和索引只保存在本地忽略目录，发布要求见
[检索基线发布门禁](docs/发布门禁.md)。
