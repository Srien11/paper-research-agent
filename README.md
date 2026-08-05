# 论文研究 Agent

一个面向小型、高质量论文集合的可复现 RAG 与研究 Agent 工程。

当前阶段已经形成从冻结语料、可追溯检索、上下文与短期记忆，到受控 ReAct 研究 Agent 和私有研究界面的本地闭环。

## 核心工程亮点

- **可信检索与引用**：围绕 80 篇、2,286 页英文论文形成 6,252 个检索块和 806 张图表语义，融合中文 BM25 / BGE 与英文科研检索式，经路内和跨语言 RRF、CrossEncoder 重排后生成回答；引用必须命中本轮真实 chunk 白名单。
- **受控 Agent Runtime**：LangGraph ReAct 工作流覆盖规划、工具执行、证据充分性反思、提前终止和有限重规划；19 项研究工具全部受严格 Schema、总时长、调用次数、重复调用和信任等级约束，不注册任意 Shell、Python 或文件系统能力。
- **状态、安全与审批**：SQLite Checkpoint 恢复 thread；笔记、报告和长期记忆写入使用与工具名、参数哈希绑定的一次性审批令牌，避免旧确认复用到新参数。
- **后端智能路由**：前端只提交问题、附件和本地论文库约束；后端模型路由器区分普通聊天、本地 RAG、联网研究、附件问答和文件修改，再由策略层校验合法性，高风险覆盖、删除和外发继续要求确认。
- **隐私可观测性**：事件日志仅记录指纹、耗时、计数、路由和原因码，不记录问题、证据正文或 Provider 原始载荷；论文原文、本地索引、密钥、运行数据库和内部资料均不进入 Git。
- **资源受控部署**：针对 2 核 2G 环境采用单 worker、串行重任务和 850MB systemd 内存上限，将模型推理与重排交给外部服务；当前完整测试集 283 项通过。

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

默认不开启 Agent，原单次 RAG 路径保持不变。启用后有两条彼此独立的 Graph：默认问答仍只用
`search_corpus / get_evidence` 形成可验证引用；Web 中手动开启“动态工具模式”后，模型才可在
固定 19 项扩展工具中逐步选择。两条路径都不是自由工具 Agent；Shell、任意 Python 和任意
网络/文件能力均未注册，写工具必须经过 Runtime 审批。

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

生产 Agent Runtime 现在除核心 `search_corpus / get_evidence` 外，还注册 19 项严格 Schema
工具。核心可引用证据循环仍用原两工具关闭失败；扩展工具通过
`ResearchAgentRuntime.execute_tool` 或完整 LangChain registry 调用，结果不会绕过现有引用验证。

- 本地证据：`get_adjacent_chunks`、`get_paper_metadata`、`trace_evidence_source`、
  `get_paper_outline`、`compare_papers`
- 学术网络：`search_scholarly_sources`、`resolve_paper_identifier`、`get_citation_graph`、
  `check_paper_status`
- 非文本理解：`extract_table`、`inspect_figure`、`extract_equation`
- 计算核验：`calculate`、`analyze_experiment_data`、`verify_claim`、
  `check_reproducibility`
- 成果管理：`save_research_note`、`export_research_report`、`manage_long_term_memory`

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
python scripts/serve_web.py --host 127.0.0.1 --port 8092
```

生产凭据优先复用个人站现有 `ZHIMO_ADMIN_USER`、`ZHIMO_ADMIN_SALT`、
`ZHIMO_ADMIN_HASH` 和 `ZHIMO_PBKDF2_ITERATIONS`；另需独立设置至少 32 字节的
`PRA_WEB_SESSION_SECRET`。本地开发也可改用 `PRA_WEB_USER` 与 `PRA_WEB_PASSWORD`。
这些变量、DashScope Key、论文派生数据、模型缓存与运行数据库均不得进入 Git 或部署代码包。

研究台返回安全字段白名单：中文回答与 claims、引用编号、论文标题、官方链接、页码、版权
分类、短摘录，以及改写状态、入选证据/记忆数量和 Token 预算。它不会返回 PDF 路径、图片
路径、完整 figure JSON、系统提示词、Provider 原始响应或未选中证据全文。部署模板见
`deploy/`，推荐问题见 `configs/web/recommended-questions-v1.json`。

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
