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
- [ ] 806 张图表的视觉模型摘要与图文混合索引重建
- [ ] 多步研究 Agent

详细安排见[RAG 检索基线实施计划](docs/plans/2026-07-26-RAG检索基线实施计划.md)。

## 数据边界

原始 PDF、生成索引、运行数据库和密钥不进入 Git。仓库通过环境变量引用本地冻结语料：

```powershell
Copy-Item .env.example .env
$env:PRA_CORPUS_DIR = 'D:\agent-study\kf\research_collection'
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
python scripts/validate_corpus.py --corpus-dir D:\agent-study\kf\research_collection
python scripts/parse_corpus.py --corpus-dir D:\agent-study\kf\research_collection

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

图片语义入库分为三个阶段。裁剪阶段不调用视觉模型，也不会上传图片：

```powershell
python scripts/crop_figures.py `
  --elements "$env:BUILD_DIR\elements.jsonl" `
  --corpus-dir "$env:PRA_CORPUS_DIR" `
  --output data/processed/figure-semantics-v1
```

视觉摘要阶段要求已安装并配置可用的 `z-ai` 命令行；`model-id` 必须填写实际
视觉模型及版本，不能用占位名称：

```powershell
python scripts/summarize_figures.py `
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
  --token-budget 8192 --output-reserve 1024
```

组装顺序固定为可信系统规则、对话历史、当前用户问题/任务状态、检索证据。历史、任务
状态和论文正文均属于低信任数据；证据使用 canonical JSON 封装，不能覆盖系统规则或
触发工具。预算计算覆盖最终消息模板并预留输出空间，证据只按完整 chunk 加入，不截断
引用血缘。

## 工程原则

- 每个结论最终绑定 `paper_id / element_id / page / evidence_span`。
- 所有实验记录语料、解析器、chunk、embedding 和检索配置版本。
- 先评测检索，再引入生成与 Agent。
- 索引可重建，数据导入幂等。
- 受限论文只用于本地研究，不能由仓库重新分发。
- 冻结清单中的选题理由、挑战提示和问题创意不得进入索引。

全量文本解析已覆盖 2286/2286 页，并通过跨记录完整性审计和 14 篇、42 页
视觉抽检。806 个图注对应区域已经裁剪为本地图片，但尚未调用视觉模型生成全量摘要，
当前生产索引仍是文本索引，不能将本阶段称为完整多模态 PDF 解析。

检索评测使用开发期单审阅者银标集，不是密封测试集。下一步应由第二名审阅者独立标注
论文相关性与证据片段，解决分歧后冻结人工金标测试集；在此之前不得据此宣称泛化效果。
模型、正文、片段和索引只保存在本地忽略目录，发布要求见
[检索基线发布门禁](docs/发布门禁.md)。
