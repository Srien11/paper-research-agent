# 论文研究 Agent

一个面向小型、高质量论文集合的可复现 RAG 与研究 Agent 工程。

当前阶段聚焦于把冻结论文语料转换成可追溯、可评测的检索基线。Agent 编排、反思和产品界面将在检索基线稳定后迭代，避免把解析、检索和推理误差混在一起。

## 当前里程碑

- [x] 冻结语料论文级数据契约
- [x] 语料清单校验器
- [ ] 全 2286 页结构化解析与稳定 chunk ID
- [ ] 基线 A：纯向量检索
- [ ] 标准问题、标准论文与标准证据片段
- [ ] 基线 B：BM25 + 向量混合检索
- [ ] 基线 C：混合检索 + 重排模型
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
```

## 工程原则

- 每个结论最终绑定 `paper_id / element_id / page / evidence_span`。
- 所有实验记录语料、解析器、chunk、embedding 和检索配置版本。
- 先评测检索，再引入生成与 Agent。
- 索引可重建，数据导入幂等。
- 受限论文只用于本地研究，不能由仓库重新分发。
- 冻结清单中的选题理由、挑战提示和问题创意不得进入索引。

当前 `parse_quality_status` 只表示下载阶段的抽页可解析性检查通过，不代表 2286
页已经完成 RAG 级结构化解析。全量逐页解析属于下一里程碑。
