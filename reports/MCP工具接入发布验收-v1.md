# MCP 工具接入发布验收 v1

## 结论

本地发布门禁全部通过。MCP 只读底座、第一方 Zotero stdio Server、GitHub 官方只读 Server
的 Host 白名单、统一 Registry snapshot、动态 Router、结果清洗和安全降级满足实施计划 v1。
GitHub live smoke 涉及外部数据出站，当前未获明确授权，因此未执行；真实 SDK fake stdio
协议验收已完成，不能用它冒充 GitHub live 成功。

## 验收环境

| 组件 | 版本 |
|---|---:|
| Python | 3.12.13 |
| mcp | 1.29.0 |
| jsonschema | 4.26.0 |
| pydantic | 2.13.4 |
| httpx | 0.28.1 |
| langchain | 1.3.14 |
| langgraph | 1.2.10 |

示例配置 `deploy/mcp-servers.example.json` 的 SHA-256 为
`2CC333CF23FA70CBF681D54217786720984579C439522AC9FE79FAF8E9B42184`。该文件只含占位路径，
不含真实 command、PAT 或其他密钥。

## 准入工具

Zotero 本地只读：

- `zotero__search_items`
- `zotero__get_item`
- `zotero__list_collections`
- `zotero__get_annotations`
- `zotero__get_attachment_metadata`
- `zotero__get_fulltext`

GitHub Host 只读白名单：

- `github__search_repositories`
- `github__search_code`
- `github__get_file_contents`
- `github__list_commits`
- `github__list_releases`
- `github__issue_read`

内置 `EXTENDED_TOOL_SPECS` 仍固定为 18 项，名称和原执行语义不变。MCP 关闭时 Registry 恰好
只包含这 18 项。

## 自动门禁结果

| 门禁 | 结果 |
|---|---|
| 聚焦真实 stdio 端到端 | 2/2 通过；initialize、list、call、超限、error、crash、PID 回收 |
| unittest | 724 项通过，0 失败 |
| pytest | 733 项及 105 subtests 通过，0 失败；2 条非阻断第三方 warning |
| Ruff | `src tests scripts` 零问题 |
| Mypy | 148 个源码文件零问题 |
| `git diff --check` | 通过 |

pytest warning 分别来自 Starlette 对当前 TestClient/httpx 组合的弃用提示，以及 FastMCP 内部
lifespan 前向引用提示；均不影响本次行为或类型门禁，没有被忽略为测试失败。

## 路由与安全指标

金标 `mcp-tool-routing-v1.jsonl` 使用 16 条合成场景，不含真实私人 Zotero 内容、PAT、私有
仓库名或论文全文。fake-server runner 只保存 case ID、公开路由、reason code、布尔计数和耗时。

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| route_accuracy | 1.0 | ≥ 0.90 |
| unsafe_tool_call_rate | 0.0 | = 0 |
| unregistered_tool_execution_rate | 0.0 | = 0 |
| write_attempt_execution_rate | 0.0 | = 0 |
| local_rag_diversion_rate | 0.0 | = 0 |
| offline_graceful_rate | 1.0 | = 1.0 |
| prompt_injection_follow_rate | 0.0 | = 0 |
| output_bounded_rate | 1.0 | = 1.0 |

## 本地 smoke 与进程清理

- 第一方 Zotero MCP Server：`ready`，`tools/list` 发现 6 项，Host 准入 6 项。
- 本地 Zotero API：只用合成键执行 loopback 读取；结果为确定性
  `insufficient/mcp_server_error`，未伪造条目成功，也未输出响应正文。
- shutdown：MCP manager 状态为 `closed`；真实 fake Server PID 在关闭后不可存活。
- crash：fixture 子进程主动退出后状态转为 `degraded`，同一生命周期不自动无限重连。
- GitHub live smoke：未执行。原因是缺少明确的数据出站授权和专用只读 PAT。

## 安全矩阵

| 边界 | 验收结果 |
|---|---|
| 配置来源 | 仅管理员文件；关闭时不打开文件、不启动进程 |
| 进程启动 | 绝对 command；不用 shell；环境变量按名称显式继承 |
| 工具发现 | 只取本地白名单与 `tools/list` 交集；远端额外工具忽略 |
| Router 目录 | 只用本地 description 和有界 Schema；一个 run 使用同一快照 |
| 参数 | Host 在调用前按 JSON Schema 2020-12 再验证；额外字段拒绝 |
| Schema | 限制字节、深度、节点、properties 和 `$ref` |
| 结果 | 限制 item、字符串和总字节；敏感字段递归移除 |
| 非文本内容 | 图片、音频、resource、resource link 第一版拒绝 |
| 信任 | MCP 不能成为 `citation_evidence`；本地 ToolSpec 覆盖远端伪造值 |
| Zotero | 固定 loopback `/api/users/0`；不跟随重定向；不访问附件文件 |
| GitHub | 示例同时要求 `--read-only`、`--tools` 和 Host 白名单；PAT 不入配置 |
| 事件与异常 | 不含问题、参数、正文、stderr、环境值或密钥 |
| 离线降级 | 单 Server 不可用不影响内置工具、其他 Server 或应用启动 |
| 关闭 | 独立资源均尝试释放；真实子进程 PID 回收已验证 |

## 数据出站边界

本次自动测试、fake stdio 和 Zotero loopback smoke 均未访问外部网络。未来 GitHub live smoke
只能发送经批准的合成公共仓库查询，且必须使用专用最小权限只读 PAT；不得发送本地论文正文、
Zotero 条目/批注/全文、对话历史、长期记忆、本地路径或 Provider 原始数据。

## 回滚

无需删除 checkpoint、Conversation Store 或事件库：

```powershell
$env:PRA_MCP_ENABLED = 'false'
python scripts/serve_web.py
```

重启后不会读取 MCP 配置或启动子进程，动态工具恢复为固定 18 项。回滚不授权手工编辑运行
SQLite；修复后必须重跑本报告列出的全部本地门禁再重新启用。
