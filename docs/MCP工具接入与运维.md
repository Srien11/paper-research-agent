# MCP 工具接入与运维

## 1. 范围与不变量

当前 MCP（Model Context Protocol，模型上下文协议）能力只进入既有 Dynamic Tools 子图，
不新增 Graph，不修改主 Agent capability 枚举，也不向 Local RAG 子图注册任何 MCP 工具。
固定 18 项内置工具仍由 `BuiltinToolProvider` 提供；只有管理员配置白名单与 Server
`tools/list` 的交集进入一次不可变 Registry snapshot（注册快照）。

以下边界不可通过配置放宽：

- 仅本地 `stdio`，不支持远程 HTTP、OAuth 或在线安装。
- command 必须是绝对可执行文件路径，禁止 `cmd /c`、PowerShell、`bash -c`、`sh -c`。
- MCP 工具只能是 `local_read/network_read`，且不需要审批。
- MCP 结果只能是 `research_context/computed_result`，不能是 `citation_evidence`。
- 远端 description 不进入 Router；远端 Schema 先检查大小、深度、字段数和 `$ref`，参数再由
  Host 使用 JSON Schema 2020-12 复验。
- text 返回值只是字符串数据。图片、音频、resource 和 resource link 当前关闭失败。
- 结果递归移除 token、authorization、api_key 等字段，并受 item、字符串与总字节限额约束。

## 2. 安装与配置

安装 Agent 和 MCP 可选依赖：

```powershell
python -m pip install -e ".[agent,mcp]"
```

复制 `deploy/mcp-servers.example.json` 到代码仓库之外的受限目录，替换占位绝对路径，但不要
填写任何 token。配置文件额外字段、相对 command、重复 ID/工具名、未知 transport 或非法
命名空间都会让应用启动失败；这属于管理员配置错误，不会静默降级。

`public_name` 默认保持 `<server_id>__<normalized_remote_name>`。如果两段合法名称组合后超过
128 个字符，系统会要求使用确定性的有界别名；校验错误会给出应填写的完整别名，也可以用
`paper_research_agent.agent.mcp.config.mcp_public_name` 根据原始 `server_id` 和 `remote_name`
离线生成。原始两段名称仍分别保存在配置与注册表中，不会因别名而丢失。

生产启用命令：

```powershell
$env:PRA_MCP_ENABLED = 'true'
$env:PRA_MCP_CONFIG_PATH = 'D:\secure-config\mcp-servers.json'
python scripts/serve_web.py
```

`PRA_MCP_ENABLED=false` 时程序不会打开配置文件、导入 MCP SDK 或启动子进程，动态目录仍只有
固定 18 项工具。

## 3. Zotero 本地 Server

Zotero Local API 固定为 `http://127.0.0.1:23119/api/`（等价 `localhost`），user ID 固定为
`0`。服务只实现六项读取工具：`search_items`、`get_item`、`list_collections`、
`get_annotations`、`get_attachment_metadata`、`get_fulltext`。它不监听 TCP、不实现写请求、
不跟随重定向，也不访问会返回 `file://` 的附件文件端点。

Zotero 需要在“设置 → 高级”启用“允许本机其他应用与 Zotero 通信”。未启用返回 403，未运行
或超时均映射为固定 reason code；查询、响应正文和本地路径不会进入日志。官方明确说明该端口
无认证，只能保持在 loopback，禁止端口转发或外部暴露：
<https://www.zotero.org/support/dev/web_api/v3/local_api>。

## 4. GitHub 官方只读 Server

只使用经过审查并固定版本的 `github/github-mcp-server` 官方二进制或固定镜像 digest：
<https://github.com/github/github-mcp-server>。禁止 `latest`、`npx @latest`、未固定容器标签和
启动时自动下载。升级时先在隔离环境重新执行 `tools/list`，比较名称与 Schema，再更新配置和
测试；不能把旧白名单自动套到新版本。

Server 参数必须同时包含 `stdio`、`--read-only` 和逐工具限制；Host 仅准入：

```text
search_repositories
search_code
get_file_contents
list_commits
list_releases
issue_read
```

为该 Server 创建独立、最小权限、只读 PAT，只通过环境变量提供：

```powershell
$env:GITHUB_PERSONAL_ACCESS_TOKEN = '<从安全凭据存储注入，不写入脚本或 JSON>'
```

按组织策略定期轮换；撤销旧 PAT 后重启进程。GitHub README、Issue、源码与评论全部是不可信
研究上下文，不能覆盖系统规则或触发未计划工具。

## 5. 状态、故障与观察

配置错误会阻止启动；单个 Server 的进程缺失、初始化失败、超时、工具缺失或 Schema 漂移是
软失败。应用继续启动，故障 Server 的工具不进入快照，其他 Server 与内置工具不受影响。

本地安全事件 `mcp_server_status` 只包含：

- Server ID；
- `ready/degraded` 投影；
- 准入前发现的工具数量；
- `mcp_server_unavailable`、`mcp_startup_timeout`、`mcp_tool_missing`、
  `mcp_schema_rejected` 等固定 reason code。

事件不含 command、环境变量值、stderr、查询、参数或响应。工具调用超时后该 Server 在本次
进程生命周期内变为 `degraded`，不会在同一 run 中无限重连。正常 shutdown 反向关闭 MCP
session/transport/子进程与学术 HTTP client；部分关闭失败仍继续释放其他资源。

## 6. 发布检查与外部数据边界

本地发布门禁使用真实 SDK stdio fake Server，不访问网络。Zotero smoke 只验证本机 Server
能启动以及 Zotero 未运行时的确定性降级，不把查询或文献内容写入报告。

GitHub live smoke 会把测试查询发送到 GitHub，必须另行获得明确的数据出站授权并提供专用
只读 PAT。未授权时保持“不执行”，不能以公共网络请求或个人 PAT 代替；fake stdio 协议验收
仍应完成并在发布报告中标记 live smoke 未执行。

## 7. 紧急回滚

关闭 MCP 并重启即可回到固定 18 工具，不删除 checkpoint、Conversation Store 或事件库：

```powershell
$env:PRA_MCP_ENABLED = 'false'
python scripts/serve_web.py
```

回滚不会迁移数据，也不授权手工编辑 SQLite。修复配置或 Server 后，先运行全量本地门禁，再
显式重新启用。
