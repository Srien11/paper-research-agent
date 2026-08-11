# 主 Agent 统一入口发布验收 v1

- 验收日期：2026-08-11（Asia/Shanghai）
- 验收基线提交：`79e281b`
- 发布提交：本报告所在 Git revision
- 生产模式：`PRA_MAIN_AGENT_MODE=primary`
- 回滚模式：`PRA_MAIN_AGENT_MODE=legacy`
- 结论：代码、接口、状态、静态检查和本地真实浏览器门禁通过；外部模型生产 smoke
  因未获得“发送问题及本地检索上下文到外部模型端点”的单独授权而未执行。

## 自动化门禁

| 门禁 | 结果 | 摘要 |
|---|---|---|
| `unittest discover -s tests -v` | PASS | 671 tests，31.747s |
| `pytest -q` | PASS | 680 passed，85 subtests，36.71s |
| `ruff check src tests scripts` | PASS | 0 violations |
| `mypy src/paper_research_agent` | PASS | 137 source files，0 errors |
| `git diff --check` | PASS | 0 whitespace errors |
| 浏览器桌面与移动端 | PASS | Edge + Playwright；真实 FastAPI/静态页面，类型化本地运行时夹具 |

测试数量高于改造前 596 项基线。唯一测试警告是 FastAPI TestClient 对当前 httpx 适配层的
弃用提示，不影响结果，也不涉及主 Agent 路由。

## 十类统一入口场景

| ID | 场景 | 自动验收结果 |
|---|---|---|
| A | 普通聊天不回显 | PASS |
| B | 单论文 RAG 保留来源 ID | PASS |
| C | 多论文比较进入固定证据图 | PASS |
| D | Local RAG 后 Dynamic Tools 的混合研究 | PASS |
| E | 附件问答与 conversation 所有权隔离 | PASS |
| F | 文件编辑生成新 attachment | PASS |
| G | 审批等待、批准、拒绝与过期恢复 | PASS |
| H | 相同 request ID 不重复副作用 | PASS |
| I | commit validation 拒绝且 workspace 版本不增加 | PASS |
| J | 断流后状态查询/同 ID 重试不重复执行 | PASS |

金标为 `evaluation/datasets/main-agent-interface-v1.jsonl`。评分覆盖路由正确性、目标连续性、
产物/引用保留、审批恢复、重复副作用、非法提交拒绝和事件契约有效性。报告只记录场景、状态、
计数和门禁结果，不记录论文正文、用户私有问题、完整工具参数、密钥或 Provider 载荷。

## 真实浏览器结果

本地 loopback 服务以 `primary` 模式启动，使用真实生产页面、认证、Origin 校验、统一 API 和
NDJSON 消费逻辑；模型与论文证据替换为严格类型化夹具，整个过程不访问外网。

| 指标 | 结果 |
|---|---:|
| `POST /paper-research/api/agent/runs` | 1 |
| 旧 `/ask`、`/chat/stream`、`/tools/run` 调用 | 0 |
| request ID 合法且完成后清除 pending | 100% |
| `rag_mode=preferred` 透传 | PASS |
| 控制台错误 | 0 |
| 页面错误 | 0 |
| 失败请求 | 0 |
| HTTP 4xx/5xx | 0 |
| 桌面/移动横向溢出 | 0 |

浏览器验收同时发现并修复了移动端 skip link 悬浮遮挡与窄屏品牌换行问题；最终截图已人工
复查，位于本地忽略目录 `data/runtime/web-browser-main-agent-v1/`，不进入 Git。

## 数据落盘与隐私核对

- Conversation Store 是 turn、goal、plan、workspace 和 run 的权威存储。
- 主图 checkpoint 只恢复执行位置，不替代 Conversation Store 的原子提交。
- 相同 request ID 只产生一个 run/turn；缓存重试返回 `run_reused`。
- 审批公开字段仅包含工具名、用途、参数 SHA-256 和过期时间。
- 事件仅包含固定类型、指纹、计数、耗时、状态和原因码；不含问题或证据正文。
- 附件按 conversation 校验所有权；文件编辑返回新 ID，不覆盖输入文件。

## 外部模型 smoke 状态

`scripts/smoke_web_runtime.py` 在受限沙箱中安全失败；请求在尝试启用外部网络权限时被策略
拒绝，因为该验收可能把问题及本地检索上下文发送给外部模型服务。未尝试绕过，也未把该项
伪装为通过。若要补做，操作者必须明确授权这一外发边界，然后在私有环境执行：

```powershell
$env:PRA_MAIN_AGENT_MODE = 'primary'
python scripts/smoke_web_runtime.py "<获准发送的验收问题>"
```

执行记录仍只能保存状态、来源数量、降级标志、耗时和结果摘要，不保存证据正文或密钥。

## 灰度与回滚

1. 首个发布周期保留全部 legacy 兼容代理并观测成功率、提交拒绝、审批恢复、旧接口使用量
   与 P95 延迟。
2. 出现跨会话污染、数据撕裂、重复副作用、引用丢失或非法状态提交时，设置
   `PRA_MAIN_AGENT_MODE=legacy` 并重启服务。
3. 回滚不得删除 checkpoint/Conversation Store，不得手工修改生产 SQLite 绕过门禁。
4. 旧端点使用量连续一个完整发布周期为 0 后，另开删除计划；本次不删除旧接口。

当前发布判断为：**代码与本地门禁 GO；外部模型生产 smoke 待明确授权。**
