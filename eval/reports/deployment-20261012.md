# 公网接入验证

本主题卷记录公网 API 的传输与部署验证。不是 AML Smoke / Full，不调用真实 Add/Search、
模型、SQLite 或 Qdrant；不计为检索收益或官方评测成绩。

## Cloudflare Tunnel 的 JSON 空白心跳（2026-10-12）

**结论**：在独立的临时 Named Tunnel 上，同一响应中的 JSON 换行心跳能维持超过默认
Cloudflare 读取超时的连接，并且最终响应仍能解析为普通 JSON。仅验证成功路径，
没有上线到正式 Add/Search。

### 测试设置

- `cloudflared`：`2026.9.1`，`protocol: http2`，与正式入口使用同一版本和 Tunnel 协议。
- 原站：本机回环上的临时 `ThreadingHTTPServer`，监听系统分配的端口。
- 单独创建临时 Named Tunnel 和代理 CNAME；测试后删除 DNS、Tunnel 并停止临时服务。
- 两个并行 GET 请求：均模拟处理耗时 `150s`，最终正文为 `{"data":[]}`。
- 心跳组：立即发送 HTTP 200 和响应头，处理期间每 `15s` 发送一个换行字节。
- 对照组：处理完毕前不发送响应头或响应体。
- 两组最终响应使用 `Content-Type: application/json`、
  `Cache-Control: no-store, no-transform`、`Transfer-Encoding: chunked`，无 `Content-Length`。
- 客户端：httpx，`Accept-Encoding: identity`，直接连接 Cloudflare，
  `trust_env=False`；等待超时 `185s`。

### 观察

| 路径 | HTTP 状态 | 结束耗时 | 正文与解析 |
| --- | --- | --- | --- |
| 有心跳 | 200 | 151.5s | 十个换行 + `{"data":[]}`；JSON 解析结果为 `{"data": []}` |
| 无心跳 | 524 | 126.0s | Cloudflare HTML 错误页 |

心跳组收到响应头的时间为 `1.5s`，随后各块到达时间为：
`1.5, 16.5, 31.5, 46.5, 61.5, 76.5, 91.5, 106.5, 121.5, 136.5, 151.5s`。
前十块各 `1` 字节；最后一块为完整 JSON 的 `11` 字节。

原始观察：[cloudflare-heartbeat-20261012.json](runs/cloudflare-heartbeat-20261012.json)
（本机产物，按目录规则忽略，不含请求原文或密钥）。

先前的临时 Quick Tunnel 和首个 Named Tunnel 探针经过本机出站代理时遇到连接/TLS 错误，
未开始长请求对照，不将它们算作心跳失败。改为客户端直连后，得到上表结果。
上述临时资源均已清理，正式 `supermem-cloudflared.service` 保持运行。

### 正式 API 尚待解决的边界

1. 本次只覆盖 GET 与合成结果，没有验证正式 POST 的请求体、Pydantic 校验与平台客户端。
2. 提前发送 HTTP 200 后，晚发生的异常无法再映射为 409/500/503。
   断开未完成的流是否触发平台重试，尚未核实；不能用 HTTP 200 的错误 JSON 替代原有错误响应。
3. Add 的持久化与立即可搜索约束以 PRD §2.1 为准，重试与错误处理以 PRD §2.2 / §15
   及 `service/errors.py` 为准。心跳不得被视为业务成功，完整成功 JSON 只能在处理完成后发送。
   平台是否接受提前发送响应头，需要进一步核验。
4. 没有验证 `30min` 请求、持续运行、多并发、压缩开启或平台的总请求超时。
   心跳只能处理链路的无数据等待，不能延长客户端的绝对总时限。

### 一手资料

- [Cloudflare 超时说明](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)：
  默认 Proxy Read Timeout，以及仅 Enterprise 可调整的说明。
- [cloudflared 2026.9.1 源码](https://github.com/cloudflare/cloudflared/blob/2026.9.1/connection/connection.go)：
  `shouldFlush` 对无 `Content-Length` 或 chunked 响应启用及时刷新，
  因此本次没有将 API 改成 SSE 来维持连接。
