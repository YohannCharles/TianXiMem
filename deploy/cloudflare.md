# supermem.021130.xyz 公网入口

该入口通过独立、由本地配置管理的 Cloudflare Tunnel 连接服务器上的现有 TianXiMem：

```text
https://supermem.021130.xyz
    → Cloudflare HTTPS → Tunnel supermem → http://127.0.0.1:28088
    → 已运行的 tianxi-app
```

`POST /add`、`POST /search`、`GET /health` 使用原有路径与请求格式，见
[`接口使用说明`](../docs/api-usage.md)。新域名与 Lenovo 域名共享 SQLite 和 Qdrant；
`user_id` 相同就访问同一份记忆。此配置没有另起应用或更换 profile。
接口契约以 PRD §2.1 / §2.2 与 [`contract.md`](../docs/contract.md) 为准。

## 服务器上的配置

| 项 | 位置 / 值 |
| --- | --- |
| Tunnel 名称 | `supermem` |
| Tunnel ID | `407635a5-71bd-4ed9-a5e8-5c39acbde216` |
| DNS | `supermem.021130.xyz` 的代理 CNAME 指向 `407635a5-71bd-4ed9-a5e8-5c39acbde216.cfargotunnel.com` |
| 路由配置 | `/home/hechj/.cloudflared/supermem.yml` |
| systemd 用户服务 | `/home/hechj/.config/systemd/user/supermem-cloudflared.service` |
| Tunnel 凭据 | `/home/hechj/.cloudflared/407635a5-71bd-4ed9-a5e8-5c39acbde216.json`（密钥文件，禁止提交） |

路由配置如下；最后的兜底规则必须保留：

```yaml
tunnel: 407635a5-71bd-4ed9-a5e8-5c39acbde216
credentials-file: /home/hechj/.cloudflared/407635a5-71bd-4ed9-a5e8-5c39acbde216.json
no-autoupdate: true
protocol: http2

ingress:
  - hostname: supermem.021130.xyz
    service: http://127.0.0.1:28088
  - service: http_status:404
```

用户服务已启用，设置 `Restart=always`；`hechj` 的 linger 已启用，退出登录后继续运行，
主机启动时自动启动。应用仍由现有 Docker Compose 管理。模型网关的
`aml-cloudflared.service` 使用另一个 Tunnel，单独管理。

## 日常检查与修改

以 `hechj` 用户执行：

```bash
systemctl --user status supermem-cloudflared.service
journalctl --user -u supermem-cloudflared.service -n 50 --no-pager
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel info 407635a5-71bd-4ed9-a5e8-5c39acbde216
curl --fail --show-error https://supermem.021130.xyz/health
curl --fail --show-error http://127.0.0.1:28088/health
```

修改路由后，先校验再重启此 Tunnel：

```bash
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel ingress validate
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel ingress rule https://supermem.021130.xyz/add
systemctl --user restart supermem-cloudflared.service
```

如修改应用的宿主端口，必须同步更新路由的 `service`。仅撤下新入口时可执行
`systemctl --user disable --now supermem-cloudflared.service`，并在 Cloudflare 删除
`supermem` 的 DNS 记录；该操作不需要停止应用。

## Cloudflare 行为与限制

- 新入口沿用服务自身的鉴权行为，见 [`接口使用说明`](../docs/api-usage.md) §0。
- curl、httpx 与 requests 的默认 User-Agent 已验证可探活；`Python-urllib/3.10`
  被 Browser Integrity Check 拦截为 403 / 1010。urllib 调用方可设置自己的非空
  User-Agent。若需要兼容默认 urllib，在 Cloudflare 的 Rules → Configuration Rules
  添加仅匹配 `http.host eq "supermem.021130.xyz"` 的规则，将 Browser Integrity Check
  设为关闭。现有 Tunnel 登录凭据不具备 Rulesets 权限，尚未应用该规则；
  [Cloudflare 1010 说明](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/)
  和[配置规则 API](https://developers.cloudflare.com/rules/configuration-rules/create-api/)提供对应说明。
- Tunnel 不会消除 Cloudflare 边缘的超时限制；本次只验证了短请求，尚未验证 PRD §2.2
  的最长请求时长。长时间无响应可能遇到 524；有这类请求时应使用已验证支持该时长的入口，
  详见 [Cloudflare 524 说明](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)。
- 独立临时 Named Tunnel 已验证 chunked JSON 的空白心跳可维持长请求，观察与边界见
  [公网接入报告](../eval/reports/deployment-20261012.md)。正式 Add/Search 尚未接入：
  提前发送成功状态后，晚发生的业务异常无法再改为原有非成功状态；平台的断流重试行为
  与 Add 的响应时机仍需核验。单独调用 `/health` 不能延长另一条请求的超时。

DNS 与 Tunnel 的关系见 [Cloudflare 官方说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/routing-to-tunnel/dns/)。
