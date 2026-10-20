# Cloudflare Tunnel Ingress

The public HTTPS endpoint is provided by a Cloudflare Tunnel:

```text
https://supermem.021130.xyz
  -> Cloudflare HTTPS
  -> Tunnel supermem
  -> http://127.0.0.1:28088
  -> tianxi-app
```

The tunnel forwards the existing `/add`, `/search`, and `/health` paths. It does not create another application or profile. The public and internal endpoints share the same SQLite database and Qdrant collection.

## Configuration

| Item | Value |
| --- | --- |
| Tunnel | `supermem` |
| Tunnel ID | `407635a5-71bd-4ed9-a5e8-5c39acbde216` |
| Route config | `/home/hechj/.cloudflared/supermem.yml` |
| systemd user unit | `/home/hechj/.config/systemd/user/supermem-cloudflared.service` |
| Credentials | `/home/hechj/.cloudflared/407635a5-71bd-4ed9-a5e8-5c39acbde216.json` |

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

The credentials file is secret. Never commit it.

## Checks

```bash
systemctl --user status supermem-cloudflared.service
journalctl --user -u supermem-cloudflared.service -n 50 --no-pager
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel info 407635a5-71bd-4ed9-a5e8-5c39acbde216
curl --fail --show-error http://127.0.0.1:28088/health
curl --fail --show-error https://supermem.021130.xyz/health
```

Validate route changes before restarting:

```bash
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel ingress validate
cloudflared --config /home/hechj/.cloudflared/supermem.yml tunnel ingress rule https://supermem.021130.xyz/add
systemctl --user restart supermem-cloudflared.service
```

If the host application port changes, update the tunnel service URL.

## Constraints

- The service itself has no authentication; the tunnel does not add application identity.
- Cloudflare Browser Integrity Check may reject clients with an empty or unusual User-Agent.
- Cloudflare edge timeouts still apply. Long Add or Search requests may exceed the proxy timeout.
- Verify long-request behavior before relying on this ingress for Full runs.
