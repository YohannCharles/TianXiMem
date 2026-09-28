# deploy/ — 部署与环境

**PRD**：§6.3（Qdrant server 模式）、§7.3（版本钉死）、§15（工程与部署）

## 要写什么

```text
compose.yaml    **整栈**：检索服务（`app`）+ Qdrant server，镜像 tag 钉死
Dockerfile      检索服务的镜像（多阶段；**上下文是仓库根**）
.env.example    服务器只需要的那几项（`cp` 成 `deploy/.env`，**不进 git**）
```

**这个目录最要紧的两件事**：① 把 **Qdrant 的版本**钉死；② 把**服务做成单进程容器**。
其余都是围绕这两条的说明。

> ### ⚠ 这套容器**还不是提交口径**
>
> `configs/submit.yaml` **尚未存在**（见根 `CLAUDE.md` 的状态表），`default.yaml` 的
> `models.embedder` 仍是开发期的 `Qwen/Qwen3-Embedding-8B`。
> ⇒ **这个镜像的用途是"服务能起来、能连通、契约合规"**，不是"这就是提交时跑的那一套"。
> Step 5 换模型 ⇒ **换向量维度 ⇒ 整个集合重建**（§4 的 runbook），
> 届时**同一份 compose 不用改**——变的是 `configs/`、`.env` 与集合。
> ⛔ **别把现在这个镜像当成"可以发 Full 的版本"**。

> ⚠ `compose.yaml` 里现在**有两个服务**，所以：
> `qdrant-up` 用的是 `up -d qdrant`（**带服务名**），而 `qdrant-down` 是 `stop qdrant`
> 不是 `down`（后者按项目拆，会把 `app` 一起拆掉）。整栈上下线是 `make up` / `make down`。

---

## 0. 容器形态：一条命令起整栈

```bash
cp deploy/.env.example deploy/.env     # 填 AML_EMB_* 等；reranker 可选
make up                                # = docker compose -f deploy/compose.yaml up -d --build
make deploy-check                      # 在容器里跑契约预检（14 条，真 HTTP）
```

**已验证**（2026-09-28，全绿）：整栈起来后 `app` 进 `healthy`；容器内自跑预检 **14/14**；
`/data` 卷由 `app`（uid 10001）持有、`tianxi.db` 重启后行数不变；`--network none` 下
`o200k_base` 可用（BPE 已烘进镜像）；绑定 `0.0.0.0:28088` 后**从别的机器打得通**、
而 Qdrant 只绑回环、外网连不上；**容器 → 宿主机回环上的网关**（`host.docker.internal`）
整条链路跑通（真 Add 200 + 真 Search 200）。

### 部署到一台固定 IP:端口 的服务器（**照着抄这一段**）

以 `10.193.135.28:28088` + 网关在同机 `127.0.0.1:9002` 为例。

**① `deploy/.env` 全文**（其余键一个都不用填）：

```bash
TIANXI_PROFILE=default                          # 集合 memories（⚠ 不是 local）
TIANXI_BIND=0.0.0.0                             # 要让别的机器访问就必须这样
TIANXI_PORT=28088                               # 对外端口
AML_EMB_BASE_URL=http://host.docker.internal:9002/v1
AML_EMB_API_KEY=<网关的 key；网关不校验也要填个非空的>
```

> ⛔ **别写 `http://127.0.0.1:9002/v1`**——容器里的 `127.0.0.1` 是**容器自己**，
> 实测报 `Connection refused`，看起来像"网关挂了"。**同机的正确写法是
> `host.docker.internal`**（compose 里已经加了 `extra_hosts: host-gateway`）。
> 跨机器才用 `http://10.193.135.28:9002/v1`，且**前提是网关不是只绑回环**。
> 两条路都记得：`https` → `http`（内网通常没证书），**`/v1` 保留**。

**② 起 + 验**：

```bash
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml ps                       # 期望 app 是 (healthy)
docker compose -f deploy/compose.yaml exec -T app \
    python eval/smoke/preflight.py --base-url http://127.0.0.1:8000    # 期望 14/14
# 从**另一台机器**验（这才是平台看到的那条路）：
curl -s http://10.193.135.28:28088/health                      # 期望 {"status":"ok"}
```

**③ 常见坑（每一条都实测过）**：

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| Add 报 500 / connection refused | `.env` 里写了 `127.0.0.1:9002` | 改 `host.docker.internal:9002` |
| 从别的机器 connection refused，**日志里什么都没有** | `TIANXI_BIND` 还是 `127.0.0.1` | 设 `0.0.0.0` |
| `Bind for 0.0.0.0:6333 failed: port is already allocated` | 老版本 compose 起的 Qdrant 容器还占着 0.0.0.0 | 先 `docker rm -f tianxi-qdrant`，或用 `TIANXI_QDRANT_PORT` 换端口 |
| Add 报 4xx 且信息里有模型名 | `configs/default.yaml` 的 `models.embedder`（`Qwen/Qwen3-Embedding-8B`）与本机网关服务的不一致 | 网关忽略 `model` 字段就没事；校验的话得改 config（重建镜像） |
| 服务起不来、日志说缺 `embed.base_url` | `deploy/.env` 没建或键名写错 | 见上面 ① |

> ⚠ **compose 文件不能挪出 `deploy/`**：它的 `build.context` 是 `..`，**相对 compose 文件所在目录**。
> 拷到别处会报 `lstat .../var/deploy: no such file or directory`。要换位置就连 `Dockerfile` 与
> 仓库根一起拷（或者改 `context`）。

### 把镜像弄到服务器上（两条路）

拓扑与开发机**不同**（D12 记的是"服务 + Qdrant + harness 都在本机"那套开发形态）：
服务器上只跑**服务 + Qdrant**，harness 与答案/裁判生成留在本地。

| 要求 | 为什么 | 怎么确认 |
| --- | --- | --- |
| **能连到自建网关**（embedding 必需 / reranker 可选） | 三个模型都不在本机（§2.3 / D12）⇒ **没有本地推理，也就不需要 GPU** | 下面那条命令，**期望 `200`** |
| **不需要外网**（除了上面那个网关） | tiktoken 的 BPE 已烘进镜像；依赖已在构建期装完 | `docker run --rm --network none --entrypoint python tianxi-am:local -c "import tiktoken; tiktoken.get_encoding('o200k_base')"` |
| **Qdrant 与真源都有持久盘** | SQLite 不可重建（§6.3） | `docker volume ls`；**备份见 §4 的表** |
| **8000 端口的暴露面想清楚** | **本服务没有任何鉴权**——`user_id` 是隔离字段，不是认证（§2.2） | `TIANXI_BIND` 默认 `127.0.0.1`（只有本机）；要对外就自己加 TLS + 反代 |
| **镜像 tag 不要用 `latest`** | 与 Qdrant 同源的理由：Full 只有 2 次、一旦接受即版本冻结 | 打 tag 时带上日期或 commit，别覆盖 `tianxi-am:local` 就上线 |

> ⚠ **服务器上跑 `default` profile，不是 `local`**：`local` 会把集合换成 `memories_dev`
> （开发期与提交期隔离用）。**提交期必须是 `memories`**——换了集合名 = 换了集合，
> 旧的 point 会**静默留在旧集合里**（V9）。

**网关可达性怎么确认**（⚠ 要**带鉴权**，否则 `401` 会被误读成"不可达"）：

```bash
docker compose -f deploy/compose.yaml exec -T app python -c "
import os, httpx
u = os.environ['AML_EMB_BASE_URL'].rstrip('/')
r = httpx.get(u + '/models', headers={'Authorization': 'Bearer ' + os.environ['AML_EMB_API_KEY']}, timeout=15)
print('GET /models ->', r.status_code, '（期望 200）')"
```

同上，reranker 那一路（**可选**，401/超时都只是降级、不是故障）：

```bash
docker compose -f deploy/compose.yaml exec -T app python -c "
import os, httpx
u = os.environ['TIANXI_RERANKER_BASE_URL'].rstrip('/')
r = httpx.get(u + '/models', headers={'Authorization': 'Bearer ' + os.environ['TIANXI_RERANKER_API_KEY']}, timeout=15)
print('GET /models ->', r.status_code)"
```

### 把镜像弄到服务器上（两条路）

```bash
# 路 A：在服务器上构建（需要服务器能访问 PyPI + ghcr.io 拉 uv 二进制）
git clone <repo> && cd TianXi_AM
cp deploy/.env.example deploy/.env      # 填值
make up                                 # 内部就是 up -d --build

# 路 B：本地构建、导出、scp 过去（服务器不需要外网，只需要 docker）
make image-build
docker save tianxi-am:local | gzip > tianxi-am.tar.gz
scp tianxi-am.tar.gz deploy/ docker-compose.yml server:/opt/tianxi/
# 服务器上：
gunzip -c tianxi-am.tar.gz | docker load
docker compose up -d          # 注意：不要带 --build，否则它会想重新构建
```

| | 路 A | 路 B |
| --- | --- | --- |
| 服务器要求 | 能拉 PyPI + ghcr.io | **只要能 `docker load`** |
| 镜像与源码一致 | ✅ 自动 | ⚠ **靠你自己保证**：`docker save` 的那一份必须就是 `git status` 里那一份 |
| 适合 | 有外网的常规服务器 | **内网机 / 出网受限**（本项目的服务器很可能属于这一类） |

> ⚠ **路 B 要连 `deploy/` 一起拷**（compose、`.env`），但**不要拷 `deploy/.env.example`
> 就以为完事**——真值在 `.env` 里。而且**别把本地 `.env` 直接 scp 上去**：
> 里面的 `TIANXI_SQLITE_PATH` / `TIANXI_QDRANT_URL` 是开发机的值。
> 服务器上的 `.env` **从 `.env.example` 起**，只填密钥。
>
> ⚠ **镜像 tag 别用 `latest`**：与 Qdrant 同源的理由——Full 只有 2 次、一旦接受即版本冻结。
> 打 tag 时带上日期或 commit（`tianxi-am:2026-09-28-4f56bcc`），下次要回退才有得回。

---

## 1. 为什么必须 server 模式（§6.3）

> **Qdrant local 模式会静默丢弃 payload 索引**——`create_payload_index` 只打**一行警告**就返回，且其数据格式与 server 不兼容。

而本项目的 `user_id` / `session_id` / `event_time` **三个筛选全部依赖 payload 索引**。

**所以 local 模式不是"性能差一点"的替代品，是"过滤静默失效"。** 没有降级方案。

> ✅ **docker 已就位**：Docker Desktop 29.8.0（WSL2 后端，`desktop-linux` context）。**daemon 走 Windows 命名管道，`docker` 命令直接从 Git Bash 可用**；发布端口转发到主机 ⇒ **`localhost:6333` 直接可用**，`.env` 默认值不用改。

---

## 2. 为什么版本要钉死（§7.3）

`{rrf: {weights, k}}` 的可用性是有版本门槛的：

| 参数 | 自哪个版本起 |
| --- | --- |
| `k` | **v1.16.0** |
| `weights` | **v1.17.0** |

⇒ **要求 ≥ v1.17.0**，且**部署时钉死**。

**与 §2.2 的可复现性同源**：**Full 只有 2 次、一旦接受即版本冻结**，而融合行为的任何变化都会改变排名 ⇒ 标签浮动会让"这个分数是哪个版本跑出来的"变成不可回答的问题。

**`compose.yaml` 里的 tag 就是这条要求的落地点**——它必须是完整版本号（`qdrant/qdrant:v1.17.0`），**不能用 `latest`、不能用大版本号浮标**。

---

## 3. 集合配置 —— 是**客户端**设置，不在 compose 里

| 项 | 落在哪 |
| --- | --- |
| 集合分片数 / 命名向量 / payload 索引 / payload 内容 / 写入 `wait=true` | [`../src/tianxi_am/store/CLAUDE.md`](../src/tianxi_am/store/CLAUDE.md)（**一处声明**） |

**本目录只管起服务。** 集合是客户端建的——拓扑本身**已经是规格**（单集合 + `user_id` tenant 过滤 + `is_tenant` 索引 + 分片 1 + payload 索引先于写数据），原文见 [`../docs/config-reference.md`](../docs/config-reference.md) §8。

---

## 4. Step 5 的重建 runbook（§2.3 / §7.4 / §16）

**换 embedding 模型时，集合必须按新维度重建。** 顺序不能反：

```text
1. 停服务（避免写入与重建交错）
2. 备份 SQLite 文件  ← 真源。Qdrant 可重建，SQLite 不可
3. 作废 embedding 缓存——开发期模型的向量对新模型无意义
4. 用新维度建新集合（分片 1、命名向量、**先建 payload 索引**）
5. 从 SQLite 的正文全量重新 embed 并写入（wait=true）
6. 起服务，**重跑 T2**（§12.1 R1 对冲 2）
```

**容器形态下的落点**（`compose.yaml` 的 `app` 服务）：

| 步骤 | 容器里怎么落 |
| --- | --- |
| 1 | `docker compose -f deploy/compose.yaml stop app`（⚠ **`stop` 而不是 `down`**——`down` 是按项目拆的，会顺手把 Qdrant 也拆了） |
| 2 | `docker compose ... exec app tar -c -C /data . > backup.tar`。⚠ **先停服务再拷**：SQLite 在 WAL 模式下直接拷文件可能拿到不一致的快照 |
| 3 | 缓存与真源**在同一个卷** `/data/embed_cache` ⇒ `rm -rf` 那一个子目录即可（**别删 `/data` 本身**） |
| 4–5 | `T2` 的转储走 [`../tools/reindex.py`](../tools/reindex.py)；换 profile（`TIANXI_PROFILE`）会换集合名 |

**第 2 步不能省**：Qdrant 是**派生读存储**，可从 SQLite 全文重建；**SQLite 是真源，丢了就没了**。

**第 3 步不能省**：缓存键不含模型标识 ⇒ 不主动作废就会**静默命中**旧向量（症状只是"检索结果很差"，**不报错**）。原因与可重建性对照表见 [`../var/CLAUDE.md`](../var/CLAUDE.md)。

> **⚠ 一处 R1 类风险（§6.4）**：本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` 不同，**本地量出的"单请求能装多少对"不能直接搬到线上**——**第 6 步之后必须重新量一次**（[`../docs/open-questions.md`](../docs/open-questions.md) E7）。

---

## 5. 24×7 运行就绪

单进程 HTTP 服务，**必须能 24×7 连续运行**——**Full run 持续 0.5–2 天**。

| 检查项 | 出处 | 容器里由谁落地 |
| --- | --- | --- |
| `--workers 1`（**理由见 [`../src/tianxi_am/service/CLAUDE.md`](../src/tianxi_am/service/CLAUDE.md)**："放开多 worker 需要的验证一件都没做"） | §15 → **D25** | `ENTRYPOINT` 写死 + `TIANXI_WORKERS=1`；`--workers N` 仍被 `assert_single_process()` 拦（**已在容器里实测拦下**） |
| Qdrant 与 SQLite 的文件都在**持久卷**上——SQLite **随 run 归档** | §15 | `qdrant_storage`（可重建，**不是备份对象**）/ `tianxi_data:/data`（**唯一备份对象**） |
| `latency/query` 有观测——**接近 30 分钟上限就削减 agent 轮数** | §14 | ——（v1 没有 agent；见 §17.1 的 S8） |
| 单请求最长 30 分钟，**预算充足但不是无限** | §2.2 | `stop_grace_period: 60s`——**默认 10s 会拦腰砍断在途请求** |
| 平台探活必须答得上 | S4 | `HEALTHCHECK` 打 `/health`（**不碰下游**：Qdrant 或网关抖动不该让容器被判不健康） |
| 24×7 能自己回来 | §15 | `restart: unless-stopped` |

> **⚠ 一条开发环路的单点风险（D12）**：answer / judge / embed / rerank **四条都打同一个网关**，而 harness 评测时会同时驱动 Add/Search（embed + rerank）与答案/裁判生成。**必须有客户端并发上限**，否则**排队超时会伪装成"模型变差了"**（[`../docs/open-questions.md`](../docs/open-questions.md) V8）。

---

## 6. 一条不能做的事

**不要为了让 Qdrant 好部署而退回 local 模式**，也**不要**把 `user_id` / `session_id` / `event_time` 的过滤改到客户端做——后者等价于"检索全部、再在内存里筛"，会**破坏隔离字段的语义**（§2.2），而且随数据量增长不可控。

**local 模式的失败是静默的**（只打一行警告），所以"跑起来没报错"**不构成**它能用的证据。
