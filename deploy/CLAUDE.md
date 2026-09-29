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

> ### ⚠ 跑哪一套由 `TIANXI_PROFILE` 决定，而 **default 不是提交口径**
>
> `configs/submit.yaml` **已建**（2026-09-28，只换了模型名 `text-embedding-v4`），
> 但它的取值要靠 `TIANXI_PROFILE=submit` 才生效；`.env.example` 里默认写的仍是 `default`
> ——那一份的 `models.embedder` 是开发期的 `qwen3-embedding-8b`。
> ⇒ **镜像本身两种口径都能跑**，差别只在 profile 与 `AML_EMB_*` 指向哪。
> ⬜ **Step 5 尚未完成**：由模型派生的阈值**还没重标定**，换模型后**整个集合要重建**
> （§4 的 runbook），届时**同一份 compose 不用改**——变的是 `configs/`、`.env` 与集合。
> ⛔ **别把现在这个镜像 + `default` 当成"可以发 Full 的版本"**；也**别用 `submit` 去连开发网关**
> （会串缓存坐标，见 [`../docs/decisions.md`](../docs/decisions.md) **D27**）。⚠ D27 那条
> "两者同为 1024 维"的前提**已随 2026-09-28 网关迁移失效**（开发期现在 **4096** 维）。

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
| `sqlite3.OperationalError: unable to open database file`，容器**启动即退** | 用了**绑定挂载**（`-v /宿主/路径:/data`）——容器以 `app`（**uid 10001**）运行，而那个目录属于宿主用户 | `chown -R 10001:10001 /宿主/路径`（或 `chmod 777`）。⚠ **命名卷不会有这个问题**——Docker 首次挂载时会继承镜像里 `/data` 的属主，所以 compose 默认那条路是好的 |
| `Bind for 0.0.0.0:6333 failed: port is already allocated` | 老版本 compose 起的 Qdrant 容器还占着 0.0.0.0 | 先 `docker rm -f tianxi-qdrant`，或用 `TIANXI_QDRANT_PORT` 换端口 |
| Add 报 4xx 且信息里有模型名 | `configs/default.yaml` 的 `models.embedder`（现在是 `qwen3-embedding-8b`）与本机网关服务的不一致 | ⚠ 网关（vLLM 直服）**会校验** `model`：必须与 `/v1/models` 列出的 id 逐字相同，否则 404（改 config 后重建镜像，或挂配置目录，见 §0.5） |
| 服务起不来、日志说缺 `embed.base_url` | `deploy/.env` 没建或键名写错 | 见上面 ① |

> ⚠ **compose 文件不能挪出 `deploy/`**：它的 `build.context` 是 `..`，**相对 compose 文件所在目录**。
> 拷到别处会报 `lstat .../var/deploy: no such file or directory`。要换位置就连 `Dockerfile` 与
> 仓库根一起拷（或者改 `context`）。

### 0.5 不改镜像改配置：挂一个 `TIANXI_CONFIG_DIR`

**阈值 / 权重 / 开关都在 `configs/*.yaml` 里，而它们是烘进镜像的**（有意为之：配置文件是被评审、
被 diff 的产物，不是运行时输入）。要在**不重建镜像**的前提下改一项，挂一个配置目录：

```bash
# ① 把整个 configs/ 拷出来，改你要改的那一项（**必须整个拷**：加载器读
#    <dir>/default.yaml + <dir>/<profile>.yaml，两个文件都得在）
cp -r configs/ /srv/tianxi/configs/
vim /srv/tianxi/configs/default.yaml        # 例如 rerank.enabled: true

# ② compose 里加两行（或写进 override 文件）
#    volumes:  - /srv/tianxi/configs:/cfg:ro
#    environment:  TIANXI_CONFIG_DIR: /cfg
```

> ⚠ **挂载后 `configs/` 就与镜像里那份脱钩了**——镜像升级不会带上新的配置，
> 而**没有人会发现**（服务照常起来、只是跑的是旧阈值）。⇒ 要么把这份目录纳入版本管理、
> 要么只在临时排查时用。**别让它变成一个没人记得的隐式偏离。**
>
> ⚠ `configs/` 里的键**不认识的会被拒绝**（拼错不会静默忽略），所以改坏了会在启动时响亮失败。

#### 具体例：**改 rerank 开关**

`rerank.enabled` 在**基线**里是 `true`（`configs/default.yaml`，2026-09-28 起——D8 把排序定为主线），
开发期由 `configs/local.yaml` 覆盖成 `false`（单题 100 篇要 5–12s、本地串行跑题 ⇒ 整轮墙钟约 3×）。
⇒ **在部署机上想临时改它**：按 §0.5 挂一份配置目录，改对应的那一份 `yaml`。
⚠ 打开它**两处都得配**，只改 yaml 或只配 `.env` 都不生效：

```bash
# ① 配置：`<profile>.yaml` 里 rerank.enabled（按 §0.5 挂目录，或重建镜像）
#    ⚠ 挂的是哪个 profile 就改哪个文件：default → default.yaml，submit → submit.yaml
# ② .env 加三行：
TIANXI_RERANKER_BASE_URL=http://host.docker.internal:9002/v1   # ⚠ 同 embedding 那条规矩：不是 127.0.0.1
TIANXI_RERANKER_API_KEY=<key>
TIANXI_RERANKER_MODEL=Qwen3-Reranker-4B                        # 只进 run record，端点忽略它
```

**怎么确认真的生效**（三层，从便宜到贵）：

| 检查 | 期望 |
| --- | --- |
| `docker compose logs app \| grep -i rerank` | 配全了**没有** WARNING；`rerank.enabled=false` 会打一条 info |
| 打一次 `/search`，看网关那侧有没有收到 `POST /v1/rerank` | 收到 ⇒ 开关通了 |
| 线格式 | 请求 `{"model","query","documents"}` → 响应 `{"results":[{"index","score","text"}]}` |

> ✅ **2026-09-28 实测过整条路**：容器 + 挂载配置（`rerank.enabled: true`）+ 假 reranker，
> RRF 原始顺序 `[问题1, 问题0]`、假 reranker 给 `问题0` 更高分 ⇒ **Search 返回 `[问题0, 问题1]`**，
> 顺序确实被重排。同一轮还验出：**绑定挂载要 `chown 10001:10001`**（见上面那张坑表）。
>
> ⚠ **reranker 是本项目里唯一不限模型的组件、也是唯一"敢花算力"的地方**（§2.3）——
> 打开它是个**质量取舍**，代价是每题 5–12s（100 篇）。`rerank.timeout_seconds` 默认 30s，
> 超时会**降级回 RRF 顺序**（不报错）。这条对照是 §13 的 **A3**，两臂见 `configs/runs/a3-{on,off}/`。

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

### 0.6 打开**请求原文采集**（S6）—— 服务器上照着做

**目的**：把官方发来的 `/add` / `/search` **原样**记下来，用来核验**官方真实的 `request_id` 形状**
——那是 **S6** 唯一没在本地验过的一环，而 D25 的位置模型整个押在它上面。
**配置语义、四条纪律、一行里有哪些字段**一处声明在
[`../docs/config-reference.md`](../docs/config-reference.md) §12 与
[`../src/tianxi_am/service/capture.py`](../src/tianxi_am/service/capture.py)。
**本文件只写"怎么在服务器上做"。**

> ⚠ **它是代码改动，不是配置改动** ⇒ §0.5 那条"挂配置目录"的路**在这里不适用**
> （挂目录只换 yaml，换不掉代码）。**服务器必须先拿到带这一层的新镜像。**

| 步 | 做什么 | 命令 / 判据 |
| --- | --- | --- |
| **①** | 拿到带这一层的代码 | **路 A**（服务器能出网）：`git pull` 到含 `src/tianxi_am/service/capture.py` 的 commit（`git log --oneline -- src/tianxi_am/service/capture.py` 查得到）· **路 B**（内网）：本地 `make image-build` → `docker save` → `scp` → `docker load`（见上一节） |
| **②** | 打开开关 | **路 A**：在**服务器**的仓库里就地改 `configs/default.yaml` → `capture: enabled: true`。**别提交它**——它是个临时诊断开关，一提交下次 `git pull` 就撞冲突（未提交状态反而会在 pull 时**拦住你**，这是好事）。**路 B**：改**本地**那份再 `make image-build`（yaml 是烘进镜像的） |
| **③** | 重建 + 起 | **路 A**：`docker compose -f deploy/compose.yaml up -d --build` · **路 B**：`docker load` 新镜像后 `up -d`（**不要带 `--build`**）。`ps` 期望 `app (healthy)` |
| **④** | **核对采集真的开了** | `docker compose -f deploy/compose.yaml logs app \| grep 请求采集` ⇒ 期望 `请求采集已开启：…/data/capture/requests.jsonl（上限 … 字节，写满即停）`。⚠ 若看到 **ERROR「打不开文件」**：采集已**自动关闭**，服务照常跑——去查 `TIANXI_CAPTURE_PATH` 与 `/data` 卷属主（坑表里那条 `chown 10001:10001`） |
| **⑤** | 跑你要跑的（冒烟一次 / 一轮 Smoke） | 见 [`../docs/submission.md`](../docs/submission.md) §1 |
| **⑥** | 取文件 | `docker compose -f deploy/compose.yaml cp app:/data/capture/requests.jsonl ./requests.jsonl` |
| **⑦** | 读它（下表） | |
| **⑧** | **关回去** | 路 A：`git checkout -- configs/default.yaml` → `up -d --build`；路 B：本地改回 `false` 再走一遍镜像。⛔ **别用 `down -v`**：那个卷同时装着唯一不可重建的 `tianxi.db` |

**⑦ 怎么读**——一行一次请求，`chunk_ordinal` 就是"服务用**当前**正则解析 `request_id` 的结果"：

```bash
python -c "
import json
for line in open('requests.jsonl', encoding='utf-8'):
    e = json.loads(line)
    if e.get('kind') == 'req':
        print(e['path'], e['status'], e['chunk_ordinal'], e.get('request_id'))"
```

| 看到什么 | 说明什么 | 下一步 |
| --- | --- | --- |
| 每一行的 `chunk_ordinal` 都是数字 | 官方 id 确实带 `chunk-<n>`，我们的正则认得 | **S6 收口**：D25 的现状是对的，什么都不用改 |
| 有行是 `null`，而那个 `request_id` 尾部**确实多了一段** | 我们的正则**太窄**（锚定末尾） | 放宽 `ingest.chunk_ordinal_pattern`（**配置项，不必改代码**）——但**先把样本留档**再动 |
| 有行的 `request_id` 里**根本没有 chunk 序号** | D25 的立论基础没了 | ⛔ **停下来**：那是**决定**不是**实现**（见 [`../docs/open-questions.md`](../docs/open-questions.md) 的 S6） |

> ⚠ 三条别踩的：① `/health` **刻意不记**（平台探活会刷屏，且它零核验价值）；
> ② `authorization` **只记"在不在"、绝不记值**；③ 文件里是**官方评测原文**
> ⇒ **不进 git、不传第三方**；`capture.max_bytes`（默认 64 MiB）写满即停，**别指望它记一整场 Full**。

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

### 同一套 runbook 的**另一个触发条件**：网关迁移（2026-09-28）

**"换模型"不是唯一的触发条件——"同一个模型换了服务端"同样会换掉向量空间。**
2026-09-28 embedding 与 rerank 从 `memory.021130.xyz` 迁到 `memory3.021130.xyz`：

| 项 | 旧 host | 新 host |
| --- | --- | --- |
| embedding 维度 | 1024 | **4096** |
| 服务实现 | 自研封装（`owned_by: aml`） | **vllm 直服** |
| reranker | 有 | **有**（id 换成 `qwen3-reranker-4b`；网关当天又只改了 nginx.conf ⇒ `/v1/rerank` rewrite 到 vLLM 原生 `/v1/score`，**请求/响应信封都换了**，见 `rank/reranker.py` 顶部的表） |

⇒ **维度变了 ⇒ 旧集合与旧缓存全部作废**，动作与上面六步**逐条相同**（第 4 步的"新维度"
不再是待定项，实测就是 4096）。⚠ **别把它当成"改个 base_url"**：跳过第 3 步的症状
只是"检索结果很差"，**没有任何东西会报错**。

> ⚠ **若只迁了 embedding、reranker 没跟上**：那是**允许的**——`RerankUnavailable`
> 会把它降级成 RRF 顺序（D12），响应照旧合法。**唯一的要求是别不吭声**：
> `make probe-reranker` 退出码 0 才算它可用，在那之前 `rerank.enabled` 保持 `false`。

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
