# deploy/ — 部署与环境

**PRD**：§6.3（Qdrant server 模式）、§7.3（版本钉死）

## 要写什么

```text
compose.yaml    Qdrant server，**镜像 tag 钉死**
```

**这个目录只有一件事要做对：把版本钉死。** 其余都是围绕它的说明。

**运行形态（D12）**：**模型（LLM / embedding / reranker）全部经自建网关远程访问；检索服务与 Qdrant 跑在本机。** 所以本机**不需要 GPU，但需要 docker**。

---

## 1. 为什么必须 server 模式（§6.3）

> **Qdrant local 模式会静默丢弃 payload 索引**——`create_payload_index` 只打**一行警告**就返回，且其数据格式与 server 不兼容。

而本项目的 `user_id` / `session_id` / `event_time` **三个筛选全部依赖 payload 索引**。

**所以 local 模式不是"性能差一点"的替代品，是"过滤静默失效"。** 没有降级方案。

> ✅ **docker 已就位（2026-09-23，D12 同日补充）**：Docker Desktop 29.8.0（WSL2 后端，`desktop-linux` context）。**daemon 走 Windows 命名管道，`docker` 命令直接从 Git Bash 可用**；发布端口转发到主机 ⇒ **`localhost:6333` 直接可用**，`.env` 默认值不用改。

---

## 2. 为什么版本要钉死（§7.3）

`{rrf: {weights, k}}` 的可用性是有版本门槛的：

| 参数 | 自哪个版本起 |
| --- | --- |
| `k` | **v1.16.0** |
| `weights` | **v1.17.0** |

⇒ **要求 ≥ v1.17.0**，且**部署时钉死**。

**这与 §2.2 的可复现性同源**：融合行为的任何变化都会改变排名，而 **Full 只有 2 次、一旦接受即版本冻结**。一个浮动的镜像 tag 会让"这个分数是哪个版本跑出来的"变成不可回答的问题。

**`compose.yaml` 里的 tag 就是这条要求的落地点**——它必须是完整版本号（`qdrant/qdrant:v1.17.0`），**不能用 `latest`、不能用大版本号浮标**。

---

## 3. 集合配置 —— 是**客户端**设置，不在 compose 里

| 项 | 落在哪 |
| --- | --- |
| 集合分片数 / 命名向量 / payload 索引 / payload 内容 / 写入 `wait=true` | [`../src/tianxi_am/store/CLAUDE.md`](../src/tianxi_am/store/CLAUDE.md)（**一处声明**） |

**本目录只管起服务。** 集合是客户端建的——拓扑本身**已经是规格**（单集合 + `user_id` tenant 过滤 + `is_tenant` 索引 + 分片 1 + payload 索引先于写数据），原文见 [`../src/tianxi_am/store/CLAUDE.md`](../src/tianxi_am/store/CLAUDE.md) 的"Qdrant 配置要点"。

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

**第 2 步不能省**：Qdrant 是**派生读存储**，可从 SQLite 全文重建；**SQLite 是真源，丢了就没了**。

**第 3 步不能省**：缓存键不含模型标识——若不主动作废，旧向量会**静默命中**。而维度不同的表现只是"检索结果很差"，**不会报错**。可重建性对照表见 [`../var/CLAUDE.md`](../var/CLAUDE.md)。

> **⚠ 一处 R1 类风险（§6.4）**：本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` 不同，**本地量出的"单请求能装多少对"不能直接搬到线上**——**第 6 步之后必须重新量一次**（[`../docs/open-questions.md`](../docs/open-questions.md) E7）。

---

---

## 6. 24×7 运行就绪

单进程 HTTP 服务，**必须能 24×7 连续运行**——**Full run 持续 0.5–2 天**。

| 检查项 | 出处 |
| --- | --- |
| `--workers 1`（按 session 的锁是**进程内**的，多 worker 会静默失效） | §15 |
| Qdrant 与 SQLite 的文件都在**持久卷**上——SQLite **随 run 归档** | §15 |
| `latency/query` 有观测——**接近 30 分钟上限就削减 agent 轮数** | §14 |
| 单请求最长 30 分钟，**预算充足但不是无限** | §2.2 |

> **⚠ 一条开发环路的单点风险（D12）**：answer / judge / embed / rerank **四条都打同一个网关**，而 harness 评测时会同时驱动 Add/Search（embed + rerank）与答案/裁判生成。**必须有客户端并发上限**，否则**排队超时会伪装成"模型变差了"**（[`../docs/open-questions.md`](../docs/open-questions.md) V8）。

---

## 7. 一条不能做的事

**不要为了让 Qdrant 好部署而退回 local 模式**，也**不要**把 `user_id` / `session_id` / `event_time` 的过滤改到客户端做——后者等价于"检索全部、再在内存里筛"，会**破坏隔离字段的语义**（§2.2），而且随数据量增长不可控。

**local 模式的失败是静默的**（只打一行警告），所以"跑起来没报错"**不构成**它能用的证据。
