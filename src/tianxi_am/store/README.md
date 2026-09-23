# store/ — 存储层

**PRD**：§6.1（真源 DDL）、§6.3（派生索引 Qdrant）；邻域查询的 SQL 属 §10

## 要写什么

```text
schema.sql        qa_pairs + applied_batches 的 DDL（§6.1）
sqlite_store.py   真源读写、事务、批次守卫、邻域查询（§10 的 SQL 也在这里）
qdrant_store.py   collection 建/写/查、payload 过滤、prefetch+RRF
```

**这是唯一接触 SQLite 与 Qdrant 的目录。** 上层拿到的是领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`——§6.3 的分工表只有在读写收口到一处时才守得住。

> **向量缓存不在这里**——它在 [`../embed/`](../embed/)，因为缓存键是"渲染后文本的哈希"，属渲染 + embedding 的关注点（§7.2）。**§6.3 把这层定义为恰好两样东西**（SQLite 真源 + Qdrant 派生索引，且明确"不可互换"），塞进第三个存储会削弱那条规则。

**邻域的 SQL 在这里，调度在 `rank/`**：§10 的扩窗跑在 rerank **之后**，属"排序 → 扩窗 → 打包"三连，所以由 [`../rank/`](../rank/) 编排；但查询本身是本目录的事。

---

## 分工不可互换（§6.3）

| | SQLite（真源） | Qdrant（派生索引） |
| -- | -- | ---- |
| 存 | QA 对正文、`status`、`pair_idx`、`event_time`、`id` | 向量（`dense` + `bm25`）+ 过滤键 payload |
| 不存 | 向量 | **正文** |
| 能做的事 | 事务、关系查询 | 向量检索、payload 过滤、RRF 融合 |
| 不能做的事 | 向量检索 | 跨 point 事务、关系查询 |
| 坏了怎么办 | —— | **可从 SQLite 全文重建** |

**为什么两边都不能省**：能力不重叠。补全一个 `pending` 对是"读-改-写"，**需要事务**；而检索**需要向量**。

**为什么正文不复制进 Qdrant payload**：会有两份正文，一旦不一致**就无法判断该信哪一份**。而按主键批量取正文是**微秒级操作**，没有性能理由去复制。

**时序**：Qdrant 出 `id` → 回 SQLite 按 `id` 批量取正文 → 渲染 → 打包。"取正文"这一步**没有缓存层，也不应该有**：正文的可信来源只有一处。

---

## DDL 要点（§6.1）

```sql
qa_pairs(
  id          TEXT PRIMARY KEY,        -- hash(user_id, session_id, pair_idx)，位置派生
  user_id     TEXT NOT NULL,           -- 隔离契约
  session_id  TEXT NOT NULL,
  pair_idx    INTEGER NOT NULL,        -- session 内连续，从 0 起
  question    TEXT,                    -- 可空
  answer      TEXT,                    -- 可空
  status      TEXT NOT NULL,           -- 'complete' | 'pending'
  event_time  INTEGER,                 -- Unix 毫秒，取该对首条消息的 timestamp，可空
  request_id  TEXT NOT NULL            -- 最后触碰该行的 Add；**不承担幂等职责**
)
UNIQUE(user_id, session_id, pair_idx);

applied_batches(
  request_id  TEXT PRIMARY KEY,        -- AML 的 Add request_id，天然唯一
  user_id     TEXT NOT NULL,
  session_id  TEXT NOT NULL,
  applied_at  INTEGER NOT NULL
)
```

**设计原则**（§6.1）：

> 除正文外，一个字段要么**进索引**、要么**进 `ORDER BY`**、要么有**明确的运维用途**（幂等守卫 / 溯源），否则它不该是列。

> 能被 `question` / `answer` 重建的东西，一律不进真源。

**不预留 v2 字段。** 早期版本留过 `kind` / `parent_id`（Fact 抽取用）与无定义键的 `meta`——**三者已一并删除**，理由相同：预留字段既不入索引也不进 `ORDER BY`，**正属于"不该是列"的一类。真要做时再加**（§6.1）。

---

## 五条不变式

| # | 不变式 | 为什么 |
| --- | --- | --- |
| **1** | `id` **位置派生** `hash(user_id, session_id, pair_idx)` | 用内容哈希会在补全时变 `id`，留下**孤儿 point** |
| **2** | `pair_idx` **session 内连续** | 有空洞则 `±1` 邻域**静默消失**；且 `event_time` 保证不了 session 内顺序——同秒消息排序未定义，会让消融不可复现 |
| **3** | `applied_batches` **只增不改**，应用成功时**在同一事务里**插入 | 唯一能防"位置重分配"的守卫（见下） |
| **4** | `UNIQUE(user_id, session_id, pair_idx)` 建出的索引**正好是邻域查询的键** | 扩窗从 ±1 改 ±2 只需改 `BETWEEN` 的界 |
| **5** | 缓存键 = **渲染后文本的哈希**（实现在 [`../embed/`](../embed/)） | **不能用 `id`**——补全时内容变了而 `id` 不变，用 `id` 会拿到**陈旧向量** |

> **不变式 1 与 5 的方向正好相反**，这是故意的：`id` 要**稳定**（所以位置派生），缓存键要**跟着内容变**（所以内容哈希）。两者互换都会静默出错。

**幂等一律查 `applied_batches`**——`qa_pairs.request_id` **不承担幂等职责**，它会被后一批覆盖，因此可能丢掉早先批次的指纹（§6.1 / §6.5）。

**跨批续接时 `pair_idx` 不能从 0 重开**：`id` 是位置派生的，重开必然与既有行撞车，而写入是 upsert，**会静默覆盖掉上一批的数据**（§6.5）。

---

## Qdrant 配置要点（§6.3）

| 项 | 值 | 理由 |
| -- | -- | ---- |
| 模式 | **server（Docker）** | local 模式**静默丢弃 payload 索引**——`create_payload_index` 只打一行警告就返回，且数据格式与 server 不兼容 |
| 版本 | **钉死 ≥ v1.17.0** | §7.3：`weights` 需 ≥ v1.17.0 |
| 集合分片数 | **1** | 根级融合跨分片合并，**分片数变化会改变排名**——为可复现必须单分片（官方文档亦指出分片数改变排名且**无报错**） |
| 命名向量 | `dense` + `bm25` | 稀疏向量的距离固定为 **Dot** |
| payload 索引 | `user_id`（**keyword** + `is_tenant`）/ `session_id`（keyword）/ `event_time`（integer） | **必须在写入数据前建**，否则 HNSW 需要重建才有过滤感知 |
| payload 内容 | `user_id` / `session_id` / `pair_idx` / `event_time` | **不含正文** |
| 写入 | **`wait=true`** | 契约要求"响应前立即可搜"；默认异步不保证 |

> **`is_tenant` 只支持 keyword / uuid 两种类型**——别把 `user_id` 建成 integer。

> ✅ **已在钉死版本上实测通过（2026-09-23，Qdrant v1.17.0 容器）**：
> `{"type":"keyword","is_tenant":true}` 建索引返回 `acknowledged`，读回 config 确认 `"is_tenant":true` 已生效；
> `dense`(size N, Cosine) + `bm25`(modifier `idf`) + `shard_number:1` 的集合可建；
> `prefetch` 两路（各带 `using`）+ 根级 `rrf{weights, k:61}` + `user_id` 过滤的查询可用，
> **且过滤确实只返回目标 tenant 的点**（测试中 `u2` 的点被正确排除）。
> **注意**：`k=61` 与默认 `k=2` 的量级差约 30 倍、而名次都看似正常——见 [`docs/decisions.md` D5](../../../docs/decisions.md)。

**融合**：`prefetch` + `rrf{weights, k}`，**`k` 必须显式设 `61`**，每个 `prefetch` 都要带 `using`（命名向量场景下不写 `using`，Qdrant 无法确定用哪一路），根级还要给 `limit`（取自请求的 `top_k`，**不要写死 100**）。完整写法见 [`../../docs/config-reference.md`](../../../docs/config-reference.md) §3。

---

## 补全 `pending` 对时必须做两件事（§6.5）

1. **重算该对的 embedding 并 upsert 覆盖原 Qdrant point。** `id` 不变——**这正是 `id` 必须位置派生的原因**。
2. **埋点**：`pending_created` / `pending_completed` / `pending_orphaned`。

若不重算，Qdrant 里留下的是**残缺文本的向量**——检索仍会命中它，但命中的是"不存在的那个版本"，而且**不会报错**。
