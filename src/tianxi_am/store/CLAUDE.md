# store/ — 存储层

**PRD**：§6.1（真源 DDL）、§6.3（派生索引 Qdrant）；邻域查询的 SQL 属 §10

## 本模块的构成（**实现已存在**）

```text
schema.sql        qa_pairs + applied_batches 的 DDL（§6.1）
sqlite_store.py   真源读写、事务、批次守卫、邻域查询（§10 的 SQL 也在这里）
qdrant_store.py   collection 建/写/查、payload 过滤、**按传入参数执行** prefetch+RRF
```

**这是唯一接触 SQLite 与 Qdrant 的目录。** 上层拿到的是领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`——§6.3 的分工表只有在读写收口到一处时才守得住。

> ### ⚠ 检索的**参数所有权**不在本目录
>
> `prefetch_limit` / `weights` / `k` / `top_k` 的**取值、校验与标定**归 [`../retrieve/`](../retrieve/)；本目录只做 **Qdrant 的请求构造与执行**——把传进来的参数翻成 `prefetch` + `rrf` 的调用，并把结果还原成领域对象。**两条边界**：本目录**不决定**用哪个 `k`（`k=61` 是正确性常量，由 `retrieve/` 校验，见 D5）、**不决定** `top_k`（它来自请求，由 `retrieve/` 传入；**写死 100 是契约错误**）。
>
> ⇒ **一句话：`retrieve/` 说"用什么参数"，`store/` 说"怎么发给 Qdrant"。** §7.3 那张表的**数值**属于 `retrieve/`，**写法**（`prefetch` 每路带 `using`、根级 `limit`）属于本目录。

> **向量缓存不在这里**——它在 [`../embed/`](../embed/)，因为缓存键是"渲染后文本的哈希"，属渲染 + embedding 的关注点（§7.2）。**§6.3 把这层定义为恰好两样东西**（SQLite 真源 + Qdrant 派生索引，且明确"不可互换"），塞进第三个存储会削弱那条规则。

**邻域的 SQL 在这里，调度在 `rank/`**：§10 的扩窗跑在 rerank **之后**，属"排序 → 扩窗 → 打包"三连，所以由 [`../rank/`](../rank/) 编排；但查询本身是本目录的事。

---

## 分工不可互换（§6.3）

| | SQLite（真源） | Qdrant（派生索引） |
| -- | -- | ---- |
| 存 | QA 对正文、`status`、`chunk_ordinal` + `local_index`、`event_time`、`id` | 向量（`dense` + `bm25`）+ 过滤键 payload |
| 不存 | 向量 | **正文** |
| 能做的事 | 事务、关系查询 | 向量检索、payload 过滤、**按给定参数**执行 RRF 融合 |
| 不能做的事 | 向量检索 | 跨 point 事务、关系查询 |
| 坏了怎么办 | —— | **可从 SQLite 全文重建** |

**两边都不能省**：正文在 SQLite（**需要事务**保证"一次 Add 的整批块要么全落要么全不落"、且与幂等守卫同生共死）；而检索**需要向量**。**正文不复制进 Qdrant payload**：会有两份正文，一旦不一致**就无法判断该信哪一份**；而按主键批量取正文是**微秒级操作**，没有性能理由去复制。（两条完整论证见 D3）

**时序**：Qdrant 出 `id` → 回 SQLite 按 `id` 批量取正文 → 渲染 → 打包。"取正文"这一步**没有缓存层，也不应该**：正文的可信来源只有一处。

---

## DDL 要点（§6.1）

```sql
qa_pairs(
  id             TEXT PRIMARY KEY,     -- hash(user_id, session_id, chunk_ordinal, local_index)，位置派生
  user_id        TEXT NOT NULL,        -- 隔离契约
  session_id     TEXT NOT NULL,
  chunk_ordinal  INTEGER NOT NULL,     -- 从 request_id 解析（D25）
  local_index    INTEGER NOT NULL,     -- 该批组合出的块序号（D25）
  question       TEXT,                 -- 可空
  answer         TEXT,                 -- 可空
  status         TEXT NOT NULL,        -- 'complete' | 'pending'（D24 起写入恒为 complete）
  event_time     INTEGER,              -- Unix 毫秒，取该对首条消息的 timestamp，可空
  request_id     TEXT NOT NULL         -- 最后触碰该行的 Add；**不承担幂等职责**
)
UNIQUE(user_id, session_id, chunk_ordinal, local_index);

applied_batches(
  request_id  TEXT PRIMARY KEY,        -- AML 的 Add request_id，天然唯一
  user_id     TEXT NOT NULL,
  session_id  TEXT NOT NULL,
  applied_at  INTEGER NOT NULL
)
```

> ⚠ **`pair_idx` 已被 D25 拆成 `chunk_ordinal` + `local_index`**。
> **相邻性不再由它们表达**（chunk 序号可以跳号），而由一条读时现算的稠密序：
> `ROW_NUMBER() OVER (PARTITION BY user_id, session_id ORDER BY chunk_ordinal, local_index) - 1`。
> ⚠ `ROW_NUMBER()` 是 **1-based**，**必须减 1**——忘了会在该列上混基，直接破坏相邻判定。
> 落点：`sqlite_store._SEQ_COLUMN`，入口 `fetch_session_ordered()`。

**设计原则**（§6.1）：

> 除正文外，一个字段要么**进索引**、要么**进 `ORDER BY`**、要么有**明确的运维用途**（幂等守卫 / 溯源），否则它不该是列。

> 能被 `question` / `answer` 重建的东西，一律不进真源。

**不预留 v2 字段**（§6.1 / D3）：`kind` / `parent_id`（Fact 抽取用）与无定义键的 `meta` **已删、也不要再加**——预留字段既不入索引也不进 `ORDER BY`，**正属于"不该是列"的一类**。真要做时再加。

---

## 五条不变式

| # | 不变式 | 为什么 |
| --- | --- | --- |
| **1** | `id` **位置派生** `hash(user_id, session_id, chunk_ordinal, local_index)`（D25） | 用内容哈希会在内容变时变 `id`，留下**孤儿 point** |
| **2** | ~~`pair_idx` **session 内连续**~~ ⛔ **D25 已作废** → 邻域由**读时稠密序** `seq` 现算 | 旧理由（有空洞则邻域**静默消失**）已随 `pair_idx` 一起消失：chunk 跳号不再破坏相邻，而"中间真的少了一块"仍被抓住。✅ **旧理由里仍然成立的那半条**：`event_time` 保证不了 session 内顺序 ⇒ 稠密序必须**按位置**算，不能按时间 |
| **3** | `applied_batches` **只增不改**，应用成功时**在同一事务里插入** | AML 重试是**正常行为**；没有它，正常重试就得靠撞 `UNIQUE` 兜（D25 之后撞得响，但那会把重试变成 500） |
| **4** | `UNIQUE(user_id, session_id, chunk_ordinal, local_index)` **既保唯一、又是 `ORDER BY` 的键** | 唯一性防同一位置被写两次；`ORDER BY` 决定 `seq`，进而决定**谁跟谁相邻** |
| **5** | 缓存键 = **渲染后文本的哈希**（实现在 [`../embed/`](../embed/)） | **不能用 `id`**——同一位置的内容若变了而 `id` 不变，用 `id` 会拿到**陈旧向量** |

> **不变式 1 与 5 的方向正好相反**，这是故意的：`id` 要**稳定**（所以位置派生），缓存键要**跟着内容变**（所以内容哈希）。两者互换都会静默出错。

**幂等一律查 `applied_batches`**——`qa_pairs.request_id` **不承担幂等职责**（§6.1）。它记的是"哪一批写下了这一行"，而守卫要回答的是"这批被应用过吗"——两件事。旁表是唯一记着后者的地方，还带着 `user_id` / `session_id` / `applied_at`。

**位置不能与既有行撞车**：`id` 是位置派生的，而写入是 upsert ⇒ **同一位置被写两次会静默覆盖**。
⚠ **D25 之后这条的含义变了**：位置不再是"服务端从 0 起的计数器"（那个"重开"的动作已经不存在），
而是**请求里带来的 `(chunk_ordinal, local_index)`**——所以真正的风险变成
**"两个不同的 Add 声称同一个 chunk 序号"**（重复投递一批、或平台侧序号错乱）。
那种情况下 UPSERT 会覆盖，**而 `applied_batches` 只能挡住完全相同的 `request_id`**。
⚠ **D24 之后这条依旧成立，而且更容易被误读**：组合的边界是**一次 Add**，位置的作用域仍是**整个 session**——两者不是一回事。

> **D24（2026-09-27）之后 `status` 恒为 `'complete'`**：块在写下那一刻就是最终形状，
> 没有 pending、没有 repair、**没有任何后台任务会回头改这些行**。`'pending'` 仍在取值域里
> 只是因为按旧规则写过的库还能读。⇒ 下面那节"补全 pending 对"描述的流程**已不存在**。

---

## 连接模型：**一次逻辑操作一个连接**（D17）

`SqliteStore` **不长期持有连接**。写是 `connect → BEGIN IMMEDIATE → 读改写 → COMMIT/ROLLBACK → close`，读是 `connect → SELECT → close`；本类只保存 `db_path` 与存储逻辑。

**三条不许越过的线**：

| # | 线 | 为什么 |
| --- | --- | --- |
| 1 | **同一事务里的每个 helper 都必须复用外层传进来的 `conn`** | helper 自己 `connect()` 会落到**另一个事务**里：拿不到本事务的写锁、也看不到它未提交的中间态 |
| 2 | helper **绝不自作 `commit` / `rollback`** | 那会让"半批"落库——§6.5 的三步必须同事务 |
| 3 | `journal_mode = WAL` **只在 `open()` 落一次**；`busy_timeout` / `synchronous` / `foreign_keys` / `row_factory` **每个新连接都要设** | 前者是**数据库级**（写进库文件、对后续所有连接生效）；后三者是**每连接级**，新连接**不继承**——`foreign_keys` 尤其（SQLite 默认 **OFF**） |

**并发由 SQLite 自己串行化**（WAL + `busy_timeout`）：抢不到写锁的一方**等待**，不抛 `SQLITE_BUSY`。**不新增应用层写锁**——真出现高尾延迟时它是**优化**，不是正确性基础。

> **但这一切的前提是 `BEGIN IMMEDIATE`**（已实测）：默认的 `BEGIN` 会让两条并发压力用例双双报 `database is locked`——**连"各写各的 session"那条也失败**，因为**升级写锁时 `busy_timeout` 不生效**（SQLite 宁可立刻报错也不冒死锁的险）。**不要把它当成"只是个位置分配优化"。**

> ⚠ **D25 起这里只剩一条职责**：SQLite 的 writer 串行化（键是整个库文件）。
> ~~"按 session 的业务顺序不在这里，属 `service/` 的 `SessionLocks`"~~——**那把锁已删**
> （位置成为请求的纯函数之后不再需要业务串行化）。职责沿革见 D17 → D25。

**为什么是短生命周期，而不是 thread-local 长连接 / 单一共享连接**：见 [D17](../../../docs/decisions.md)。一句话是"**连接与线程的约束不该泄漏到任何上层**"，而 FastAPI 的 `def` 路由**就跑在线程池里**。

---

## Qdrant 配置要点（§6.3）—— 配置项见 [`../../docs/config-reference.md`](../../../docs/config-reference.md) §8

（七项配置 —— 模式 / 版本 / 分片数 / 命名向量 / payload 索引 / payload 内容 / 写入方式 —— 的**值与理由都在那份文档**，本文件不复制。下面是本层特有的三条。）

> **`is_tenant` 只支持 keyword / uuid 两种类型**——别把 `user_id` 建成 integer。

> ✅ **已在钉死的 Qdrant v1.17.0 容器上实测通过**：`is_tenant` 建索引返回 `acknowledged` 且读回 config 已生效；`dense` + `bm25`（modifier `idf`）+ `shard_number:1` 的集合可建；`prefetch` 两路 + 根级 `rrf{weights, k:61}` + `user_id` 过滤的查询可用，**且过滤确实只返回目标 tenant 的点**。

**融合写法**：`prefetch` + `rrf{weights, k}`，**`k` 必须显式设 `61`**，每个 `prefetch` 都要带 `using`（命名向量场景下不写 `using`，Qdrant 无法确定用哪一路），根级还要给 `limit`（取自请求的 `top_k`，**不要写死 100**）。完整写法见 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) §3；`k=61` 的由来见 D5。

---

## ~~补全 `pending` 对时必须做两件事（§6.5）~~ ⛔ **D24 已作废**

~~1. **重算该对的 embedding 并 upsert 覆盖原 Qdrant point。** `id` 不变——**这正是 `id` 必须位置派生的原因**。
2. **埋点**：`pending_created` / `pending_completed` / `pending_orphaned`。

若不重算，Qdrant 里留下的是**残缺文本的向量**——检索仍会命中它，但命中的是"不存在的那个版本"，而且**不会报错**。~~

**2026-09-27（D24）起没有 `pending` 对可补**：一次 Add 的块在写下那一刻就是最终形状，
`apply_batch` 在同一个事务里把正文写死，**此后再没有任何改写路径**。

> **但这一段的反面教训仍然有效，只是换了落点**：Qdrant 与 SQLite 一旦不一致，
> 检索命中的就是"不存在的那个版本"而**不报错**。D24 之后这条风险只剩一个来源——
> **`index_pairs` 失败**（SQLite 已提交、Qdrant 没写）。它对冲在
> [`../service/pipeline.py`](../service/pipeline.py) 的修复路径（重放时按 session 幂等重建），
> 由 [`../service/CLAUDE.md`](../service/CLAUDE.md) 那一节的失败窗口描述。
