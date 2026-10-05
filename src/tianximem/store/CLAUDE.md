# store/ — 存储层

**PRD**：§6.1（真源 DDL）、§6.3（派生索引 Qdrant）；邻域查询的 SQL 属 §10

## 本模块的构成（**实现已存在**）

```text
schema.sql        qa_pairs + applied_batches（真源）及 memory_facts + evidence_coverage（派生）
sqlite_store.py   真源读写、事务、批次守卫、邻域查询、共同事实索引
qdrant_store.py   collection 建/写/查、payload 过滤、**按传入参数执行** prefetch+RRF
```

**这是唯一接触业务真源（SQLite）与 Qdrant 的目录。** 上层拿到的是领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`——§6.3 的分工表只有在读写收口到一处时才守得住。

2026-10-05 起事实索引只使用共同 `memory_facts` 与版本覆盖表 `evidence_coverage`。
事实逐字引句必须来自声明的 question/answer 侧，所属用户必须与原文一致。
旧业务派生表在已有库中封存，不再读写；新库不创建这些表。抽取器共用于 Add、
幂等重放及有界补索引；任何重建不得修改 `qa_pairs` 或 `applied_batches`。
字段与操作边界见 [当前架构](../../../docs/architecture.md)。

> ⚠ 那句限定词"业务真源"是必要的：`embed/` 的 `DiskVectorCache` **也直接开 SQLite**，
> 但它是一个**可重建的派生缓存**（另一个库文件、另一套 PRAGMA），
> 与这里"真源"的职责不同。两处的连接模型刻意一致（D17），见
> [`../embed/base.py`](../embed/base.py) 的 `DiskVectorCache`。

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
| 存 | QA 对正文、`status`、`(request_id, local_index)`、`prev` / `next`、`event_time`、`id` | 向量（`dense` + `bm25`）+ 过滤键 payload |
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
  id             TEXT PRIMARY KEY,     -- hash(user_id, session_id, request_id, local_index)，位置派生
  user_id        TEXT NOT NULL,        -- 隔离契约
  session_id     TEXT NOT NULL,
  request_id     TEXT NOT NULL,        -- 写下这一行的 Add；**opaque：只做溯源 + 位置的一半**（D28）
  local_index    INTEGER NOT NULL,     -- 它在【那一次 Add 的块列表】里的 0-based 序号（D28）
  prev_memory_id TEXT,                 -- Add 内**显式**邻接（D28）：只连完整 QA，
  next_memory_id TEXT,                 -- 不完整的块（A-only / Q-only）恒为 NULL
  question       TEXT,                 -- 可空
  answer         TEXT,                 -- 可空
  status         TEXT NOT NULL,        -- 'complete' | 'pending'（D24 起写入恒为 complete）
  event_time     INTEGER               -- Unix 毫秒，取该对首条消息的 timestamp，可空
)
UNIQUE(user_id, session_id, request_id, local_index);

applied_batches(
  request_id   TEXT PRIMARY KEY,       -- AML 的 Add request_id，天然唯一
  user_id      TEXT NOT NULL,
  session_id   TEXT NOT NULL,
  payload_hash TEXT,                   -- 规范化 payload 的 sha256（D28）；NULL = 旧行（未知⇒放行）
  applied_at   INTEGER NOT NULL
)
```

> ⚠ **位置是 `(request_id, local_index)`，相邻性由 `prev_memory_id` / `next_memory_id`
> 显式给出**（D28）——**没有**"会话内稠密序 `seq`"这种东西了（它随 D28 一起删掉：
> 跨 Add 不存在可信的顺序，按下标推相邻会**跨 Add 拼接到一起**而不报错）。

**设计原则**（§6.1）：

> 除正文外，一个字段要么**进索引**、要么**进 `ORDER BY`**、要么有**明确的运维用途**（幂等守卫 / 溯源），否则它不该是列。

> 能被 `question` / `answer` 重建的东西，一律不进真源。

**不预留 v2 字段**（§6.1 / D3）：`kind` / `parent_id`（Fact 抽取用）与无定义键的 `meta` **不要加**——预留字段既不入索引也不进 `ORDER BY`，**正属于"不该是列"的一类**。真要做时再加。

---

## 五条不变式

| # | 不变式 | 为什么 |
| --- | --- | --- |
| **1** | `id` **位置派生** `hash(user_id, session_id, request_id, local_index)`（D28，`request_id` 作为整体参与、**不解析**） | 用内容哈希会在内容变时变 `id`，留下**孤儿 point** |
| **2** | 邻接由**存储里的显式指针**给出（D28）：`prev` / `next` 在**写下时**就按"只连完整 QA"算好，跨 Add 永不连 | 跨 Add 没有可信顺序（`request_id` 不透明）⇒ 任何"推出来的"相邻性都会**跨 Add 拼接** |
| **3** | `applied_batches` **只增不改**，应用成功时**在同一事务里插入** | AML 重试是**正常行为**；没有它，正常重试就得靠撞 `UNIQUE` 兜（D25 之后撞得响，但那会把重试变成 500） |
| **4** | `UNIQUE(user_id, session_id, request_id, local_index)` **既保唯一、又是"取一次 Add"的排序键** | 唯一性防同一位置被写两次；`ORDER BY local_index` 是 `fetch_by_request` 的顺序 |
| **5** | 缓存键 = **渲染后文本的哈希**（实现在 [`../embed/`](../embed/)） | **不能用 `id`**——同一位置的内容若变了而 `id` 不变，用 `id` 会拿到**陈旧向量** |

> **不变式 1 与 5 的方向正好相反**，这是故意的：`id` 要**稳定**（所以位置派生），缓存键要**跟着内容变**（所以内容哈希）。两者互换都会静默出错。

**幂等一律查 `applied_batches`**——`qa_pairs.request_id` **不承担幂等职责**（§6.1）。它记的是"哪一批写下了这一行"，而守卫要回答的是"这批被应用过吗"——两件事。旁表是唯一记着后者的地方，还带着 `user_id` / `session_id` / `applied_at`。

**位置由请求唯一决定**（D28）：`(request_id, local_index)` 里的两半**都来自这一次请求**
（`request_id` 原样、`local_index` 是本批组合的产物）⇒ **两个不同的 Add 不可能撞位置**
（`request_id` 不同 ⇒ `id` 不同）。唯一能撞的是"**同一批被写两次**"，而那由
`applied_batches` 守卫 + `UNIQUE` 两道一起挡。

⚠ **D28 之后组合与位置是同一个作用域**（都是一次 Add）——D25 时代那个"组合按 Add 切、
位置按 session 算"的错位**没有了**。

> **`status` 恒为 `'complete'`**（D24）：块在写下那一刻就是最终形状，
> 没有 pending、没有 repair、**没有任何后台任务会回头改这些行**。`'pending'` 仍在取值域里
> 只是为了让**按旧规则写过的库**还能读回来。

---

## 连接模型：**一次逻辑操作一个连接**（D17）

`SqliteStore` **不长期持有连接**。写是 `connect → BEGIN IMMEDIATE → 读改写 → COMMIT/ROLLBACK → close`，读是 `connect → SELECT → close`；本类只保存 `db_path`、`busy_timeout_ms` 与存储逻辑。

**三条不许越过的线**：

| # | 线 | 为什么 |
| --- | --- | --- |
| 1 | **同一事务里的每个 helper 都必须复用外层传进来的 `conn`** | helper 自己 `connect()` 会落到**另一个事务**里：拿不到本事务的写锁、也看不到它未提交的中间态 |
| 2 | helper **绝不自作 `commit` / `rollback`** | 那会让"半批"落库——§6.5 的三步必须同事务 |
| 3 | `journal_mode = WAL` **只在 `open()` 落一次**；`busy_timeout` / `synchronous` / `foreign_keys` / `row_factory` **每个新连接都要设** | 前者是**数据库级**（写进库文件、对后续所有连接生效）；后三者是**每连接级**，新连接**不继承**——`foreign_keys` 尤其（SQLite 默认 **OFF**） |

**并发由 SQLite 自己串行化**（WAL + `busy_timeout`）：抢不到写锁的一方**等待**，不抛 `SQLITE_BUSY`。**不新增应用层写锁**——真出现高尾延迟时它是**优化**，不是正确性基础。

> **但这一切的前提是 `BEGIN IMMEDIATE`**（已实测）：默认的 `BEGIN` 会让两条并发压力用例双双报 `database is locked`——**连"各写各的 session"那条也失败**，因为**升级写锁时 `busy_timeout` 不生效**（SQLite 宁可立刻报错也不冒死锁的险）。**不要把它当成"只是个位置分配优化"。**

> **这里只有一条职责**：SQLite 的 writer 串行化（键是整个库文件）。
> 按 session 的业务串行化**不在这里、也不需要**（D28）——
> 一块的位置由**请求本身**决定，不经过任何跨请求的分配。
> 连接模型的完整论证见 **D17**。

**为什么是短生命周期，而不是 thread-local 长连接 / 单一共享连接**：见 [D17](../../../docs/decisions.md)。一句话是"**连接与线程的约束不该泄漏到任何上层**"，而 FastAPI 的 `def` 路由**就跑在线程池里**。

---

## Qdrant 配置要点（§6.3）—— 配置项见 [`../../docs/config-reference.md`](../../../docs/config-reference.md) §8

（七项配置 —— 模式 / 版本 / 分片数 / 命名向量 / payload 索引 / payload 内容 / 写入方式 —— 的**值与理由都在那份文档**，本文件不复制。下面是本层特有的三条。）

> **`is_tenant` 只支持 keyword / uuid 两种类型**——别把 `user_id` 建成 integer。

> ✅ **已在钉死的 Qdrant v1.17.0 容器上实测通过**：`is_tenant` 建索引返回 `acknowledged` 且读回 config 已生效；`dense` + `bm25`（modifier `idf`）+ `shard_number:1` 的集合可建；`prefetch` 两路 + 根级 `rrf{weights, k:61}` + `user_id` 过滤的查询可用，**且过滤确实只返回目标 tenant 的点**。

**融合写法**：`prefetch` + `rrf{weights, k}`，**`k` 必须显式设 `61`**，每个 `prefetch` 都要带 `using`（命名向量场景下不写 `using`，Qdrant 无法确定用哪一路），根级还要给 `limit`（取自请求的 `top_k`，**不要写死 100**）。完整写法见 [`../../../docs/config-reference.md`](../../../docs/config-reference.md) §3；`k=61` 的由来见 D5。

---

## Qdrant 与 SQLite 不一致时**不报错**（§6.5）

**一次 Add 的块在写下那一刻就是最终形状**（D24）：`apply_batch` 在同一个事务里把正文写死，
**此后再没有任何改写路径**——没有 pending、也没有"补全"这一步。

> **风险只剩一个来源**：**`index_pairs` 失败**（SQLite 已提交、Qdrant 没写）。
> 此时检索命中的是"不存在的那个版本"而**不报错**。它对冲在
> [`../service/pipeline.py`](../service/pipeline.py) 的修复路径（重放时按 session 幂等重建），
> 由 [`../service/CLAUDE.md`](../service/CLAUDE.md) 那一节的失败窗口描述。
>
> ⚠ **任何"后台回头改这些行"的方案都要按 D24 重新论证**——它引入的正是这里要防的那种不一致。
