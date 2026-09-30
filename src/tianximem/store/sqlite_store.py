"""SQLite 真源（PRD §6.1 / §6.3）。

**这是唯一接触业务真源 SQLite 的模块。** 上层拿到的是领域对象 `QaPair`，不是 `sqlite3.Row`
——§6.3 的分工表只有在读写收口到一处时才守得住。
（⚠ `embed/base.py` 的 `DiskVectorCache` 也开 SQLite，但那是可重建的派生缓存。）

本模块【不认识】Qdrant、embedding、渲染，也【不认识】任何数据集——
它只认 §2.1 的 canonical Add 契约字段（`user_id` / `session_id` / `request_id` / 消息）。

不变式归属（store/CLAUDE.md 的五条）：
  1. `id` 位置派生 .................... `make_pair_id()` + `schema.sql` 的 PRIMARY KEY
  2. 邻接是 **Add 内显式链** ............ `prev_memory_id` / `next_memory_id`，**D28** 起
                                        （跨 Add 不存在可信的全局序 ⇒ 不推断）
  3. `applied_batches` 只增不改 ....... `record_batch()`（只 INSERT，不 UPDATE）
  4. UNIQUE 索引 = 一次 Add 的排序键 ... `schema.sql` 的 UNIQUE + `fetch_by_request()`
  5. 缓存键 = 渲染文本哈希 ............ **不在本模块**，属 `embed/`（§7.2）

⚠ **当前库形状与 D28 之前写的库不兼容**——两者混用会**静默出错**。
但 `open()` 会**自动搬一次**（`migrate()`，事务内、保正文、重算 id、重连链）。

位置 = `(request_id, local_index)`，`request_id` 是 **opaque string**（D28）。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

logger = logging.getLogger(__name__)

Status = Literal["complete", "pending"]

STATUS_COMPLETE: Final[Status] = "complete"
_VALID_STATUSES: Final[frozenset[str]] = frozenset({"complete", "pending"})

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: D28 重建表用的 DDL —— **刻意在这里复制一份、刻意不跟 `schema.sql` 走**。
#:
#: 它是一次**历史时刻的快照**（"把 D25 的库搬成 D28 的形状"），而不是"当前 schema"：
#: 将来 schema 再变，这段也不该跟着变——它描述的永远是那个已经发生过的搬移。
_MIGRATE_DDL: Final[str] = """
CREATE TABLE qa_pairs__d28 (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    request_id     TEXT NOT NULL,
    local_index    INTEGER NOT NULL,
    prev_memory_id TEXT,
    next_memory_id TEXT,
    question       TEXT,
    answer         TEXT,
    status         TEXT NOT NULL,
    event_time     INTEGER,
    UNIQUE(user_id, session_id, request_id, local_index)
)
"""


def _canonical(*parts: str) -> str:
    """把若干个字符串**无歧义**地拼成一个。

    ⚠ **不能只用分隔符 join。** 那样只在"字段本身不含分隔符"时才无歧义——
    例如 `("a", "b|c")` 与 `("a|b", "c")` 会拼出同一个串（这正是被
    `test_pair_id_does_not_collide_across_field_boundaries` 抓到过的一个真 bug）。

    这里用长度前缀（netstring 风格）：`len(part):part` 逐个拼接。
    长度数字以 `:` 终结、且后随的正是那么多字符，所以切分是唯一的 ⇒ 拼接**单射**。
    """
    return "".join(f"{len(p)}:{p}" for p in parts)


def make_pair_id(user_id: str, session_id: str, request_id: str, local_index: int) -> str:
    """不变式 1：`id` **位置派生** = hash(user_id, session_id, request_id, local_index)。

    ⚠ **绝不能用内容哈希。** 按内容算的 `id` 会在两段**内容相同**的记忆上撞车，
    而**重建索引**要求「同一行永远是同一个 point」——做不到就会在 Qdrant 里留下
    **孤儿 point**（旧的还在，新的也写进去，检索命中"不存在的那个版本"，**且不报错**）。

    ⚠ **`request_id` 在这里只是"一个字符串"**（D28）：它作为一个**整体**参与哈希，
    **绝不解析**它的内容。于是任意形状的 id（`abc` / `foo:bar` / UUID / 平台实发的
    `r_3115…`）都等价可用——这正是要买的那件事。

    ⚠ **位置仍然全是请求的纯函数**（`request_id` 来自请求、`local_index` 来自组合结果）：
    同一 `request_id` + 同一 payload 重试时，必然算出**同一个 `id`** ⇒ upsert 是覆盖，
    不会留下孤儿；不同 Add 之间也不会撞（`request_id` 不同 ⇒ `id` 不同）。

    哈希算法本身**不是规格规定的**——规格只规定了**输入**是那四元组。
    这里取 SHA-256 全 64 位十六进制：确定性、无碰撞顾虑、长度对 TEXT 主键无所谓。
    """
    payload = _canonical(user_id, session_id, request_id, str(local_index))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QaPair:
    """一个记忆块的领域对象（§6.2 的 QA 对 + D24 的块模型 + D28 的 Add 内邻接）。

    ⚠ **没有"会话内序号"这种字段**（D28）：跨 Add 不存在可信的全局序，所以位置只在
    一次 Add 内有意义（`local_index`），而"谁跟谁相邻"由 `prev_memory_id` /
    `next_memory_id` **显式**给出。
    """

    id: str
    user_id: str
    session_id: str
    #: 写下这一行的 Add。**opaque**：只做溯源与位置的一半，一律不解析（D28）。
    request_id: str
    #: 它在**那一次 Add 的块列表**里的 0-based 序号。⚠ 跨 Add 没有意义。
    local_index: int
    #: Add 内**显式**邻接（D28）：指向前/后**完整的 QA 块**；不完整或链端为 `None`。
    prev_memory_id: str | None
    next_memory_id: str | None
    question: str | None
    answer: str | None
    status: Status
    event_time: int | None

    @property
    def is_complete(self) -> bool:
        """两侧都在的块（= 一个完整的 QA）。

        ⚠ **只有它进 Add 内的邻接链**（D28）：A-only / Q-only 照样存、照样 embedding、
        照样能被检索到，但它们**不参与邻接**（`prev` / `next` 恒为 `None`）。
        """
        return self.question is not None and self.answer is not None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _remap_legacy_rows(
    rows: Sequence[sqlite3.Row],
) -> list[tuple[str, str | None, str | None, sqlite3.Row]]:
    """旧库（D25）的行 → `(新 id, prev_memory_id, next_memory_id, 原行)`。

    **邻接规则与 `apply_batch` 逐字相同**（只连完整 QA、只在同一次 Add 内），
    否则搬完之后"同一个库、两种邻接"——那是最难查的一类不一致。

    ⚠ 分组键里**必须**带 `request_id`：那正是"邻接只在一次 Add 内"这条规则的落点。
    行是按 `(user_id, session_id, request_id, local_index)` 排好序进来的 ⇒ 同组必然连续。
    """
    out: list[tuple[str, str | None, str | None, sqlite3.Row]] = []  # (id, prev, next, row)
    group: list[tuple[str, sqlite3.Row]] = []
    owner: tuple[str, str, str] | None = None

    def flush() -> None:
        complete = [row_id for row_id, row in group if row["is_complete"]]
        prev_by = {rid: complete[i - 1] for i, rid in enumerate(complete) if i}
        next_by = {rid: complete[i + 1] for i, rid in enumerate(complete) if i + 1 < len(complete)}
        for row_id, row in group:
            out.append((row_id, prev_by.get(row_id), next_by.get(row_id), row))

    for row in rows:
        this = (row["user_id"], row["session_id"], row["request_id"])
        if this != owner:
            flush()
            group = []
            owner = this
        group.append((make_pair_id(*this, row["local_index"]), row))
    flush()
    return out


class SqliteStore:
    """`qa_pairs` + `applied_batches` 的读写与事务。

    ## 连接模型：**短生命周期**（D17）

    本类**只保存 `db_path` 与存储逻辑**，**不长期持有** `sqlite3.Connection`。
    连接的生命周期恰好是**一次逻辑操作**：

    ```text
    写： connect → BEGIN IMMEDIATE → 读改写 → COMMIT / ROLLBACK → close
    读： connect → SELECT                                        → close
    ```

    **为什么必须这样**：FastAPI 的 `def` 路由跑在**线程池**里，而 `sqlite3` 的连接
    **只能被创建它的线程使用**。任何"启动时建一个连接、之后长期复用"的写法，
    到线上就是**每个请求都抛** `sqlite3.ProgrammingError`。短生命周期把这个问题
    **从根上消掉**——连接永远在同一次调用的同一个线程里建与关，
    于是"连接与线程的约束"**不需要泄漏到任何上层**（完整论证见 D17）。

    代价是每次操作多一次 `connect`（约几十微秒）。用 WAL + `busy_timeout` 让
    **SQLite 自己**负责多连接之间的 writer 串行化，**不另加应用层写锁**。

    ## 事务边界由调用方持有

    所有方法都要求传入 `conn`，并且**不做 connect、不做 commit/rollback**——
    只有 [`transaction()`][SqliteStore.transaction] 管这两件事。
    这样 §6.5 的三步（守卫查表 → 位置恢复 → 挂接 + 记录本批）才能落在**同一个事务**里。

    ⚠ **同一事务里的所有 helper 必须复用外层传进来的那个 `conn`**，
    **禁止 helper 自己 connect、也禁止 helper 自己 commit**——否则原子性就破了，
    而且**不会报错**（半批数据看起来完全正常）。
    """

    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._db_path = str(db_path)
        #: 抢不到写锁时**等待**而不是立刻抛 `SQLITE_BUSY` 的毫秒数（D17）。
        #: 来自 `configs/*.yaml` 的 `storage.sqlite.busy_timeout_ms`——
        #: ⚠ **不要在这里读配置**：本模块只受 `store/` 与 `common/` 的约束，
        #: 由上层的装配点把值传进来（见 `service/app.py` 的 `build_services`）。
        self._busy_timeout_ms = busy_timeout_ms

    # ── 生命周期 ────────────────────────────────────────────────────────

    @classmethod
    def open(cls, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> SqliteStore:
        """建表（幂等）→ **若有旧库就搬一次**（D28）→ 把数据库级设置落一次。

        ⚠ **`journal_mode = WAL` 只在这里执行一次**（见 `_init_journal_mode` 的理由），
        不在每个短生命周期连接上重复。与之相对，`busy_timeout` / `synchronous` /
        `foreign_keys` / `row_factory` 是**每连接**的，必须每次新连接都设——见 `_connect()`。
        """
        store = cls(db_path, busy_timeout_ms=busy_timeout_ms)
        store.init_schema()
        store.migrate()
        store._init_journal_mode()
        return store

    # ── 旧库迁移（D28）─────────────────────────────────────────────────

    def migrate(self) -> bool:
        """把**按旧规则（D25）写过的库**搬到 D28 的形态。返回是否真的搬了。

        ## 为什么是"重建表"而不是 `ALTER TABLE`

        要做的三件事里有两件只有重建才做得到：**去掉 `chunk_ordinal` 列**，
        以及**重算 `id`**——`id` 是位置派生的，而位置的定义变了（`chunk_ordinal` 来自解析
        `request_id`，而 D28 取消解析）。SQLite 的 `ALTER TABLE DROP COLUMN` 还带着一堆
        版本与索引限制，重建顺带把 UNIQUE 也换成新的。

        ## 搬什么、不搬什么

        | 东西 | 怎么处理 |
        | --- | --- |
        | **正文**（`question` / `answer` / `event_time` / `status`） | **原样搬**——真源不可丢 |
        | `id` | **重算** = `hash(user_id, session_id, request_id, local_index)` |
        | `prev` / `next` | 按 `(user_id, session_id, request_id)` 分组、组内按 `local_index` 重连，
        只连完整 QA（与 `apply_batch` 同一条规则） |
        | `payload_hash`（旁表） | **留 NULL**（旧行没有 payload 可核）⇒ 那一批的重放只能放行，
        见 `applied_batch_payload_hash` |

        ⚠ **搬完之后 Qdrant 里的 point 全是孤儿**（`id` 变了，point id 由它派生）⇒
        **必须跑一次 `tools/reindex.py --drop`**。这一步是**免费**的：embedding 缓存按
        "渲染文本的哈希"作键，而渲染没变 ⇒ 不会重付 embedding 的钱（§7.2）。

        ⚠ 全程在**一个事务**里：搬一半的库比旧库更糟。
        """
        with self.transaction() as conn:
            if not self._needs_migration(conn):
                return False
            rows = conn.execute(
                "SELECT user_id, session_id, request_id, local_index,"
                " question, answer, status, event_time,"
                " (question IS NOT NULL AND answer IS NOT NULL) AS is_complete"
                " FROM qa_pairs"
                " ORDER BY user_id, session_id, request_id, local_index"
            ).fetchall()
            values = _remap_legacy_rows(rows)

            conn.execute(_MIGRATE_DDL)
            conn.executemany(
                "INSERT INTO qa_pairs__d28"
                " (id, user_id, session_id, request_id, local_index, prev_memory_id,"
                "  next_memory_id, question, answer, status, event_time)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row_id,
                        row["user_id"],
                        row["session_id"],
                        row["request_id"],
                        row["local_index"],
                        prev_id,
                        next_id,
                        row["question"],
                        row["answer"],
                        row["status"],
                        row["event_time"],
                    )
                    for row_id, prev_id, next_id, row in values
                ],
            )
            conn.execute("DROP TABLE qa_pairs")
            conn.execute("ALTER TABLE qa_pairs__d28 RENAME TO qa_pairs")

            batch_columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(applied_batches)")
            }
            if "payload_hash" not in batch_columns:
                conn.execute("ALTER TABLE applied_batches ADD COLUMN payload_hash TEXT")

        logger.warning(
            "真源库按旧规则（D25：位置来自 request_id 里的 chunk 序号）写过，已搬到 D28 的形态"
            "（位置 = `(request_id, local_index)`，邻接改成 Add 内显式链）。\n"
            "  ⚠ **`id` 全变了 ⇒ Qdrant 里的 point 现在是孤儿**：跑一次"
            " `tools/reindex.py --drop`（embedding 缓存按内容哈希，不会重付）。"
        )
        return True

    def _needs_migration(self, conn: sqlite3.Connection) -> bool:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(qa_pairs)")}
        return bool(columns) and ("chunk_ordinal" in columns or "prev_memory_id" not in columns)

    def _init_journal_mode(self) -> None:
        """把 `journal_mode = WAL` **落到数据库文件**上。只在 `open()` 里调用一次。

        WAL 是**数据库级**设置：它写进库文件头，对**之后所有连接**持续生效
        （WAL 下读不阻塞写、写不阻塞读）。所以**不需要**在每个短生命周期连接上重复执行
        ——重复执行只是白跑一次 IO，且 `PRAGMA journal_mode` 会返回设置**之后**的模式，
        看起来"生效了"，掩盖了它本来就已经生效这件事。

        ⚠ `PRAGMA journal_mode` **不能在事务里执行**。这里用的是自管事务的连接
        （`isolation_level=None`），处于自动提交态，所以直接可用。
        """
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """造一个**已配好必要 PRAGMA** 的新连接。调用方负责关它。

        **每连接级**的设置都在这里；**数据库级**的（WAL）在 `open()` 里落一次。
        """
        conn = sqlite3.connect(self._db_path, isolation_level=None)  # 自己管事务
        # synchronous=FULL：契约要求"持久化完成且立即可搜索"后才能响应（§2.1）。
        # 用 NORMAL 会在 OS/断电级崩溃下丢掉最后几个已提交事务——那正是"重试"的来源，
        # 虽然幂等守卫兜得住，但没有理由在这里省。
        conn.execute("PRAGMA synchronous = FULL")
        # 多连接并发写时，让**输的那一方等待**而不是立刻抛 SQLITE_BUSY。
        # 取自配置的 `storage.sqlite.busy_timeout_ms`（默认 5000）。
        conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")
        # 外键约束默认是 **OFF**（SQLite 的默认值），且它是**每连接**生效的。
        # 当前 schema 里没有外键 ⇒ 这条是**空转**；但留着它，将来加 FK 时不会静默失效。
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def db_path(self) -> str:
        """真源文件路径。

        §6.3：**SQLite 是唯一的真源，也是唯一不可重建物**——Qdrant 可以从它全量重建。
        测试用它来模拟"进程重启后重连同一个库"。
        """
        return self._db_path

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """一次**只读**操作的作用域：开连接 → 交给调用方 SELECT → 关。

        ⚠ **不要为读拿 `BEGIN IMMEDIATE`**——那会取写锁，让只读查询去和写事务抢。
        WAL 下读不阻塞写、写不阻塞读。
        """
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """执行 `schema.sql`（`IF NOT EXISTS`，可重复调用）。

        用自己的**短生命周期连接**跑——因为建表本身是幂等的，**不需要**包在写事务里，
        也**不应该**去拿 `BEGIN IMMEDIATE` 的写锁。
        """
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """一个写事务。**这是唯一管 "connect / begin / commit / rollback / close" 的地方。**

        **`BEGIN IMMEDIATE` 而不是默认的 `BEGIN`**（**已实测，不是引文档**）：

        * **升级写锁时 `busy_timeout` 不生效**：
          写事务里**先读后写**（守卫 `SELECT` → `INSERT`）。deferred `BEGIN` 让事务
          以**读**身份开始，要写时得**升级**成写锁；若对方也持着读锁并同样想升级，
          SQLite **立刻**返回 `SQLITE_BUSY`（"database is locked"）
          ——它宁可立刻报错也不冒死锁的险，**不应用 `busy_timeout`**。

        实测：把这一行换成 `BEGIN`，`tests/test_store.py` 的两条并发压力用例
        **双双失败**，报 `OperationalError('database is locked')` ×5——**而且跨 session
        那条也失败**（不同 `(user_id, session_id)` 一样撞），所以这不只是"位置撞车"的防护。

        IMMEDIATE 在事务开头就拿写锁，让并发写事务退化成"排队"而不是"互锁"
        （完整论证见 D17）。**它是数据库级的写者串行**（键是整个库文件），
        与"按 session 的业务顺序"**不是一件事**——后者不存在，位置是请求的纯函数。

        ⚠ **这一层是并发安全的唯一落点**：同 `(user_id, session_id)` 的 Add 可以并发
        进入本层，它们在**这里**排队（等待，而不是失败）
        ⇒ `busy_timeout` 的取值（配置项）比过去更要紧。

        事务体里的**每一个 helper 都必须复用这里 yield 出去的 `conn`**——
        helper 自己 `connect()` 会落到另一个事务里（拿不到本事务的写锁、也看不到本事务
        未提交的中间态），helper 自己 `commit()` 则会让"半批"落库。**两种情况都不报错。**

        WAL 下**多个连接**可以同时开事务：写与写之间由 SQLite 自己串行化
        （拿不到写锁的那一方等 `busy_timeout`，不是立刻抛），读与写互不阻塞。
        所以除了这条 `BEGIN IMMEDIATE`，**不需要**再叠一层应用层的写锁。
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                # 用 `conn.rollback()` 而不是 `execute("ROLLBACK")`：前者在"无事务可回滚"时
                # 是 no-op，后者会抛 `no transaction is active` 并**盖掉真正的异常**。
                conn.rollback()
                raise
            else:
                conn.commit()
        finally:
            # `close()` 放在 finally：异常路径、成功路径、commit 自身失败，都要还掉这个连接。
            # （若关闭时还有未提交事务，SQLite 会隐式回滚——不会留下半个批次的脏数据。）
            conn.close()

    # ── 第 1 步：幂等守卫（§6.5）────────────────────────────────────────

    def is_batch_applied(self, conn: sqlite3.Connection, request_id: str) -> bool:
        """本批是否已应用过。

        ⚠ 查的是 **`applied_batches`**，**不是 `qa_pairs.request_id`**——
        后者记的是"哪一批写下了这一行"，与"这批被应用过吗"是两件事（见 D4）。
        """
        row = conn.execute(
            "SELECT 1 FROM applied_batches WHERE request_id = ? LIMIT 1", (request_id,)
        ).fetchone()
        return row is not None

    def applied_batch_payload_hash(self, conn: sqlite3.Connection, request_id: str) -> str | None:
        """那批当时写下的 `payload_hash`。**`None` = 没这行，或那行早于 D28（未知）**。

        ⚠ 调用方**必须**把这两种 `None` 都当成"核不了"（放行），而**不能**当成
        "payload 不同"——后者会把**每一个老库里的 request_id 重放**变成 409。
        判据是 `is_batch_applied()` 那个布尔，不是这个返回值。
        """
        row = conn.execute(
            "SELECT payload_hash FROM applied_batches WHERE request_id = ? LIMIT 1", (request_id,)
        ).fetchone()
        return None if row is None else row["payload_hash"]

    def record_batch(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str | None = None,
        applied_at: int | None = None,
    ) -> None:
        """记下"本批已应用"。**只增不改**——没有对应的 UPDATE，也不该有。

        必须在**与业务写入同一个事务**里调用（§6.5 第 1 步的"※"）。
        主键冲突会直接抛 `IntegrityError`：那是**调用方漏了守卫**的信号，要响。

        `payload_hash` 是规范化 payload 的 sha256（D28）——它让"同 id 不同 payload"
        能被判出来（否则只能静默当成重放，而那是**两份不同的记忆里必有一份消失**）。
        """
        conn.execute(
            "INSERT INTO applied_batches"
            " (request_id, user_id, session_id, payload_hash, applied_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                request_id,
                user_id,
                session_id,
                payload_hash,
                _now_ms() if applied_at is None else applied_at,
            ),
        )

    # ── 位置来自请求，**没有"分配"这一步**（D25 / D28）───────────────────
    #
    # 位置 = `(request_id, local_index)`——**请求的纯函数**：
    #
    #   * `request_id`  原样来自请求（**opaque，不解析**，D28）
    #   * `local_index` 是这一批组合出的块的 0-based 序号（`compose_memory_blocks` 确定）
    #
    # ⇒ 没有共享计数器、没有读-改-写 ⇒ **不需要串行化**，不同 Add 之间**也不存在
    #   任何顺序关系**（跨 Add 的"谁先谁后"既不表达、也不推断）。
    # ⚠ **不要在这里加回一个"取下一个位置"的方法**——那是 D25 之前的读-改-写。
    # ⚠ **也不要加回"从 request_id 里取序号"**——那正是 D28 取消的东西。

    # ── 第 3 步：写 ────────────────────────────────────────────────────

    def insert_pair(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        session_id: str,
        request_id: str,
        local_index: int,
        prev_memory_id: str | None,
        next_memory_id: str | None,
        question: str | None,
        answer: str | None,
        status: Status,
        event_time: int | None,
        pair_id: str | None = None,
    ) -> QaPair:
        """新建一个块。`id` 由位置派生，调用方通常不必传 `pair_id`。

        ⚠ **位置与邻接都是传进来的、不是这里算的**（D28）：调用方（`pairing/apply.py`）
        从组合结果拿 `local_index`，并按"只连完整 QA"那条规则算出 `prev` / `next`。
        这里**不许**有"看起来更方便"的默认值——一个默认值就会让调用方悄悄退回旧口径。

        ⚠ **邻接在一次 INSERT 里就写死**：没有任何 UPDATE、没有"回填上一条的 next"这种事。
        这一条是 D24 那条纪律（块在写下那一刻就是最终形状）在邻接上的延伸——
        只要一次 Add 的块列表在写之前就全知道了，就不需要第二遍写。

        ⚠ 主键/UNIQUE 冲突会直接抛 `IntegrityError`（同一 `(request_id, local_index)`
        被写两次）：那是"同一批被应用了两次"的信号，**要响**——守卫本该拦住它。
        """
        if status not in _VALID_STATUSES:
            raise ValueError(f"非法 status: {status!r}（只允许 'complete' | 'pending'）")
        pid = (
            make_pair_id(user_id, session_id, request_id, local_index)
            if pair_id is None
            else pair_id
        )
        conn.execute(
            "INSERT INTO qa_pairs"
            " (id, user_id, session_id, request_id, local_index,"
            "  prev_memory_id, next_memory_id, question, answer, status, event_time)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                user_id,
                session_id,
                request_id,
                local_index,
                prev_memory_id,
                next_memory_id,
                question,
                answer,
                status,
                event_time,
            ),
        )
        return QaPair(
            id=pid,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            local_index=local_index,
            prev_memory_id=prev_memory_id,
            next_memory_id=next_memory_id,
            question=question,
            answer=answer,
            status=status,
            event_time=event_time,
        )

    # ── 读 ─────────────────────────────────────────────────────────────

    def fetch_by_request(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> list[QaPair]:
        """**一次 Add 写下的全部块**，按 `local_index` 升序（D28）。

        这是"Add 内"这个作用域在存储层的唯一入口：邻接（`prev` / `next`）只在
        这一批里成立，所以**凡是需要"前后"的代码都从这里出发**，而不是从 session 出发。

        两个调用方：

        * `rank/neighbor.py` 的扩窗（沿 `prev` / `next` 走，见那边的 `expand_neighbors`）
        * `service/pipeline.py` 的修复路径（守卫命中后按 **request_id** 幂等重建，
          而不是按 session——那正是 D28 收紧的那一处）

        ⚠ **`user_id` / `session_id` 也在 `WHERE` 里**：`request_id` 虽然是全局唯一的，
        但隔离契约（§2.2）不该由"调用方记得传对 id"来保证——带上另外两个键之后，
        跨 user 取数**结构上做不到**。

        ⚠ 走的是 `UNIQUE(user_id, session_id, request_id, local_index)` 建出的索引
        （不变式 4）：三列前缀命中 `WHERE`、第四列命中 `ORDER BY` ⇒ **不扫全表、不额外排序**。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND request_id = ?"
            " ORDER BY local_index",
            (user_id, session_id, request_id),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def fetch_pairs_by_ids(self, conn: sqlite3.Connection, pair_ids: Sequence[str]) -> list[QaPair]:
        """按主键批量取正文（§6.3 的时序：Qdrant 出 `id` → 回这里取正文）。

        正文**不进** Qdrant payload；"取正文"这一步没有缓存层，也不应该有——
        正文的可信来源只有这一处。

        ⚠ 回来的行带着它们**真实的** `prev_memory_id` / `next_memory_id`（那两列就在表里）
        ——D28 起邻接是**显式存储的**，不再需要"取回整个 session 再算下标"那一步。
        """
        if not pair_ids:
            return []
        placeholders = ",".join("?" * len(pair_ids))
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE id IN ({placeholders})",
            tuple(pair_ids),
        ).fetchall()
        by_id = {r["id"]: _to_pair(r) for r in rows}
        # 保持调用方给来的顺序（检索名次顺序），而不是数据库返回的顺序
        return [by_id[pid] for pid in pair_ids if pid in by_id]

    def iter_pairs(self, conn: sqlite3.Connection, *, user_id: str | None = None) -> list[QaPair]:
        """枚举全部块（可选按 `user_id` 限定），按
        `(session_id, request_id, local_index)` 排序。

        ⚠ **这个顺序不是"会话顺序"**（D28 起不存在这种东西）：它只是"同一 session 的
        各次 Add 挨在一起，每次 Add 内部按 `local_index`"——够用，因为调用方
        （`tools/reindex.py` 的全量重建）要的就是"每条都写到"。

        ⚠ **只读访问器**，不改动任何既有语义。存在的理由是 §6.3 那条"Qdrant 可从 SQLite
        全量重建"：重建需要一次枚举，而让 `qdrant_store.py` 自己写 SQL 会破坏
        "store/ 是唯一接触 SQLite 的目录"。

        **不是检索路径**——检索按主键批量取正文走 `fetch_pairs_by_ids`。
        """
        if user_id is None:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM qa_pairs"
                " ORDER BY user_id, session_id, request_id, local_index"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id = ?"
                " ORDER BY session_id, request_id, local_index",
                (user_id,),
            ).fetchall()
        return [_to_pair(r) for r in rows]

    def assert_isolation(self, conn: sqlite3.Connection, user_id: str) -> list[QaPair]:
        """取某 user 的全部对（供隔离测试断言用）。

        `user_id` 是**唯一**的检索隔离字段；`session_id` 只是分组字段，
        **不是** Search 的过滤器（§2.2）。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id = ?"
            " ORDER BY session_id, request_id, local_index",
            (user_id,),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def explain_by_request(
        self, conn: sqlite3.Connection, user_id: str, session_id: str, request_id: str
    ) -> str:
        """ "取一次 Add 的全部块"那条查询的 `EXPLAIN QUERY PLAN`。

        测试用：断言它**不扫全表、不额外排序**（走 `UNIQUE` 建出的那个索引）。
        """
        rows = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND request_id = ?"
            " ORDER BY local_index",
            (user_id, session_id, request_id),
        ).fetchall()
        return " | ".join(str(r["detail"]) for r in rows)


_COLUMNS = (
    "id, user_id, session_id, request_id, local_index,"
    " prev_memory_id, next_memory_id, question, answer, status, event_time"
)


def _to_pair(row: sqlite3.Row) -> QaPair:
    status = row["status"]
    if status not in _VALID_STATUSES:
        raise ValueError(f"库中出现非法 status: {status!r}")
    return QaPair(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        request_id=row["request_id"],
        local_index=row["local_index"],
        prev_memory_id=row["prev_memory_id"],
        next_memory_id=row["next_memory_id"],
        question=row["question"],
        answer=row["answer"],
        status=status,
        event_time=row["event_time"],
    )
