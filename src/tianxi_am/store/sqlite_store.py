"""SQLite 真源（PRD §6.1 / §6.3）。

**这是唯一接触 SQLite 的模块。** 上层拿到的是领域对象 `QaPair`，不是 `sqlite3.Row`
——§6.3 的分工表只有在读写收口到一处时才守得住。

本模块【不认识】Qdrant、embedding、渲染，也【不认识】任何数据集——
它只认 §2.1 的 canonical Add 契约字段（`user_id` / `session_id` / `request_id` / 消息）。

不变式归属（store/CLAUDE.md 的五条）：
  1. `id` 位置派生 .................... `make_pair_id()` + `schema.sql` 的 PRIMARY KEY
  2. 位置 = 请求的纯函数 ............... `(chunk_ordinal, local_index)`，**D25** 起
                                        **不再有读-改-写**（⇒ 没有"分配位置"这一步）
  3. `applied_batches` 只增不改 ....... `record_batch()`（只 INSERT，不 UPDATE）
  4. UNIQUE 索引 = 会话顺序的排序键 .... `schema.sql` 的 UNIQUE + `fetch_session_ordered()`
  5. 缓存键 = 渲染文本哈希 ............ **不在本模块**，属 `embed/`（§7.2）

⚠ **D25（2026-09-27）与 D24（同日）的区别**：D24 只改"哪些消息进同一个块"（组合），
D25 改"块的位置从哪来"（`MAX+1` → 请求里的 chunk 序号）。**位置模型变了 ⇒ 旧库不可复用。**
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Status = Literal["complete", "pending"]

STATUS_COMPLETE: Final[Status] = "complete"
STATUS_PENDING: Final[Status] = "pending"
_VALID_STATUSES: Final[frozenset[str]] = frozenset({"complete", "pending"})

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _canonical(*parts: str) -> str:
    """把若干个字符串**无歧义**地拼成一个。

    ⚠ **不能只用分隔符 join。** 那样只在"字段本身不含分隔符"时才无歧义——
    例如 `("a", "b|c")` 与 `("a|b", "c")` 会拼出同一个串（这正是被
    `test_pair_id_does_not_collide_across_field_boundaries` 抓到过的一个真 bug）。

    这里用长度前缀（netstring 风格）：`len(part):part` 逐个拼接。
    长度数字以 `:` 终结、且后随的正是那么多字符，所以切分是唯一的 ⇒ 拼接**单射**。
    """
    return "".join(f"{len(p)}:{p}" for p in parts)


def make_pair_id(user_id: str, session_id: str, chunk_ordinal: int, local_index: int) -> str:
    """不变式 1：`id` **位置派生** = hash(user_id, session_id, chunk_ordinal, local_index)。

    **D25 起位置是这两个数**（旧口径是 session 内连续的 `pair_idx`，由 `MAX+1` 现算）。

    ⚠ **绝不能用内容哈希。** 按内容算的 `id` 会在两段**内容相同**的记忆上撞车，
    而**重建索引**要求「同一行永远是同一个 point」——做不到就会在 Qdrant 里留下
    **孤儿 point**（旧的还在，新的也写进去，检索命中"不存在的那个版本"，**且不报错**）。

    ⚠ **这两个数都必须是请求的纯函数**（chunk 来自 `request_id`、local 来自组合结果）：
    它们一旦依赖到达顺序，位置就又变成读-改-写，D25 想买的两件事（并发安全、乱序不翻转）
    会一起消失。

    哈希算法本身**不是规格规定的**——规格只规定了**输入**是位置四元组。
    这里取 SHA-256 全 64 位十六进制：确定性、无碰撞顾虑、长度对 TEXT 主键无所谓。
    """
    payload = _canonical(user_id, session_id, str(chunk_ordinal), str(local_index))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QaPair:
    """一个记忆块的领域对象（§6.2 的 QA 对 + D24 的块模型）。

    `seq` **不是列**——它是**读时算出来的会话内稠密序**（见 `fetch_session_ordered`），
    扩窗与段合并只用它，所以"缺号的 chunk 不破坏相邻"这件事是结构性成立的。
    """

    id: str
    user_id: str
    session_id: str
    chunk_ordinal: int
    local_index: int
    question: str | None
    answer: str | None
    status: Status
    event_time: int | None
    request_id: str
    #: 会话内稠密序（0-based，按 `(chunk_ordinal, local_index)` 排）。
    #: 只有 `fetch_session_ordered` / `iter_pairs` 会填；`insert_pair` 不知道它。
    seq: int = -1


def _now_ms() -> int:
    return int(time.time() * 1000)


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
        """建表（幂等），并**把数据库级设置落一次**。

        ⚠ **`journal_mode = WAL` 只在这里执行一次**（见 `_init_journal_mode` 的理由），
        不在每个短生命周期连接上重复。与之相对，`busy_timeout` / `synchronous` /
        `foreign_keys` / `row_factory` 是**每连接**的，必须每次新连接都设——见 `_connect()`。
        """
        store = cls(db_path, busy_timeout_ms=busy_timeout_ms)
        store.init_schema()
        store._init_journal_mode()
        return store

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

        * **升级写锁时 `busy_timeout` 不生效**（**这是 D25 之后剩下的那一条理由**）：
          写事务里**先读后写**（守卫 `SELECT` → `INSERT`）。deferred `BEGIN` 让事务
          以**读**身份开始，要写时得**升级**成写锁；若对方也持着读锁并同样想升级，
          SQLite **立刻**返回 `SQLITE_BUSY`（"database is locked"）
          ——它宁可立刻报错也不冒死锁的险，**不应用 `busy_timeout`**。
        * ~~**读-改-写**：`pair_idx` 的分配是 `MAX(pair_idx)` → 写位置~~ ⛔ **D25 已取消**：
          位置现在从 `request_id` 的 chunk 序号派生，**没有读-改-写**。
          ⇒ 这条理由**不再成立**，但**上一条仍成立**，所以 `BEGIN IMMEDIATE` 照留。

        实测：把这一行换成 `BEGIN`，`tests/test_store.py` 的两条并发压力用例
        **双双失败**，报 `OperationalError('database is locked')` ×5——**而且跨 session
        那条也失败**（不同 `(user_id, session_id)` 一样撞），所以这不只是"位置撞车"的防护。

        IMMEDIATE 在事务开头就拿写锁，让并发写事务退化成"排队"而不是"互锁"
        （完整论证见 D17）。**它是数据库级的写者串行**（键是整个库文件），
        与"按 session 的业务顺序"**不是一件事**——后者是 D25 删掉的那把锁。

        ⚠ D25 之后**没有应用层的 session 锁了**：同 `(user_id, session_id)` 的 Add
        可以并发进入本层，它们在**这里**排队（等待，而不是失败）。⇒ 本方法是
        "并发安全"的**唯一**落点，`busy_timeout` 的取值（配置项）比过去更要紧。

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
        后者会被后一批覆盖，指纹会丢（见 D4）。
        """
        row = conn.execute(
            "SELECT 1 FROM applied_batches WHERE request_id = ? LIMIT 1", (request_id,)
        ).fetchone()
        return row is not None

    def record_batch(
        self,
        conn: sqlite3.Connection,
        request_id: str,
        user_id: str,
        session_id: str,
        applied_at: int | None = None,
    ) -> None:
        """记下"本批已应用"。**只增不改**——没有对应的 UPDATE，也不该有。

        必须在**与业务写入同一个事务**里调用（§6.5 第 1 步的"※"）。
        主键冲突会直接抛 `IntegrityError`：那是**调用方漏了守卫**的信号，要响。
        """
        conn.execute(
            "INSERT INTO applied_batches (request_id, user_id, session_id, applied_at)"
            " VALUES (?, ?, ?, ?)",
            (request_id, user_id, session_id, _now_ms() if applied_at is None else applied_at),
        )

    # ── 位置：D25 起**没有"分配"这一步** ────────────────────────────────
    #
    # 旧口径是 `next_pair_idx() = MAX(pair_idx)+1`（读-改-写，必须靠 `SessionLocks` 串行化，
    # 且顺序 = 到达顺序）。D25 把它换成 `(chunk_ordinal, local_index)`——**请求的纯函数**：
    #
    #   * `chunk_ordinal` 从 `request_id` 解析（`pairing.parse_chunk_ordinal`）
    #   * `local_index`   是这一批组合出的块的 0-based 序号（`compose_memory_blocks` 确定）
    #
    # ⇒ 没有共享计数器、没有读-改-写 ⇒ **不需要串行化**，且**乱序到达不会翻转顺序**。
    # ⚠ **不要在这里加回一个"取下一个位置"的方法**——那会把 D25 买的两件事一起还回去。

    # ── 第 3 步：写 ────────────────────────────────────────────────────

    def insert_pair(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        session_id: str,
        chunk_ordinal: int,
        local_index: int,
        question: str | None,
        answer: str | None,
        status: Status,
        event_time: int | None,
        request_id: str,
        pair_id: str | None = None,
    ) -> QaPair:
        """新建一个块。`id` 由位置派生，调用方通常不必传 `pair_id`。

        ⚠ **位置是传进来的、不是这里算的**（D25）：调用方（`pairing/apply.py`）从
        `request_id` 解析 `chunk_ordinal`、从组合结果拿 `local_index`。
        这里**不许**有"看起来更方便"的默认值——一个默认值就会让调用方悄悄退回旧口径。
        ⚠ 主键/UNIQUE 冲突会直接抛 `IntegrityError`（同一 `(chunk, local)` 被写两次）：
        那是"同一批被应用了两次"的信号，**要响**——守卫本该拦住它。
        """
        if status not in _VALID_STATUSES:
            raise ValueError(f"非法 status: {status!r}（只允许 'complete' | 'pending'）")
        pid = (
            make_pair_id(user_id, session_id, chunk_ordinal, local_index)
            if pair_id is None
            else pair_id
        )
        conn.execute(
            "INSERT INTO qa_pairs"
            " (id, user_id, session_id, chunk_ordinal, local_index,"
            "  question, answer, status, event_time, request_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                user_id,
                session_id,
                chunk_ordinal,
                local_index,
                question,
                answer,
                status,
                event_time,
                request_id,
            ),
        )
        return QaPair(
            id=pid,
            user_id=user_id,
            session_id=session_id,
            chunk_ordinal=chunk_ordinal,
            local_index=local_index,
            question=question,
            answer=answer,
            status=status,
            event_time=event_time,
            request_id=request_id,
        )

    # ── 读 ─────────────────────────────────────────────────────────────

    def fetch_session_ordered(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
    ) -> list[QaPair]:
        """该 session 的**全部块，按会话顺序**，并给每行填上 `seq`（D25）。

        这是 §10 邻域扩展的数据来源——**取代**旧的 `fetch_pairs_by_idx_range`。

        ## 为什么改成"取整段"而不是"取 `BETWEEN` 窗口"

        D25 把位置从"`session` 内连续整数"换成 `(chunk_ordinal, local_index)`：后者
        **可能有空洞**（某个 chunk 一批都没产出块、或 chunk 序号跳号），于是
        `pair_idx BETWEEN k-r AND k+r` 那种**整数算术**不再等价于"会话里前后各 r 个"。

        改成：一次取整个 session 的有序列表（session 很小，实测最长 ~40 块），
        在 Python 里按 `seq` 切窗口 ⇒ **"相邻"由列表位置给出，缺号不破坏相邻**
        （这正是"仍然能相邻片段扩展"的落点）。

        ## `seq` 的定义

        `ROW_NUMBER() OVER (PARTITION BY user_id, session_id ORDER BY chunk_ordinal, local_index)`

        ⚠ **稠密序必须是 0-based 连续的**：`merge_segments` 靠 `seq == end + 1` 判相邻。
        `ROW_NUMBER()` 天然满足（有空洞的是 `chunk_ordinal`，不是 `seq`）。

        ⚠ **排序键是 `(chunk_ordinal, local_index)`，不是 `event_time`**：`event_time`
        在 session 内**没有区分度**（时间戳是 session 级的，两个数据集都如此），
        用它排序会让邻域产生抖动（§6.1）。**顺序来自请求里的 chunk 序号**
        ——这是 D25 与旧口径在"顺序从哪来"上的唯一区别，但那是根本区别。

        ⚠ 跨 session **永远看不到对方**（`WHERE` 里带着两个键），所以"禁止跨对话拼接"
        是结构性成立的，不靠调用方自觉（§2.2 的隔离契约 / §10）。

        ⚠ 走的是 `UNIQUE(user_id, session_id, chunk_ordinal, local_index)` 建出的索引
        （不变式 4）：`WHERE` 命中前两列作前缀，`ORDER BY` 命中后两列 ⇒ **不扫全表、不排序**。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS}, {_SEQ_COLUMN}"
            " FROM qa_pairs WHERE user_id = ? AND session_id = ?"
            " ORDER BY chunk_ordinal, local_index",
            (user_id, session_id),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def fetch_pairs_by_ids(self, conn: sqlite3.Connection, pair_ids: Sequence[str]) -> list[QaPair]:
        """按主键批量取正文（§6.3 的时序：Qdrant 出 `id` → 回这里取正文）。

        正文**不进** Qdrant payload；"取正文"这一步没有缓存层，也不应该有——
        正文的可信来源只有这一处。

        ⚠ **这里回来的行 `seq` 全是 `-1`**（`QaPair.seq` 的默认值）：`seq` 是
        **会话内**的稠密序，而"按主键取一批"跨 session、拿不到完整的会话上下文。
        需要 `seq` 的调用方（`rank/neighbor.py`）自己去 `fetch_session_ordered` 取——
        那边本来就要取整段来切窗口，顺带就能把 `id → seq` 补上。
        ⚠ **别在这里用窗口函数"顺手"算**：`ROW_NUMBER() OVER (PARTITION BY ...)`
        的窗口是 **`IN` 列表里那几行**，算出来的是"这一批里的第几个"，
        不是"会话里的第几个"——**数值看起来完全正常**，而那正是最坏的一种错。
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
        """枚举全部块（可选按 `user_id` 限定），按 `(session_id, chunk_ordinal, local_index)` 排序。

        ⚠ **只读访问器**，不改动任何既有语义。存在的理由是 §6.3 那条"Qdrant 可从 SQLite
        全量重建"：重建需要一次枚举，而让 `qdrant_store.py` 自己写 SQL 会破坏
        "store/ 是唯一接触 SQLite 的目录"。

        **不是检索路径**——检索按主键批量取正文走 `fetch_pairs_by_ids`。
        """
        if user_id is None:
            rows = conn.execute(
                f"SELECT {_COLUMNS}, {_SEQ_COLUMN} FROM qa_pairs"
                " ORDER BY user_id, session_id, chunk_ordinal, local_index"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS}, {_SEQ_COLUMN} FROM qa_pairs WHERE user_id = ?"
                " ORDER BY session_id, chunk_ordinal, local_index",
                (user_id,),
            ).fetchall()
        return [_to_pair(r) for r in rows]

    def assert_isolation(self, conn: sqlite3.Connection, user_id: str) -> list[QaPair]:
        """取某 user 的全部对（供隔离测试断言用）。

        `user_id` 是**唯一**的检索隔离字段；`session_id` 只是分组字段，
        **不是** Search 的过滤器（§2.2）。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS}, {_SEQ_COLUMN} FROM qa_pairs WHERE user_id = ?"
            " ORDER BY session_id, chunk_ordinal, local_index",
            (user_id,),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def explain_session_ordered(
        self, conn: sqlite3.Connection, user_id: str, session_id: str
    ) -> str:
        """会话有序查询的 `EXPLAIN QUERY PLAN`（测试用：断言它**不扫全表、不额外排序**）。"""
        rows = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ?"
            " ORDER BY chunk_ordinal, local_index",
            (user_id, session_id),
        ).fetchall()
        return " | ".join(str(r["detail"]) for r in rows)


_COLUMNS = (
    "id, user_id, session_id, chunk_ordinal, local_index,"
    " question, answer, status, event_time, request_id"
)

#: `seq` = **会话内稠密序**（0-based、连续）。它**不是列**，每次都现算。
#:
#: ⚠ **凡是要喂给扩窗 / 段合并的查询都必须带上它**——`QaPair.seq` 的默认值是 `-1`，
#: 而 `-1` 一旦流进 `merge_segments` 的 `seq == end + 1` 判断，会让**所有块各自成段**
#: （看起来像"检索质量差"，而不像 bug）。⇒ 这里用**一个片段**而不是各查询各写一份。
_SEQ_COLUMN = (
    # ⚠ `ROW_NUMBER()` 是 **1-based** 的，而本仓的 `seq` 语义是 **0-based**
    #   （`rank/neighbor.py` 的 `merge_segments` 拿它当列表下标用）⇒ **必须减 1**。
    #   不减的话两处会用**不同的基准**：合并的 `seq == end + 1` 判断照样自洽
    #   （1-based 也是稠密的），但 `expand_neighbors` 里"给候选补 seq"那条路径
    #   （`enumerate` 出来的 0-based 下标）与邻居的 1-based 会**混在同一个列表里**，
    #   于是相邻判断在**跨候选/邻居**处静默错位——只表现为"段切得碎了一点"。
    "(ROW_NUMBER() OVER (PARTITION BY user_id, session_id"
    " ORDER BY chunk_ordinal, local_index) - 1) AS seq"
)


def _to_pair(row: sqlite3.Row) -> QaPair:
    status = row["status"]
    if status not in _VALID_STATUSES:
        raise ValueError(f"库中出现非法 status: {status!r}")
    return QaPair(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        chunk_ordinal=row["chunk_ordinal"],
        local_index=row["local_index"],
        question=row["question"],
        answer=row["answer"],
        status=status,
        event_time=row["event_time"],
        request_id=row["request_id"],
        # `seq` 只有带 `_SEQ_COLUMN` 的查询才填得上（`fetch_pairs_by_ids` 就没带）。
        # 其余查询留 `-1`：它**不是列**，谁都不该拿它当持久字段用。
        seq=row["seq"] if "seq" in row.keys() else -1,  # noqa: SIM118 — Row 没有 __contains__
    )
