"""SQLite 真源（PRD §6.1 / §6.3）。

**这是唯一接触 SQLite 的模块。** 上层拿到的是领域对象 `QaPair`，不是 `sqlite3.Row`
——§6.3 的分工表只有在读写收口到一处时才守得住。

本模块【不认识】Qdrant、embedding、渲染，也【不认识】任何数据集——
它只认 §2.1 的 canonical Add 契约字段（`user_id` / `session_id` / `request_id` / 消息）。

不变式归属（store/CLAUDE.md 的五条）：
  1. `id` 位置派生 .................... `make_pair_id()` + `schema.sql` 的 PRIMARY KEY
  2. `pair_idx` session 内连续 ........ `next_pair_idx()`（读-改-写在**一个事务**里）
  3. `applied_batches` 只增不改 ....... `record_batch()`（只 INSERT，不 UPDATE）
  4. UNIQUE 索引 = 邻域查询的键 ....... `schema.sql` 的 UNIQUE + `fetch_pairs_by_idx_range()`
  5. 缓存键 = 渲染文本哈希 ............ **不在本模块**，属 `embed/`（§7.2）
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


def make_pair_id(user_id: str, session_id: str, pair_idx: int) -> str:
    """不变式 1：`id` **位置派生** = hash(user_id, session_id, pair_idx)。（§6.1）

    ⚠ **绝不能用内容哈希。** 补全（`pending` → `complete`）会改写 `question` / `answer`，
    内容哈希会随之变化 ⇒ 同一个对拿到新 `id` ⇒ Qdrant 里留下一个**孤儿 point**
    （旧的还在，新的也写进去，检索命中"不存在的那个版本"，**且不报错**）。

    这也正是 `pair_idx` 必须连续的原因：`id` 依赖它。

    哈希算法本身**不是 §6.1 规定的**——规格只规定了**输入**是这三个字段。
    这里取 SHA-256 全 64 位十六进制：确定性、无碰撞顾虑、长度对 TEXT 主键无所谓。
    """
    payload = _canonical(user_id, session_id, str(pair_idx))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QaPair:
    """一个 QA 对的领域对象（§6.2）。"""

    id: str
    user_id: str
    session_id: str
    pair_idx: int
    question: str | None
    answer: str | None
    status: Status
    event_time: int | None
    request_id: str


def _now_ms() -> int:
    return int(time.time() * 1000)


class SqliteStore:
    """`qa_pairs` + `applied_batches` 的读写与事务。

    ## 连接模型：**短生命周期**（2026-09-24 改）

    本类**只保存 `db_path` 与存储逻辑**，**不长期持有** `sqlite3.Connection`。
    连接的生命周期恰好是**一次逻辑操作**：

    ```text
    写： connect → BEGIN IMMEDIATE → 读改写 → COMMIT / ROLLBACK → close
    读： connect → SELECT                                        → close
    ```

    **为什么必须这样**：FastAPI 的 `def` 路由跑在**线程池**里，而 `sqlite3` 的连接
    **只能被创建它的线程使用**。任何"启动时建一个连接、之后长期复用"的写法，
    到线上就是**每个请求都抛** `sqlite3.ProgrammingError`（2026-09-24 实测）。
    短生命周期把这个问题**从根上消掉**——连接永远在同一次调用的同一个线程里建与关，
    于是"连接与线程的约束"**不需要泄漏到任何上层**。

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

        ⚠ **`journal_mode = WAL` 只在这里执行一次**，不在每个短生命周期连接上重复：
        它写进数据库文件、对后续所有连接持续生效（`PRAGMA journal_mode` 返回的是
        设置**之后**的模式，所以重复执行只是白跑一次 IO）。
        与之相对，`busy_timeout` / `synchronous` / `foreign_keys` / `row_factory`
        是**每连接**的，必须每次新连接都设——见 `_connect()`。
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

        * **读-改-写**：`pair_idx` 的分配是 `MAX(pair_idx)` → 写位置。deferred `BEGIN`
          在第一次**写**的时候才拿写锁，两个并发事务会先读到同一个 `MAX(pair_idx)`
          ⇒ 位置撞车（撞 `UNIQUE`）。
        * **升级写锁时 `busy_timeout` 不生效**：deferred 事务已持读锁，要写时得**升级**成
          写锁；若对方正持写锁，SQLite **立刻**返回 `SQLITE_BUSY`（"database is locked"）
          ——它宁可立刻报错也不冒死锁的险，**不应用 `busy_timeout`**。

        2026-09-24 实测：把这一行换成 `BEGIN`，`tests/test_store.py` 的两条并发压力用例
        **双双失败**，报 `OperationalError('database is locked')` ×5——**而且跨 session
        那条也失败**（不同 `(user_id, session_id)` 一样撞），所以这不只是"位置撞车"的防护。

        IMMEDIATE 在事务开头就拿写锁，把整个读-改-写串行化，也让并发写事务退化成
        "排队"而不是"互锁"。

        ⚠ 这只解决**单库内**的并发。§15 还要求 Add 按 `(user_id, session_id)` 串行化
        （进程内按 session 的锁），那一层属 `service/`，不在本模块。

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

    # ── 第 2 步：恢复位置（§6.5）────────────────────────────────────────

    def next_pair_idx(self, conn: sqlite3.Connection, user_id: str, session_id: str) -> int:
        """`next_idx = COALESCE(MAX(pair_idx) + 1, 0)`，限于该 `(user_id, session_id)`。

        ⚠ **绝不能从 0 重开**：`id` 是位置派生的，重开必然与既有行撞 `id`，
        而写入是 upsert ⇒ **静默覆盖上一批的数据**。
        """
        row = conn.execute(
            "SELECT COALESCE(MAX(pair_idx) + 1, 0) FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ).fetchone()
        return int(row[0])

    def open_pair(self, conn: sqlite3.Connection, user_id: str, session_id: str) -> QaPair | None:
        """该 session 中**本批可以续写**的那一对（§6.5 第 2 步的"恢复位置"）。

        两个析取项，缺一不可：

        * `status = 'pending'` —— §6.5 的原有判据：上一批命中了上限，答话可能还没写完。
        * **`answer IS NULL`** —— 本轮的**内容判据**：一个对只要还没收到任何非 user 消息，
          它的 `question` 就**还没写完** ⇒ 后续的 user 消息是它的续写（配对规则见
          [`../pairing/pairing.py`](../pairing/pairing.py) 的 `question_is_open`）。

        ⚠ **第二个析取项不能省，也不能只靠第一个。** `pending` 是从"本批是否命中上限"
        推出来的，而上限里的**词数计数我们复现不了**（S2："Adapter 计的词"官方从未定义）。
        若 AML 按它的口径切出接近但不足 2,000 词的碎片、我们算出更少，这一对就会被标成
        `complete`，下一批的碎片再也接不上——**静默退化成"有问无答"的对**。
        `answer IS NULL` 是**内容事实**，与任何计数无关。

        至多一个，且**必在末尾**，所以取 `ORDER BY pair_idx DESC LIMIT 1`：
        在配对规则下，一个 `answer IS NULL` 的对会吃掉后续的 user（并入 `question`）
        与非 user（并入 `answer`），所以它只能是最后一对。
        （⚠ 这条不变式是**本轮规则**带来的：旧规则下 `q1 q2 a` 会产出非末尾的 `(q1, NULL)`，
        所以按旧规则写过的库不满足它——跨 arm 必须用干净的库，见 V9。）
        """
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND (status = ? OR answer IS NULL)"
            " ORDER BY pair_idx DESC LIMIT 1",
            (user_id, session_id, STATUS_PENDING),
        ).fetchone()
        return None if row is None else _to_pair(row)

    # ── 第 3 步：写（填空 + 追加，绝不覆盖 —— §6.5）──────────────────────

    def insert_pair(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        session_id: str,
        pair_idx: int,
        question: str | None,
        answer: str | None,
        status: Status,
        event_time: int | None,
        request_id: str,
        pair_id: str | None = None,
    ) -> QaPair:
        """新建一个对。`id` 由位置派生，调用方通常不必传 `pair_id`。"""
        if status not in _VALID_STATUSES:
            raise ValueError(f"非法 status: {status!r}（只允许 'complete' | 'pending'）")
        pid = make_pair_id(user_id, session_id, pair_idx) if pair_id is None else pair_id
        conn.execute(
            "INSERT INTO qa_pairs"
            " (id, user_id, session_id, pair_idx, question, answer, status, event_time, request_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, user_id, session_id, pair_idx, question, answer, status, event_time, request_id),
        )
        return QaPair(
            id=pid,
            user_id=user_id,
            session_id=session_id,
            pair_idx=pair_idx,
            question=question,
            answer=answer,
            status=status,
            event_time=event_time,
            request_id=request_id,
        )

    def append_question(self, conn: sqlite3.Connection, pair_id: str, text: str) -> bool:
        """`question` **填空 + 追加**：原值为空则填入，否则**追加**（append-only）。

        追加必须允许，理由与 `answer` 同源：一个对的 `question` 现在是一段**连续 user
        消息**的拼接，而这段消息**可以跨批次**——续接批次带来的 user 消息是它的续写，
        必须并进去而不是丢掉（配对规则见 `../pairing/pairing.py`）。

        ⚠ 旧版的 `fill_question_if_null`（"只在原值为 NULL 时写入"）**已删除**：
        那条规则的前提是"一个 `question` 恰好来自一条 user 消息"，而这个前提**正是本轮
        修正的东西**（AML 可能把一条超长 user 消息按句边界切成多条同 role 消息）。
        `question` 与 `answer` 现在是同一个写模式，所以共用同一种写法——
        包括把空串当作空值处理，让"填空"与"追加"的边界与 `append_answer` 逐字一致。

        追加的安全性由**批次级守卫**保证（同一批至多被应用一次），
        所以这里**不需要**额外判重——与 `append_answer` 同一套论证（见 D4）。
        """
        if not text:
            return False
        cur = conn.execute(
            "UPDATE qa_pairs"
            " SET question = CASE"
            "   WHEN question IS NULL OR question = '' THEN ?"
            "   ELSE question || char(10) || ?"
            " END"
            " WHERE id = ?",
            (text, text, pair_id),
        )
        return cur.rowcount > 0

    def append_answer(self, conn: sqlite3.Connection, pair_id: str, text: str) -> bool:
        """`answer` **填空 + 追加**：原值为空则填入，否则**追加**（append-only）。

        追加必须允许（§6.5），因为 §6.2 承认"一条 user 后跟多条非 user 消息"，
        这类对若跨批次，续接批次带来的消息必须并进去而不是丢掉。

        追加的安全性由**批次级守卫**保证（同一批至多被应用一次），
        所以这里**不需要**额外判重——两者解决的是两个不同的问题（见 D4）。
        """
        if not text:
            return False
        # 连接符是单个换行（char(10)）。AML 侧只做 "\n".join(...)、不插分隔符（§11.3），
        # 所以这里用换行，保证渲染出来的 A: 块就是 §11.3 示例的形状。
        cur = conn.execute(
            "UPDATE qa_pairs"
            " SET answer = CASE"
            "   WHEN answer IS NULL OR answer = '' THEN ?"
            "   ELSE answer || char(10) || ?"
            " END"
            " WHERE id = ?",
            (text, text, pair_id),
        )
        return cur.rowcount > 0

    def mark_complete(self, conn: sqlite3.Connection, pair_id: str) -> bool:
        """`status` **只允许 pending → complete，不允许反向**。

        `WHERE status = 'pending'` 是单向性的落点：对已经 complete 的行调用它是
        **空操作**（返回 False），且**没有任何方法**能把状态写回 pending。
        """
        cur = conn.execute(
            "UPDATE qa_pairs SET status = ? WHERE id = ? AND status = ?",
            (STATUS_COMPLETE, pair_id, STATUS_PENDING),
        )
        return cur.rowcount > 0

    def touch_request_id(self, conn: sqlite3.Connection, pair_id: str, request_id: str) -> bool:
        """把该行的 `request_id` 改写为最近触碰它的那一批。

        §6.1：`request_id` 记的是"最后触碰该行的 Add"，**只用于溯源**。
        ⚠ 它被后一批覆盖正是幂等守卫**不能**复用它做判重的原因（D4）。
        """
        cur = conn.execute("UPDATE qa_pairs SET request_id = ? WHERE id = ?", (request_id, pair_id))
        return cur.rowcount > 0

    # ── 读 ─────────────────────────────────────────────────────────────

    def fetch_pairs_by_idx_range(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        lo: int,
        hi: int,
    ) -> list[QaPair]:
        """邻域查询（§10 的 ±1 扩窗）。**不变式 4 的落点。**

        `WHERE user_id = ? AND session_id = ? AND pair_idx BETWEEN ? AND ?`
        **正好命中** `UNIQUE(user_id, session_id, pair_idx)` 建出的索引——
        扩窗从 ±1 改成 ±2 只需改这里的界，不动 schema。

        `ORDER BY pair_idx` 而不是 `event_time`：`event_time` 在 session 内
        **没有区分度**（时间戳是 session 级的，两个数据集都如此），用它排序
        会让 ±1 邻域产生抖动，进而让消融实验不可复现（§6.1）。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND pair_idx BETWEEN ? AND ?"
            " ORDER BY pair_idx",
            (user_id, session_id, lo, hi),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def fetch_pairs_by_ids(self, conn: sqlite3.Connection, pair_ids: Sequence[str]) -> list[QaPair]:
        """按主键批量取正文（§6.3 的时序：Qdrant 出 `id` → 回这里取正文）。

        正文**不进** Qdrant payload；"取正文"这一步没有缓存层，也不应该有——
        正文的可信来源只有这一处。
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
        """枚举全部对（可选按 `user_id` 限定），按 `(session_id, pair_idx)` 排序。

        ⚠ **② 新增的只读访问器**，不改动 ① 的任何既有语义。存在的理由是 §6.3 那条
        "Qdrant 可从 SQLite 全量重建"：重建需要一次枚举，而让 `qdrant_store.py`
        自己写 SQL 会破坏"store/ 是唯一接触 SQLite 的目录"。

        **不是检索路径**——检索按主键批量取正文走 `fetch_pairs_by_ids`。
        """
        if user_id is None:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM qa_pairs ORDER BY session_id, pair_idx"
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id = ? ORDER BY session_id, pair_idx",
                (user_id,),
            ).fetchall()
        return [_to_pair(r) for r in rows]

    def assert_isolation(self, conn: sqlite3.Connection, user_id: str) -> list[QaPair]:
        """取某 user 的全部对（供隔离测试断言用）。

        `user_id` 是**唯一**的检索隔离字段；`session_id` 只是分组字段，
        **不是** Search 的过滤器（§2.2）。
        """
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id = ? ORDER BY session_id, pair_idx",
            (user_id,),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def explain_idx_range(
        self, conn: sqlite3.Connection, user_id: str, session_id: str, lo: int, hi: int
    ) -> str:
        """邻域查询的 `EXPLAIN QUERY PLAN`（测试用：断言它**不扫全表**）。"""
        rows = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND pair_idx BETWEEN ? AND ?",
            (user_id, session_id, lo, hi),
        ).fetchall()
        return " | ".join(str(r["detail"]) for r in rows)


_COLUMNS = "id, user_id, session_id, pair_idx, question, answer, status, event_time, request_id"


def _to_pair(row: sqlite3.Row) -> QaPair:
    status = row["status"]
    if status not in _VALID_STATUSES:
        raise ValueError(f"库中出现非法 status: {status!r}")
    return QaPair(
        id=row["id"],
        user_id=row["user_id"],
        session_id=row["session_id"],
        pair_idx=row["pair_idx"],
        question=row["question"],
        answer=row["answer"],
        status=status,
        event_time=row["event_time"],
        request_id=row["request_id"],
    )
