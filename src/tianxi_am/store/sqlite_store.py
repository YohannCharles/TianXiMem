"""SQLite 真源（PRD §6.1 / §6.3）。

**这是唯一接触 SQLite 的模块。** 上层拿到的是领域对象 `QaPair`，不是 `sqlite3.Row`
——§6.3 的分工表只有在读写收口到一处时才守得住。

本模块【不认识】Qdrant、embedding、渲染，也【不认识】任何数据集——
它只认 §2.1 的 canonical Add 契约字段（`user_id` / `session_id` / `request_id` / 消息）。

不变式归属（store/README.md 的五条）：
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

    **事务边界由调用方持有**：所有写方法都要求传入一个 `conn`，并且必须在
    `transaction()` 里调用。这样 §6.5 的三步（守卫查表 → 位置恢复 → 挂接 + 记录本批）
    才能落在**同一个事务**里——否则崩溃在中间会留下半个批次。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ── 生命周期 ────────────────────────────────────────────────────────

    @classmethod
    def open(cls, db_path: str | Path) -> SqliteStore:
        """连接 + 建表，一步到位（建表是幂等的）。"""
        store = cls(db_path)
        store.connect()
        store.init_schema()
        return store

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self._db_path, isolation_level=None)  # 自己管事务
        # WAL：读写不互相阻塞，且崩溃后能恢复。
        conn.execute("PRAGMA journal_mode = WAL")
        # synchronous=FULL：契约要求"持久化完成且立即可搜索"后才能响应（§2.1）。
        # 用 NORMAL 会在 OS/断电级崩溃下丢掉最后几个已提交事务——那正是"重试"的来源，
        # 虽然幂等守卫兜得住，但没有理由在这里省。
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.row_factory = sqlite3.Row
        self._conn = conn
        return conn

    @property
    def db_path(self) -> str:
        """真源文件路径。

        §6.3：**SQLite 是唯一的真源，也是唯一不可重建物**——Qdrant 可以从它全量重建。
        测试用它来模拟"进程重启后重连同一个库"。
        """
        return self._db_path

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("store 尚未 connect()")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SqliteStore:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """执行 `schema.sql`（`IF NOT EXISTS`，可重复调用）。"""
        self.connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """一个写事务。

        **`BEGIN IMMEDIATE` 而不是默认的 `BEGIN`**：`pair_idx` 的分配是**读-改-写**
        （`MAX(pair_idx)` → 写位置）。默认事务在第一次**写**的时候才拿写锁，
        两个并发批次会先读到同一个 `MAX(pair_idx)`、再先后写入 ⇒ 位置撞车。
        IMMEDIATE 在事务开头就拿写锁，把读-改-写整体串行化。

        ⚠ 这只解决**单库内**的并发。§15 还要求 Add 按 `(user_id, session_id)` 串行化
        （进程内按 session 的锁），那一层属 `service/`，不在本模块。
        """
        conn = self.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

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

    def pending_pair(
        self, conn: sqlite3.Connection, user_id: str, session_id: str
    ) -> QaPair | None:
        """该 session 中 `status = 'pending'` 的那一对。

        至多一个，且必在末尾——所以取 `ORDER BY pair_idx DESC LIMIT 1`。
        """
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND status = ?"
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

    def fill_question_if_null(self, conn: sqlite3.Connection, pair_id: str, question: str) -> bool:
        """`question` **只在原值为 NULL 时**写入。返回是否真的写了。

        ① 里没有任何代码路径会用到它——新建的对总是带着 question 落库，
        而"无问的对"的 question 永远保持 NULL（它会被下一条 user 消息关闭，
        那条 user 消息开的是一个**新**对）。保留它是为了让 §6.5 的写入规则
        有一个可被直接断言的落点，而不是只写在文档里。
        """
        cur = conn.execute(
            "UPDATE qa_pairs SET question = ? WHERE id = ? AND question IS NULL",
            (question, pair_id),
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
