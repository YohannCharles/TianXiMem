from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from tianximem.facts.evidence import EVIDENCE_VERSION, FIRST_PERSON, EvidenceFact, key, relation_key

logger = logging.getLogger(__name__)

Status = Literal["complete", "pending"]

STATUS_COMPLETE: Final[Status] = "complete"
_VALID_STATUSES: Final[frozenset[str]] = frozenset({"complete", "pending"})

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

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

    return "".join(f"{len(p)}:{p}" for p in parts)


def make_pair_id(user_id: str, session_id: str, request_id: str, local_index: int) -> str:

    payload = _canonical(user_id, session_id, request_id, str(local_index))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_source_message_id(parent_memory_id: str, source_index: int) -> str:

    payload = _canonical("source-message", parent_memory_id, str(source_index))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QaPair:
    id: str
    user_id: str
    session_id: str
    request_id: str
    local_index: int
    prev_memory_id: str | None
    next_memory_id: str | None
    question: str | None
    answer: str | None
    status: Status
    event_time: int | None

    @property
    def is_complete(self) -> bool:

        return self.question is not None and self.answer is not None


@dataclass(frozen=True, slots=True)
class SourceMessage:
    source_index: int
    role: str
    content: str
    timestamp: int | None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _remap_legacy_rows(
    rows: Sequence[sqlite3.Row],
) -> list[tuple[str, str | None, str | None, sqlite3.Row]]:

    out: list[tuple[str, str | None, str | None, sqlite3.Row]] = []
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
    def __init__(self, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> None:
        self._db_path = str(db_path)
        self._busy_timeout_ms = busy_timeout_ms

    @classmethod
    def open(cls, db_path: str | Path, *, busy_timeout_ms: int = 5000) -> SqliteStore:

        path = Path(db_path)
        # 默认配置指向 `var/tianxi.db`，而 `var/` 是被 gitignore 的运行时目录：
        # 全新 checkout 里它不存在，SQLite 只会报 "unable to open database file"。
        path.parent.mkdir(parents=True, exist_ok=True)
        store = cls(path, busy_timeout_ms=busy_timeout_ms)
        store.init_schema()
        store.migrate()
        store._init_journal_mode()
        return store

    def migrate(self) -> bool:

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
            "The source database was migrated to request-scoped positions "
            "`(request_id, local_index)` and explicit Add-local adjacency.\n"
            "  All memory IDs changed, so existing Qdrant points are orphaned. "
            "Drop and rebuild the Qdrant collection; the content-addressed embedding cache remains reusable."
        )
        return True

    def _needs_migration(self, conn: sqlite3.Connection) -> bool:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(qa_pairs)")}
        return bool(columns) and ("chunk_ordinal" in columns or "prev_memory_id" not in columns)

    def _init_journal_mode(self) -> None:

        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:

        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute(f"PRAGMA busy_timeout = {int(self._busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def db_path(self) -> str:

        return self._db_path

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:

        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def read_snapshot(self) -> Iterator[sqlite3.Connection]:

        with self.read() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
            finally:
                conn.rollback()

    def init_schema(self) -> None:

        conn = self._connect()
        try:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            self._add_missing_fact_columns(conn)
        finally:
            conn.close()

    def _add_missing_fact_columns(self, conn: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(memory_facts)")}
        if columns and "supersedes" not in columns:
            conn.execute(
                "ALTER TABLE memory_facts ADD COLUMN supersedes INTEGER NOT NULL DEFAULT 0"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()
        finally:
            conn.close()

    def is_batch_applied(self, conn: sqlite3.Connection, request_id: str) -> bool:

        row = conn.execute(
            "SELECT 1 FROM applied_batches WHERE request_id = ? LIMIT 1", (request_id,)
        ).fetchone()
        return row is not None

    def applied_batch_payload_hash(self, conn: sqlite3.Connection, request_id: str) -> str | None:

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

    def insert_pair_messages(
        self,
        conn: sqlite3.Connection,
        parent_memory_id: str,
        messages: Sequence[tuple[int, str, str, int | None]],
    ) -> None:

        conn.executemany(
            "INSERT INTO qa_pair_messages(parent_memory_id,source_index,role,content,timestamp)"
            " VALUES (?,?,?,?,?) ON CONFLICT(parent_memory_id,source_index) DO NOTHING",
            [
                (parent_memory_id, index, role, content, timestamp)
                for index, role, content, timestamp in messages
            ],
        )

    def fetch_pair_messages(
        self, conn: sqlite3.Connection, parent_memory_id: str
    ) -> list[SourceMessage]:
        rows = conn.execute(
            "SELECT source_index,role,content,timestamp FROM qa_pair_messages"
            " WHERE parent_memory_id=? ORDER BY source_index",
            (parent_memory_id,),
        ).fetchall()
        return [
            SourceMessage(
                source_index=row["source_index"],
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def fetch_by_request(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> list[QaPair]:

        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs"
            " WHERE user_id = ? AND session_id = ? AND request_id = ?"
            " ORDER BY local_index",
            (user_id, session_id, request_id),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def fetch_pairs_by_ids(self, conn: sqlite3.Connection, pair_ids: Sequence[str]) -> list[QaPair]:

        if not pair_ids:
            return []
        placeholders = ",".join("?" * len(pair_ids))
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE id IN ({placeholders})",
            tuple(pair_ids),
        ).fetchall()
        by_id = {r["id"]: _to_pair(r) for r in rows}
        return [by_id[pid] for pid in pair_ids if pid in by_id]

    def fetch_unindexed_pairs(
        self, conn: sqlite3.Connection, user_id: str, *, version: str, limit: int
    ) -> list[QaPair]:
        if limit <= 0:
            return []
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id=? AND NOT EXISTS"
            " (SELECT 1 FROM evidence_coverage c WHERE c.parent_memory_id=qa_pairs.id"
            " AND c.user_id=qa_pairs.user_id AND c.version=?) ORDER BY id LIMIT ?",
            (user_id, version, limit),
        ).fetchall()
        return [_to_pair(row) for row in rows]

    def has_fact_index_coverage(
        self, conn: sqlite3.Connection, user_id: str, *, version: str
    ) -> bool:
        return not self.fetch_unindexed_pairs(conn, user_id, version=version, limit=1)

    def mark_fact_index_coverage(
        self, conn: sqlite3.Connection, pair: QaPair, *, version: str
    ) -> None:
        parent = conn.execute("SELECT user_id FROM qa_pairs WHERE id=?", (pair.id,)).fetchone()
        if parent is None or parent["user_id"] != pair.user_id:
            raise ValueError("扫描覆盖的原始来源缺失或属于其他用户")
        conn.execute(
            "INSERT INTO evidence_coverage(parent_memory_id,user_id,version) VALUES (?,?,?)"
            " ON CONFLICT(parent_memory_id,version) DO NOTHING",
            (pair.id, pair.user_id, version),
        )

    def insert_evidence(self, conn: sqlite3.Connection, facts: Sequence[EvidenceFact]) -> None:

        for fact in facts:
            parent = conn.execute(
                "SELECT user_id,question,answer FROM qa_pairs WHERE id=?", (fact.parent_memory_id,)
            ).fetchone()
            if parent is None or parent["user_id"] != fact.user_id:
                raise ValueError("共同事实的来源缺失或属于其他用户")
            if fact.source_side not in {"question", "answer"} or not fact.source_quote:
                raise ValueError("事实必须声明来源侧边并携带来源引句")
            item_quote = fact.get("item_quote")
            if item_quote is not None and (not isinstance(item_quote, str) or not item_quote):
                raise ValueError("物品定位引句必须是非空字符串")
            conn.execute(
                "INSERT INTO memory_facts(id,parent_memory_id,user_id,subject,subject_key,"
                "relation,relation_key,object,object_key,qualifiers,source_side,source_quote,"
                "statement,content,event_time,source_date,supersedes,version)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO NOTHING",
                (
                    fact.id,
                    fact.parent_memory_id,
                    fact.user_id,
                    fact.subject,
                    key(fact.subject),
                    fact.relation,
                    relation_key(fact.relation),
                    fact.object,
                    key(str(fact.get("match_object", fact.object))),
                    json.dumps(dict(fact.qualifiers), ensure_ascii=False, sort_keys=True),
                    fact.source_side,
                    fact.source_quote,
                    fact.statement,
                    fact.content,
                    fact.event_time,
                    fact.source_date,
                    1 if fact.supersedes else 0,
                    fact.version,
                ),
            )

    def owner_name(self, conn: sqlite3.Connection, user_id: str) -> str:

        row = conn.execute(
            "SELECT subject_key FROM memory_facts"
            " WHERE user_id=? AND version=? AND relation_key='name'"
            " ORDER BY source_date, id LIMIT 1",
            (user_id, EVIDENCE_VERSION),
        ).fetchone()
        return row[0] if row else ""

    def _owner_display(self, conn: sqlite3.Connection, user_id: str) -> str:

        row = conn.execute(
            "SELECT subject FROM memory_facts"
            " WHERE user_id=? AND version=? AND relation_key='name'"
            " ORDER BY source_date, id LIMIT 1",
            (user_id, EVIDENCE_VERSION),
        ).fetchone()
        return row[0] if row else ""

    def fetch_facts_for_entities(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        entities: Sequence[str],
        *,
        limit: int,
    ) -> list[EvidenceFact]:

        keys = tuple(dict.fromkeys(key(entity) for entity in entities if entity.strip()))
        if not keys:
            return []
        # 查询里常用 "user" 指代说话人，而事实主语存的是 "I" / "我"。只在这种
        # 指代出现时补上别名，避免任意实体查询都扩大到全部首人称事实。
        if "user" in keys or "用户" in keys:
            keys = (*keys, *(alias for alias in ("i", "我") if alias not in keys))
        if limit <= 0:
            return []
        placeholders = ",".join("?" * len(keys))

        # 先找出被命中的「主体-关系」分组，再回表取这些分组的完整取值历史，
        # 否则被取代的旧值会因为实体不再命中而看不到，无法在读取侧裁决。
        rows = conn.execute(
            "SELECT f.* FROM memory_facts f"
            " WHERE f.user_id=? AND f.version=?"
            " AND EXISTS ("
            "  SELECT 1 FROM memory_facts g JOIN qa_pairs p ON p.id=g.parent_memory_id"
            "  WHERE g.user_id=? AND p.user_id=? AND g.version=?"
            "    AND g.subject_key=f.subject_key AND g.relation_key=f.relation_key"
            f"    AND (g.subject_key IN ({placeholders}) OR g.object_key IN ({placeholders}))"
            " )"
            " ORDER BY f.event_time IS NULL,f.event_time,f.parent_memory_id,f.id",
            (user_id, EVIDENCE_VERSION, user_id, user_id, EVIDENCE_VERSION, *keys, *keys),
        ).fetchall()
        owner_display = self._owner_display(conn, user_id)
        return [
            EvidenceFact(
                id=row["id"],
                parent_memory_id=row["parent_memory_id"],
                user_id=row["user_id"],
                subject=(
                    owner_display
                    if owner_display and row["subject_key"] in FIRST_PERSON
                    else row["subject"]
                ),
                relation=row["relation"],
                object=row["object"],
                source_quote=row["source_quote"],
                source_side=row["source_side"],
                statement=_rebind(row["statement"], row["statement"], owner_display),
                content=_rebind(row["content"], row["statement"], owner_display),
                event_time=row["event_time"],
                source_date=row["source_date"],
                qualifiers=tuple(sorted(json.loads(row["qualifiers"]).items())),
                version=row["version"],
            )
            for row in _apply_supersession(rows)[:limit]
        ]

    def iter_pairs(self, conn: sqlite3.Connection, *, user_id: str | None = None) -> list[QaPair]:

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

        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM qa_pairs WHERE user_id = ?"
            " ORDER BY session_id, request_id, local_index",
            (user_id,),
        ).fetchall()
        return [_to_pair(r) for r in rows]

    def explain_by_request(
        self, conn: sqlite3.Connection, user_id: str, session_id: str, request_id: str
    ) -> str:

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


def _fact_order(row: sqlite3.Row) -> tuple[int, int, str, str]:

    event_time = row["event_time"]
    return (
        0 if event_time is not None else 1,
        0 if event_time is None else int(event_time),
        str(row["parent_memory_id"]),
        str(row["id"]),
    )


def _apply_supersession(rows: Sequence[sqlite3.Row]) -> list[sqlite3.Row]:
    """读取时裁决同一「主体-关系」的取值历史。

    写入侧全量保留；这里只决定返回哪一部分：取时间最新的更新批次，
    丢弃该时点之前的普通陈述，保留更新批次本身，以及该时点之后且与新版
    一致（同一 object_key）的普通陈述。分组内没有更新句时原样返回。
    时间缺失的陈述无法定位在更新时点前后，保守保留。
    """

    by_group: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        by_group.setdefault((row["subject_key"], row["relation_key"]), []).append(row)

    kept: list[sqlite3.Row] = []
    for group in by_group.values():
        updates = [row for row in group if row["supersedes"] and row["event_time"] is not None]
        if not updates:
            kept.extend(group)
            continue
        cutoff = max(int(row["event_time"]) for row in updates)
        current = {row["object_key"] for row in updates if int(row["event_time"]) == cutoff}
        for row in group:
            event_time = row["event_time"]
            if event_time is None:
                kept.append(row)
                continue
            instant = int(event_time)
            if (row["supersedes"] and instant == cutoff) or (
                instant >= cutoff and row["object_key"] in current
            ):
                kept.append(row)
    kept.sort(key=_fact_order)
    return kept


def _rebind(text: str, subject_key: str, owner: str) -> str:

    if not owner or not text:
        return text
    marker = "\nSource quotation:"
    head, sep, tail = text.partition(marker)
    rebound = re.sub(r"\b(?:I['’]s|My|my)\b", f"{owner}'s", head, count=1)
    if rebound != head:
        return rebound + sep + tail
    if head.startswith("I "):
        head = owner + head[1:]
    return head + sep + tail
