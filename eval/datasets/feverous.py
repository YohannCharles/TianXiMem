"""FEVEROUS dev：claim-only 页面检索 → 整页语料 Add → Search → 标签及证据判分。

本地候选生成器在完整 Wikipedia SQLite 上建 title/前五句 FTS5 索引，按 claim
的非停用词做 BM25，固定取前五页。生成器完全不读取 label、evidence 或标注者
操作；漏召页面不由 gold 补齐。候选整页的句子、表格、列表及原始元素 ID 进入 Add。
limit 数 claim，spread 按 label/challenge 分层（仅用于选题）。

评分复用上游标签与完整证据组函数（包括 NEI），句子/单元格数量按上游截断。
这个固定候选检索步骤是本地适配，不是让被测服务对全 Wikipedia 进行检索，
也不是 AML 线上候选池的复刻；其严格分不能与完整官方系统或 AML 榜分对齐。
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from .preprocess import Message, Question, Sample, Session, normalize_content
from .sampling import stratified_sample

DATA_DIR = "feverous"
ANNOTATIONS = "feverous_dev_challenges.jsonl"
DATABASE = "feverous_wikiv1.db"
INDEX = ".cache/title-intro-v1.sqlite"
INDEX_VERSION = "feverous-title-intro-v1"
PAGE_COUNT = 5
SCORER = "feverous/feverous_scorer.py"
ANSWER_CONTRACT = "feverous-claim-pages-evidence-v3"
LABELS = frozenset({"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"})
SHAPE_NOTE = (
    f"FEVEROUS dev ({ANSWER_CONTRACT}): fixed claim-only {INDEX_VERSION} candidate "
    f"retrieval (top {PAGE_COUNT} pages) from full upstream SQLite; no gold evidence "
    "or annotator paths select corpus pages. Entire candidate pages with original element "
    "IDs enter Add; only Search results enter answer. Invalid answer JSON/labels or invisible "
    "element IDs receive at most one model repair using the same Search context, without gold "
    "or scoring feedback; a small guide lists visible IDs with similar spellings, without "
    "automatically replacing citations. Each model call has a 360-second local timeout. "
    "Both attempts are retained. Pinned upstream "
    "label/evidence-group "
    "scoring applies to all labels including NEI. This local bounded candidate retrieval "
    "does not reproduce full-Wikipedia retrieval by the tested service or AML candidate pools."
)

# 候选生成的固定口径，进入 INDEX_VERSION/SHAPE_NOTE；不由金标或分数调节。
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
        "whose",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
        "than",
        "also",
        "one",
        "all",
        "only",
        "not",
        "no",
        "more",
        "most",
        "first",
        "second",
        "third",
        "during",
        "after",
        "before",
        "according",
        "about",
        "between",
        "into",
        "over",
        "under",
        "when",
        "where",
        "while",
        "can",
        "could",
        "would",
        "will",
    ]
)
_ELEMENT_ID = re.compile(r"^(.+?)_(header_cell|table_caption|sentence|cell|item)_(\d+(?:_\d+)*)$")


def evidence_triplet(element_id: str) -> list[str]:
    match = _ELEMENT_ID.fullmatch(element_id)
    if not match:
        raise ValueError(f"FEVEROUS: invalid evidence element ID {element_id!r}")
    return list(match.groups())


def corpus_path(root: Path) -> Path:
    path = root / DATA_DIR / DATABASE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run make fetch-data DATASET=feverous")
    return path


@contextmanager
def _readonly(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def _source_identity(root: Path, source_sha256: str) -> dict[str, str]:
    stat = corpus_path(root).stat()
    return {
        "version": INDEX_VERSION,
        "source_sha256": source_sha256,
        "source_bytes": str(stat.st_size),
        "source_mtime_ns": str(stat.st_mtime_ns),
    }


def validate_index(root: Path, source_sha256: str) -> None:
    path = root / DATA_DIR / INDEX
    try:
        with _readonly(path) as conn:
            metadata = dict(conn.execute("SELECT key, value FROM metadata"))
            conn.execute("SELECT page FROM pages LIMIT 1").fetchone()
        if any(metadata.get(k) != v for k, v in _source_identity(root, source_sha256).items()):
            raise ValueError("index source/version changed")
    except (sqlite3.Error, ValueError) as error:
        raise ValueError(
            f"FEVEROUS: missing/stale index {path}; run make fetch-data DATASET=feverous"
        ) from error


def _intro(page: dict) -> str:
    sentences = [page[key] for key in page.get("order", []) if key.startswith("sentence_")][:5]
    return " ".join(s for s in sentences if isinstance(s, str))


def prepare_index(root: Path, source_sha256: str, *, check_only: bool = False) -> None:
    """CLI 专用派生索引构建；原数据库只读，临时索引完成后原子发布。"""
    try:
        validate_index(root, source_sha256)
        return
    except ValueError:
        if check_only:
            raise
    source = corpus_path(root)
    target = root / DATA_DIR / INDEX
    staging_root = root / ".tmp"
    staging_root.mkdir(parents=True, exist_ok=True)
    print("  FEVEROUS: building claim-only title/intro FTS index", flush=True)
    with tempfile.TemporaryDirectory(prefix="feverous-index-", dir=staging_root) as tmp:
        staged = Path(tmp) / "index.sqlite"
        with _readonly(source) as src, closing(sqlite3.connect(staged)) as dst:
            dst.execute("PRAGMA journal_mode=OFF")
            dst.execute("PRAGMA synchronous=OFF")
            dst.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            dst.execute(
                "CREATE VIRTUAL TABLE pages USING fts5("
                "page UNINDEXED, title, intro, tokenize='unicode61')"
            )
            batch = []
            count = 0
            for page_id, data in src.execute("SELECT id, data FROM wiki"):
                if not isinstance(page_id, str) or not page_id.strip() or not data:
                    continue
                page = json.loads(data)
                batch.append((page_id, page_id.replace("_", " "), _intro(page)))
                if len(batch) == 1000:
                    dst.executemany("INSERT INTO pages VALUES (?, ?, ?)", batch)
                    count += len(batch)
                    batch.clear()
                    if count % 100000 == 0:
                        print(f"  FEVEROUS: indexed {count:,} pages", flush=True)
            dst.executemany("INSERT INTO pages VALUES (?, ?, ?)", batch)
            count += len(batch)
            if count == 0:
                raise ValueError("FEVEROUS: empty wiki database")
            dst.execute("INSERT INTO pages(pages) VALUES ('optimize')")
            dst.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                list(_source_identity(root, source_sha256).items()) + [("pages", str(count))],
            )
            dst.commit()
        target.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(target)
    print(f"  FEVEROUS: index ready ({count:,} pages)", flush=True)


def candidate_pages(index: sqlite3.Connection, claim: str) -> list[str]:
    """只有 claim 进入该函数；SQL 参数化、稳定排序，不依赖任何金标。"""
    claim = unicodedata.normalize("NFC", claim)
    terms = list(dict.fromkeys(t.casefold() for t in re.findall(r"[^\W_]+", claim)))
    terms = [term for term in terms if term not in _STOPWORDS and len(term) > 1]
    if not terms:
        return []
    query = " OR ".join(f'"{term}"' for term in terms)
    # FTS5 的 rank 游标可提前停止，比对全部命中行调用 bm25 再排序快。
    # 读完边界上的同分行再按 page 排序，保持 bm25(...), page 的确定性口径。
    matches: list[tuple[str, float]] = []
    with closing(
        index.execute(
            "SELECT page, rank FROM pages WHERE pages MATCH ? "
            "AND rank MATCH 'bm25(0, 8, 1)' ORDER BY rank",
            (query,),
        )
    ) as cursor:
        for page, rank in cursor:
            if len(matches) >= PAGE_COUNT and rank != matches[PAGE_COUNT - 1][1]:
                break
            matches.append((page, rank))
    matches.sort(key=lambda match: (match[1], match[0]))
    return [page for page, _ in matches[:PAGE_COUNT]]


def _value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("FEVEROUS: wiki element text must be a string")
    return value.strip()


def render_page(page_id: str, page: dict) -> tuple[Message, ...]:
    """保留原始元素 ID、表格结构和全页文本；不接收题目或 gold。"""
    texts = [f"Page: {page_id}"]
    section = ""
    for key in page["order"]:
        value = page[key]
        prefix = f"Page: {page_id}" + (f"\nSection: {section}" if section else "")
        if key.startswith("section_"):
            section = _value(value["value"])
            texts.append(f"{prefix}\nSection: {section}")
        elif key.startswith("sentence_"):
            texts.append(f"{prefix}\n[{page_id}_{key}] {_value(value)}")
        elif key.startswith("table_"):
            lines = [prefix, f"Table: {key}"]
            if value.get("caption"):
                caption_id = f"{page_id}_table_caption_{key.removeprefix('table_')}"
                lines.append(f"[{caption_id}] {_value(value['caption'])}")
            for row_index, row in enumerate(value["table"]):
                lines.append(f"Row {row_index}:")
                for cell in row:
                    cell_id = cell["id"]
                    evidence_triplet(f"{page_id}_{cell_id}")
                    lines.append(
                        f"[{page_id}_{cell_id}] {_value(cell['value'])} "
                        f"(row_span={cell['row_span']}, column_span={cell['column_span']})"
                    )
            texts.append("\n".join(lines))
        elif key.startswith("list_"):
            lines = [prefix, f"List: {key}"]
            for item in value["list"]:
                element_id = f"{page_id}_{item['id']}"
                evidence_triplet(element_id)
                lines.append(f"[{element_id}] {_value(item['value'])} (level={item['level']})")
            texts.append("\n".join(lines))
        else:
            raise ValueError(f"FEVEROUS {page_id}: unsupported wiki element {key!r}")
    return tuple(
        Message(role="user", content=normalize_content(text, where=f"FEVEROUS {page_id}"))
        for text in texts
    )


def read_annotations(root: Path) -> list[dict]:
    with (root / DATA_DIR / ANNOTATIONS).open(encoding="utf-8") as handle:
        raw = [json.loads(line) for line in handle if line.strip()]
    # 上游第一行是全空字符串的格式示例；只跳过这个确切形状。
    rows = [row for row in raw if not (row and all(value == "" for value in row.values()))]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("FEVEROUS: duplicate claim ID")
    for row in rows:
        if type(row["id"]) is not int or row["label"] not in LABELS:
            raise ValueError("FEVEROUS: invalid claim ID/label")
    return rows


def source_sha256() -> str:
    from .manifest import MANIFEST

    return next(e["sha256"] for e in MANIFEST if e["name"] == f"{DATA_DIR}/{DATABASE}")


def load_feverous(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    root = Path(bench_dir)
    rows = read_annotations(root)
    if limit is not None:
        if limit < 0:
            raise ValueError("FEVEROUS: limit must be nonnegative")
        if limit == 0:
            return []
        rows = (
            stratified_sample(rows, limit, key=lambda row: f"{row['label']}/{row['challenge']}")
            if spread
            else rows[:limit]
        )
    validate_index(root, source_sha256())
    samples = []
    with _readonly(root / DATA_DIR / INDEX) as index, _readonly(corpus_path(root)) as wiki:
        for row in rows:
            user_id = f"feverous-{row['id']}-{INDEX_VERSION}-p{PAGE_COUNT}"
            claim = normalize_content(row["claim"], where=f"FEVEROUS {row['id']} claim")
            sessions = []
            for position, page_id in enumerate(candidate_pages(index, claim)):
                record = wiki.execute("SELECT data FROM wiki WHERE id = ?", (page_id,)).fetchone()
                if record is None:
                    raise ValueError(f"FEVEROUS index references missing page {page_id!r}")
                sessions.append(
                    Session(f"{user_id}-p{position}", render_page(page_id, json.loads(record[0])))
                )
            evidence = [
                [evidence_triplet(element_id) for element_id in group["content"]]
                for group in row["evidence"]
            ]
            samples.append(
                Sample(
                    user_id=user_id,
                    dataset="feverous",
                    sessions=tuple(sessions),
                    questions=(
                        Question(
                            qid=f"feverous-{row['id']}",
                            question=claim,
                            gold={"label": row["label"], "evidence": evidence},
                            category=f"{row['label']}/{row['challenge']}",
                            evidence=tuple(
                                e for group in row["evidence"] for e in group["content"]
                            ),
                        ),
                    ),
                )
            )
    return samples
