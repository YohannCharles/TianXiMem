"""HybridQA dev：题目给定整张表及其全部关联段落，按表隔离用户。

limit 数题目，spread 按上游 reference 的 table/passage/未分类分层，再按 table_id
合并语料相同的题。Add 包含表格标题、上下文、所有行及 request_tok 的所有段落；
不读取 answer-node，不挑答案行/段落，也不注入题目、答案或类型标注。
本地作答只读取 Search 返回内容；EM/token-F1 复用固定版本的上游函数。
这是文本化的本地输入适配，不代表 AML 的原始表格序列化方式。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

from .preprocess import Message, Question, Sample, Session, normalize_content
from .sampling import stratified_sample

DATA_DIR = "hybridqa"
QUESTIONS = "dev.json"
REFERENCE = "dev_reference.json"
ARCHIVE = "wiki_tables_with_links.tar.gz"
CORPUS_RECEIPT = "corpus-manifest.json"
SCORER = "hybridqa/evaluate_script.py"
ANSWER_CONTRACT = "hybridqa-table-passages-v1"
SHAPE_NOTE = (
    f"HybridQA dev ({ANSWER_CONTRACT}): limit counts questions; spread by upstream "
    "table/passage/other categories. One user per table; Add contains the entire table "
    "and every linked passage, without question/gold/answer-node annotations. "
    "Text rendering and Search-only answer input are local adaptations. "
    "Answer EM and token-F1 reuse pinned upstream functions."
)


def safe_table_id(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"HybridQA: invalid table_id {value!r}")
    if any(c in value for c in ("/", "\\", "\x00")):
        raise ValueError(f"HybridQA: unsafe table_id {value!r}")
    return value


#: Windows/NTFS 不能作为文件名的字符（`:` 是盘符/数据流分隔符，其余同保留集）。
_UNSAFE_FILENAME_CHARS = '<>:"/\\|?*'


def corpus_filename(table_id: str) -> str:
    """`table_id` → 落盘文件名（`tables_tok/<这段>.json` 里的那段）。

    上游表名是维基标题，含 `:`（如 `ISO_3166-2:FR_2`）。`:` 在 NTFS 上是保留字符：
    直接落盘会写成隐藏数据流，且原子发布（`os.replace`）会失败。这里把**文件名里
    非法/有歧义的字符**（`<>:"/\\|?*`、控制字符，以及 `%` 自身）编码成 `%XX`
    （UTF-8 字节，大写十六进制）；其余字符（含中文）原样保留，于是同一份语料在
    Windows 与 Linux 上同名、可互换、可拷贝。

    ⚠ `table_id` 本身不变——分组键与内容哈希仍用原值；编码只作用于磁盘路径。
    `%` 也编码，保证映射可逆、无歧义（不可能与原始名撞车）。
    同 [`memory500.portable_hybrid`](./memory500.py) 一套动机（那边直接哈希 tid）。
    """
    return "".join(
        "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
        if char == "%" or char in _UNSAFE_FILENAME_CHARS or ord(char) < 0x20
        else char
        for char in table_id
    )


def corpus_relpath(name: str) -> str:
    """归档成员相对名 `folder/<table_id>.json` → 落盘相对名（只编码文件名那段）。

    扩展名 `.json` 全由安全字符组成，编码后原样保留。
    """
    folder, sep, filename = name.partition("/")
    encoded = corpus_filename(filename)
    return f"{folder}/{encoded}" if sep else encoded


def required_corpus_files(root: Path) -> set[str]:
    rows = json.loads((root / DATA_DIR / QUESTIONS).read_text(encoding="utf-8"))
    return {
        f"{folder}/{safe_table_id(row['table_id'])}.json"
        for row in rows
        for folder in ("tables_tok", "request_tok")
    }


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("HybridQA: table text must be a string")
    return value.strip()


def _cell(cell: list) -> str:
    text, links = cell
    if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
        raise ValueError("HybridQA: invalid table links")
    return _text(text) + (f" (links: {', '.join(links)})" if links else "")


def read_corpus(root: Path, table_id: str, receipt: dict) -> tuple[dict, dict]:
    """共享经 receipt 校验的整表和全部段落，保留未文本化的上游结构。"""
    docs = []
    for folder in ("tables_tok", "request_tok"):
        name = f"{folder}/{table_id}.json"
        # New archives encode NTFS-reserved characters.  Linux archives made
        # before that rule may still contain the literal legacy filename.
        encoded = root / DATA_DIR / corpus_relpath(name)
        legacy = root / DATA_DIR / name
        path = encoded if encoded.is_file() else legacy
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["files"].get(name):
            raise ValueError(f"HybridQA: corpus sha256 mismatch: {name}")
        docs.append(json.loads(raw))
    table, passages = docs
    return table, passages


def _corpus(root: Path, table_id: str, receipt: dict) -> tuple[Message, ...]:
    table, passages = read_corpus(root, table_id, receipt)
    headers = [_cell(cell) for cell in table["header"]]
    context = [f"Table: {table_id}", f"Title: {_text(table['title'])}"]
    for key in ("intro", "section_title", "section_text"):
        if table.get(key):
            context.append(f"{key}: {_text(table[key])}")
    context.append("Columns: " + " | ".join(headers))
    texts = ["\n".join(context)]
    for index, row in enumerate(table["data"]):
        if len(row) != len(headers):
            raise ValueError(f"HybridQA {table_id}: row/header lengths differ")
        texts.append(
            f"Table: {table_id}\nRow {index + 1}: "
            + " | ".join(
                f"{header}: {_cell(cell)}" for header, cell in zip(headers, row, strict=True)
            )
        )
    if not isinstance(passages, dict):
        raise ValueError(f"HybridQA {table_id}: passages must be a mapping")
    for link, body in passages.items():
        title = unquote(link.removeprefix("/wiki/")).replace("_", " ")
        # 空页面仍保留链接与标题；不根据是否含答案选择段落。
        texts.append(f"Page: {title}\nLink: {link}\n{_text(body)}")
    return tuple(
        Message(role="user", content=normalize_content(text, where=f"HybridQA {table_id}"))
        for text in texts
    )


def load_hybridqa(
    bench_dir: str | Path,
    *,
    limit: int | None = None,
    spread: bool = False,
    question_ids: set[str] | None = None,
) -> list[Sample]:
    root = Path(bench_dir)
    rows = json.loads((root / DATA_DIR / QUESTIONS).read_text(encoding="utf-8"))
    reference = json.loads((root / DATA_DIR / REFERENCE).read_text(encoding="utf-8"))
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("HybridQA: duplicate question_id")
    if set(ids) != set(reference["reference"]):
        raise ValueError("HybridQA: dev/reference question IDs differ")
    categories = {qid: kind for kind in ("table", "passage") for qid in reference[kind]}
    if question_ids is not None:
        rows = [r for r in rows if f"hybridqa-{r['question_id']}" in question_ids]
    if limit is not None:
        if limit < 0:
            raise ValueError("HybridQA: limit must be nonnegative")
        if limit == 0:
            return []
        rows = (
            stratified_sample(
                rows, limit, key=lambda row: categories.get(row["question_id"], "other")
            )
            if spread
            else rows[:limit]
        )
    receipt = json.loads((root / DATA_DIR / CORPUS_RECEIPT).read_text(encoding="utf-8"))
    from .manifest import MANIFEST

    expected = next(e["sha256"] for e in MANIFEST if e["name"] == f"{DATA_DIR}/{ARCHIVE}")
    if receipt.get("archive_sha256") != expected:
        raise ValueError("HybridQA: corpus receipt belongs to another archive version")
    grouped: dict[str, list[Question]] = {}
    for row in rows:
        qid = row["question_id"]
        answer = reference["reference"][qid]
        if not isinstance(answer, str) or not answer.strip() or row["answer-text"] != answer:
            raise ValueError(f"HybridQA {qid}: dev/reference answers differ or are empty")
        table_id = safe_table_id(row["table_id"])
        grouped.setdefault(table_id, []).append(
            Question(
                qid=f"hybridqa-{qid}",
                question=normalize_content(row["question"], where=f"HybridQA {qid} question"),
                gold={"answer": answer},
                category=categories.get(qid, "other"),
            )
        )
    samples = []
    for table_id, questions in grouped.items():
        user_id = "hybridqa-" + hashlib.sha256(table_id.encode()).hexdigest()[:20]
        samples.append(
            Sample(
                user_id=user_id,
                dataset="hybridqa",
                sessions=(Session(f"{user_id}-corpus", _corpus(root, table_id, receipt)),),
                questions=tuple(questions),
            )
        )
    return samples
