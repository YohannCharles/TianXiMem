"""共享声明页面池的原始 JSON；页面选择不接收 claim 的金标。

缺省从本地采集 Add 恢复页面标题，并按同一用户的原始 Search 选公开 claim。
页面全文和评分标注分别取自公开 Wikipedia 数据库与 dev annotations。
采集不可用时需显式标题清单，不能用 gold evidence 补页或悄悄回落到每题 top-5。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from .. import feverous as source
from ..preprocess import Message, Question, Sample, normalize_content
from ..registry import capture_dir
from ..sampling import stratified_sample
from .common import payload
from .plan import AddEvent, InputPlan, SearchEvent, digest

ANSWER_CONTRACT = "feverous-aml-json-evidence-v1"
SEARCH_INSTRUCTION = (
    "Verify the following claim using only the supplied corpus. Return one label: "
    "SUPPORTS, REFUTES, or NOT ENOUGH INFO. Also return the native evidence IDs "
    "(page title plus sentence, cell, header_cell, table_caption, or item ID) for a "
    "sufficient evidence set. For NOT ENOUGH INFO, include the available relevant "
    "evidence and identify the missing information. Do not use outside knowledge."
)


def _titles(values: object) -> list[str]:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError("AML FEVEROUS pool must contain a nonempty titles list")
    if len({unicodedata.normalize("NFC", value) for value in values}) != len(values):
        raise ValueError("AML FEVEROUS pool contains duplicate titles")
    return values


def check_pool_available(pool: Path | None) -> None:
    if pool is not None:
        if not pool.is_file():
            raise FileNotFoundError(f"AML FEVEROUS corpus pool missing: {pool}")
        _explicit_pool(pool)
        return
    root = capture_dir()
    if not all(
        (root / name).is_file()
        for name in ("official-adds.jsonl", "official-attribution.jsonl", "official-searches.jsonl")
    ):
        raise ValueError(
            "AML FEVEROUS needs the local capture Add/Search corpus or --aml-pool with an explicit "
            "JSON titles list. Gold evidence must not select the corpus."
        )


def _explicit_pool(pool: Path) -> tuple[list[str], dict]:
    raw = pool.read_bytes()
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj) - {"titles", "provenance"}:
        raise ValueError("AML FEVEROUS pool only accepts titles and optional provenance")
    return _titles(obj.get("titles")), {
        "origin": "explicit titles file",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": obj.get("provenance", "caller-declared"),
    }


def corpus_titles(pool: Path | None) -> tuple[list[str], dict]:
    check_pool_available(pool)
    if pool is not None:
        return _explicit_pool(pool)
    root = capture_dir()
    attribution = (root / "official-attribution.jsonl").read_bytes()
    users = {
        row["user_id"]
        for line in attribution.splitlines()
        if line.strip() and (row := json.loads(line)).get("dataset") == "feverous"
    }
    if len(users) != 1:
        raise ValueError("AML FEVEROUS: capture must identify one shared corpus; use --aml-pool")
    fragments = []
    adds_digest = hashlib.sha256()
    seen = set()
    with (root / "official-adds.jsonl").open("rb") as handle:
        for line in handle:
            adds_digest.update(line)
            row = json.loads(line)
            if row["user_id"] not in users:
                continue
            if row["request_id"] in seen:
                raise ValueError("AML FEVEROUS: duplicate capture Add request")
            seen.add(row["request_id"])
            for message in row["messages"]:
                text = message["content"]
                if not text.startswith("Corpus: "):
                    raise ValueError("AML FEVEROUS: unexpected capture page wrapper")
                fragments.append(text.removeprefix("Corpus: "))
    text = "".join(fragments)
    decoder = json.JSONDecoder()
    titles = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        page, position = decoder.raw_decode(text, position)
        if not isinstance(page, dict) or not isinstance(page.get("order"), list):
            raise ValueError("AML FEVEROUS: captured corpus contains a non-page JSON value")
        titles.append(page["title"])
    return _titles(titles), {
        "origin": "capture Add page titles; pages reloaded from public wiki",
        "user_id": next(iter(users)),
        "adds_sha256": adds_digest.hexdigest(),
        "attribution_sha256": hashlib.sha256(attribution).hexdigest(),
    }


def _claim_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def captured_claims(rows: list[dict], user_id: str) -> tuple[list[dict], dict[str, str], dict]:
    """以原始 Search 题面选题，不由页面内的金标覆盖率筛选 claim。"""
    path = capture_dir() / "official-searches.jsonl"
    query_digest = hashlib.sha256()
    queries: dict[str, str] = {}
    request_count = 0
    prefix = SEARCH_INSTRUCTION + "\n\nClaim: "
    with path.open("rb") as handle:
        for line in handle:
            query_digest.update(line)
            row = json.loads(line)
            if row["user_id"] != user_id:
                continue
            query = row["query"]
            if not isinstance(query, str) or not query.startswith(prefix):
                raise ValueError(
                    "AML FEVEROUS: captured Search has an unexpected claim instruction"
                )
            key = _claim_key(query[len(prefix) :])
            if not key:
                raise ValueError("AML FEVEROUS: captured Search has an empty claim")
            queries.setdefault(key, query)
            request_count += 1
    public: dict[str, list[dict]] = {}
    for row in rows:
        public.setdefault(_claim_key(row["claim"]), []).append(row)
    selected = []
    for key in queries:
        matches = public.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                "AML FEVEROUS: captured claim has missing/ambiguous public annotations"
            )
        selected.append(matches[0])
    if not selected:
        raise ValueError("AML FEVEROUS: no captured claims for the declared corpus user")
    return (
        selected,
        queries,
        {
            "origin": "same-user raw capture Search matched to public dev claims",
            "searches_sha256": query_digest.hexdigest(),
            "n_captured_requests": request_count,
            "n_unique_claims": len(queries),
        },
    )


def load(
    root: Path, *, limit: int | None, spread: bool, pool: Path | None = None
) -> list[InputPlan]:
    rows = source.read_annotations(root)
    titles, provenance = corpus_titles(pool)
    queries = {}
    query_scope = {"origin": "public dev claims; caller-declared page pool"}
    if pool is None:
        rows, queries, query_scope = captured_claims(rows, provenance["user_id"])
    if limit is not None:
        if limit < 0:
            raise ValueError("AML FEVEROUS: limit must be nonnegative")
        rows = (
            stratified_sample(rows, limit, key=lambda r: f"{r['label']}/{r['challenge']}")
            if spread and limit
            else rows[:limit]
        )
    if not rows:
        return []
    messages = []
    with source._readonly(source.corpus_path(root)) as wiki:
        for position, title in enumerate(titles):
            record = wiki.execute("SELECT data FROM wiki WHERE id = ?", (title,)).fetchone()
            if record is None:
                for form in ("NFD", "NFC"):
                    record = wiki.execute(
                        "SELECT data FROM wiki WHERE id = ?", (unicodedata.normalize(form, title),)
                    ).fetchone()
                    if record is not None:
                        break
            if record is None:
                raise ValueError(f"AML FEVEROUS: declared page missing from public wiki: {title!r}")
            page = json.loads(record[0])
            if (
                not isinstance(page.get("title"), str)
                or unicodedata.normalize("NFC", page["title"])
                != unicodedata.normalize("NFC", title)
                or not isinstance(page.get("order"), list)
            ):
                raise ValueError(f"AML FEVEROUS: invalid page structure for {title!r}")
            messages.append(
                payload(
                    Message("user", json.dumps(page, ensure_ascii=False, indent=2)),
                    "Corpus",
                    position=position,
                )
            )
    questions = tuple(
        Question(
            qid=f"feverous-{row['id']}",
            question=normalize_content(row["claim"], where="AML FEVEROUS claim"),
            gold={
                "label": row["label"],
                "evidence": [
                    [source.evidence_triplet(e) for e in group["content"]]
                    for group in row["evidence"]
                ],
            },
            category=f"{row['label']}/{row['challenge']}",
            evidence=tuple(e for group in row["evidence"] for e in group["content"]),
        )
        for row in rows
    )
    return [
        InputPlan(
            Sample("feverous-shared-corpus", "feverous", (), questions),
            (
                AddEvent("corpus", tuple(messages)),
                *(
                    SearchEvent(
                        q.qid,
                        queries.get(
                            _claim_key(q.question), SEARCH_INSTRUCTION + "\n\nClaim: " + q.question
                        ),
                    )
                    for q in questions
                ),
            ),
            {
                "scope": "one shared declared page pool; full upstream page JSON",
                "pool": provenance,
                "query_scope": query_scope,
                "page_titles": titles,
                "pool_sha256": digest(titles),
                "wiki_source_sha256": source.source_sha256(),
                "time_style": "synthetic",
                "answer_contract": ANSWER_CONTRACT,
            },
        )
    ]


def visible_json_evidence(context: str) -> set[str]:
    """只从完整可解析的 Search JSON 页面导出 ID；不读取 Add 池或评分标注。

    去掉检索块中的固定 Corpus 包装以拼接相邻片段；乱序/缺失片段不能补全。
    不完整页面不提供 ID；这个保守限制写入新 answer_contract 的说明。
    """
    text = re.sub(r"(^|\n)(?:\[\d{4}-\d{2}-\d{2}\]\s*)?(?:Q:\s*)?Corpus: ", "", context)
    decoder = json.JSONDecoder()
    result = set()
    position = 0
    while (start := text.find("{", position)) >= 0:
        try:
            page, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = end
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("title"), str)
            or not isinstance(page.get("order"), list)
        ):
            continue
        title = page["title"]
        for key in page["order"]:
            if not isinstance(key, str) or key not in page:
                continue
            value = page[key]
            if key.startswith("sentence_") and isinstance(value, str):
                result.add(f"{title}_{key}")
            elif key.startswith("table_") and isinstance(value, dict):
                if value.get("caption"):
                    result.add(f"{title}_table_caption_{key.removeprefix('table_')}")
                for row in value.get("table", []):
                    for cell in row:
                        if isinstance(cell, dict) and isinstance(cell.get("id"), str):
                            result.add(f"{title}_{cell['id']}")
            elif key.startswith("list_") and isinstance(value, dict):
                for item in value.get("list", []):
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        result.add(f"{title}_{item['id']}")
    return {element for element in result if _valid_id(element)}


def _valid_id(element: str) -> bool:
    try:
        source.evidence_triplet(element)
    except ValueError:
        return False
    return True
