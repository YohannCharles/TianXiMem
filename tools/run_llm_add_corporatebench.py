"""Run an isolated Qwen fact-index experiment for CorporateBench.

This is an evaluation tool, deliberately outside ``src/tianximem``.  It reads
the already ingested CorporateBench QA pairs, asks Qwen to extract source-backed
facts, and replaces only the derived fact tables in a copied SQLite database.
The production Add/Search path remains deterministic and unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

from tianximem.facts.evidence import EVIDENCE_VERSION, key, make_fact, relation_key
from tianximem.llm.openai_chat import OpenAiCompatChat
from tianximem.store.sqlite_store import SqliteStore

PROMPT_VERSION = "corporatebench-qwen-add-v2"
ALLOWED_RELATIONS = (
    "document title",
    "employee of",
    "role commencement",
    "orientation schedule",
    "department",
    "team",
    "task",
    "meeting topic",
    "meeting frequency",
    "meeting type",
    "attendee",
    "organizer",
    "absent from meeting",
)


def prompt_for(text: str) -> str:
    relations = ", ".join(ALLOWED_RELATIONS)
    return f"""You extract explicit facts from one CorporateBench company document.
Return ONLY valid JSON with this exact shape:
{{"facts":[{{"subject":"...","relation":"...","object":"...","source_quote":"...","qualifiers":{{}}}}]}}

Allowed relation labels: {relations}.

Rules:
1. Extract every distinct fact explicitly stated in the document, including the
   email Subject, sender's employment signature, explicit role commencement or
   orientation statements, meeting topic/type/frequency, organizer, attendees,
   and explicit department/team/task mentions.
2. The subject, relation meaning, and object must be supported by the same
   sentence, header, signature, or labeled field. Never use world knowledge,
   inference, likely attendees, or a second hop.
3. source_quote MUST be copied verbatim from DOCUMENT, including punctuation and
   capitalization. If you cannot copy an exact quote, omit that fact.
4. A labeled field is explicit evidence, not inference. The "Attendees:" block of
   a calendar invite licenses one fact per listed person: subject = the person's
   name, relation = "attendee", object = the meeting's name, source_quote = the
   "Attendees:" header plus that person's line. An "Organizer: <name>" line
   licenses relation "organizer" the same way. The To: header of an ordinary
   email is NOT an attendee list — never emit attendee facts from it.
5. For Subject, use relation "document title", subject = "document", and the
   exact Subject line as source_quote. For an explicit employee signature, use
   the person's displayed name and relation "employee of".
6. A fact is not licensed by a mere mention. For example, mentioning a task is
   not a meeting topic unless the document says the meeting discusses it.
7. Do not emit dates as standalone facts. Put explicit ISO dates, frequency,
   or meeting-type information in qualifiers when useful, for example
   {{"scope_date":"2024-02-22","meeting_date":"2024-02-22"}}.
8. Meeting examples: “upcoming biweekly task meeting on optimization
implementation coordination” licenses relation “meeting topic”, object
“optimization implementation coordination”, with qualifier
meeting_frequency=biweekly. “For our upcoming work session, I'd like us to focus
on <X>” licenses “meeting topic” X for that session. “You and I held optimization implementation
coordination review sessions with directors” licenses relation “meeting type”,
object “review session”. A title alone licenses only document title.
9. If there are no supported facts, return {{"facts":[]}}. Never add commentary.

DOCUMENT:
{text}
"""


def load_env() -> tuple[list[str], list[str], str]:
    values: dict[str, str] = {}
    path = Path(".env")
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values.setdefault(name.strip(), value.strip().strip("\"'"))
    values.update({k: v for k, v in os.environ.items() if v})
    bases = [values.get("AML_BASE_URL", ""), values.get("AML_BASE_URL_2", "")]
    keys = [values.get("AML_API_KEY", ""), values.get("AML_API_KEY_2", "")]
    pairs = [(b, k) for b, k in zip(bases, keys, strict=True) if b and k]
    if not pairs:
        raise SystemExit("缺少 AML_BASE_URL/AML_API_KEY（或 _2）")
    # 探活：死端点会让一半请求白跑。
    live = []
    for base, api_key in pairs:
        chat = OpenAiCompatChat(
            base_url=base,
            api_key=api_key,
            model=values.get("AML_MODEL", "Qwen/Qwen3.5-9B"),
            timeout=30,
        )
        try:
            chat.complete("ping", max_tokens=4, temperature=0)
            live.append((base, api_key))
        except Exception:  # noqa: BLE001 - 探活失败就跳过
            pass
        finally:
            chat.close()
    if not live:
        raise SystemExit("两个对话端点都不可用")
    return [x[0] for x in live], [x[1] for x in live], values.get("AML_MODEL", "Qwen/Qwen3.5-9B")


def parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
        raise ValueError("Qwen 输出缺少 facts 数组")
    return value


def supported_fact(relation: str, subject: str, obj: str, quote: str, document: str = "") -> bool:
    """Reject semantic upgrades that happen to pass the exact-quote check."""
    rel = relation_key(relation)
    lower = quote.casefold()
    if rel in {relation_key(x) for x in ("attendee", "organizer", "absent from meeting")}:
        # 点名清单是**标注字段**：引句里有字段头和人名，会议名在全文里出现过即可。
        return (
            subject.casefold() in lower
            and re.search(r"\b(?:attendees?|organizer|absent)\s*:", lower) is not None
            and obj.casefold() in document.casefold()
        )
    if obj.casefold() not in lower:
        return False
    if rel == relation_key("document title"):
        return subject.casefold() == "document" and lower.startswith("subject:")
    if rel == relation_key("employee of"):
        if re.search(r"\b(?:works?\s+(?:for|at)|employed\s+by|employee\s+of)\b", lower):
            return True
        # 署名块：人名行 + 职务行 + 组织行。规则抽取器对**同一形状**就是这么认的
        # （`_JOB_WORDS` + `_organization`），所以这不是放宽，是同一口径。
        lines = [line.strip() for line in quote.splitlines() if line.strip()]
        return len(lines) >= 2 and any(
            re.search(
                r"\b(?:manager|specialist|engineer|director|analyst|officer|consultant|"
                r"lead|coordinator|assistant|representative)\b",
                line,
                re.I,
            )
            for line in lines
        )
    if rel == relation_key("role commencement"):
        return bool(re.search(r"\b(?:commenced|began|started)\b.*\b(?:role|working|work)\b", lower))
    if rel == relation_key("orientation schedule"):
        return "orientation schedule" in lower
    if rel == relation_key("meeting topic"):
        return bool(re.search(r"\b(?:meeting|session|agenda|discuss|discussing|cover(?:ing)?|regarding)\b", lower))
    if rel == relation_key("meeting type"):
        return bool(re.search(r"\b(?:meeting|session|huddle|all[- ]hands|review)\b", lower))
    if rel in {relation_key("department"), relation_key("team"), relation_key("task")}:
        return rel in lower or bool(re.search(r"\b(?:department|team|task|workstream|initiative)\b", lower))
    if rel in {relation_key("attendee"), relation_key("organizer"), relation_key("absent from meeting")}:
        return bool(re.search(r"\b(?:attend|attendee|organizer|organized|absent|held|with)\b", lower))
    return False


def extract_one(row: tuple[Any, ...], index: int, bases: list[str], keys: list[str], model: str) -> dict[str, Any]:
    parent_id, event_time, text = row
    last_error = ""
    for attempt in range(3):
        endpoint = index % len(bases)
        chat = OpenAiCompatChat(base_url=bases[endpoint], api_key=keys[endpoint], model=model, timeout=180)
        try:
            raw = chat.complete(prompt_for(text), max_tokens=3500, temperature=0)
            payload = parse_json(raw)
            facts = []
            for item in payload["facts"]:
                if not isinstance(item, dict):
                    continue
                subject = str(item.get("subject", "")).strip()
                relation = str(item.get("relation", "")).strip()
                obj = str(item.get("object", "")).strip()
                quote = str(item.get("source_quote", "")).strip()
                qualifiers = item.get("qualifiers") if isinstance(item.get("qualifiers"), dict) else {}
                if not subject or not relation or not obj or not quote:
                    continue
                if relation_key(relation) not in {relation_key(x) for x in ALLOWED_RELATIONS}:
                    continue
                if quote not in text:
                    continue
                if not supported_fact(relation, subject, obj, quote, text):
                    continue
                facts.append({"subject": subject, "relation": relation, "object": obj, "source_quote": quote, "qualifiers": qualifiers})
            return {"parent_memory_id": parent_id, "event_time": event_time, "facts": facts, "error": None}
        except Exception as exc:  # noqa: BLE001 - cache failure and retry
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
        finally:
            chat.close()
    return {"parent_memory_id": parent_id, "event_time": event_time, "facts": [], "error": last_error}


def deterministic_metadata(text: str) -> list[dict[str, Any]]:
    """Safe metadata guardrails used only when Qwen omits them.

    Field blocks are read deterministically: the model reliably emits the first
    attendee and then stops, while the "Attendees:" / "Organizer:" lines are
    labeled fields that can be copied verbatim without any inference.
    """
    out: list[dict[str, Any]] = []
    subject = re.search(r"^Subject:\s*(.+)$", text, re.M)
    date = re.search(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$", text, re.M)
    if subject:
        out.append({"subject": "document", "relation": "document title", "object": subject[1].strip(), "source_quote": subject[0], "qualifiers": {"scope_date": date[1] if date else ""}})
    invited = re.search(r"^You are invited to:\s*(.+)$", text, re.M)
    meeting = (invited or subject)
    if meeting is None:
        return out
    name = meeting[1].strip()
    organizer = re.search(r"^Organizer:\s*([^(\n]+?)\s*(?:\([^)]*\))?\s*$", text, re.M)
    if organizer and organizer[1].strip():
        out.append({"subject": organizer[1].strip(), "relation": "organizer", "object": name, "source_quote": organizer[0].strip(), "qualifiers": {}})
    block = re.search(r"^Attendees:\s*\n((?:[ \t]*-[^\n]*\n?)+)", text, re.M)
    if block:
        for line in re.finditer(r"^[ \t]*-[ \t]*([^(\n]+?)[ \t]*(?:\([^)]*\))?[ \t]*$", block[1], re.M):
            person = line[1].strip()
            if person:
                out.append({
                    "subject": person,
                    "relation": "attendee",
                    "object": name,
                    # 引句是**逐字的那一行**（带缩进与括号）——"Attendees:" 后面跟的未必是这一行。
                    "source_quote": line[0].rstrip("\n"),
                    "qualifiers": {},
                })
    return out


def write_facts(db_path: Path, rows: list[dict[str, Any]], *, user_id: str) -> tuple[int, int]:
    store = SqliteStore(db_path)
    total = 0
    errors = 0
    with store.transaction() as conn:
        conn.execute("DELETE FROM memory_facts WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM evidence_coverage WHERE user_id=?", (user_id,))
        for row in rows:
            if row.get("error"):
                errors += 1
            pair_id = row["parent_memory_id"]
            source = conn.execute("SELECT question FROM qa_pairs WHERE id=?", (pair_id,)).fetchone()
            text = source[0] if source else ""
            seen: set[tuple[str, str, str, str]] = set()
            facts = list(row.get("facts", []))
            for fact in deterministic_metadata(text):
                facts.append(fact)
            for index, fact in enumerate(facts):
                sig = (key(fact["subject"]), relation_key(fact["relation"]), key(fact["object"]), fact["source_quote"])
                if sig in seen or fact["source_quote"] not in text:
                    continue
                seen.add(sig)
                qualifiers = dict(fact.get("qualifiers") or {})
                qualifiers.setdefault("projection", "source")
                made = make_fact(
                    parent_memory_id=pair_id,
                    user_id=user_id,
                    subject=fact["subject"],
                    relation=fact["relation"],
                    object=fact["object"],
                    source_quote=fact["source_quote"],
                    statement=f"{fact['subject']} {fact['relation']} {fact['object']}",
                    event_time=row.get("event_time"),
                    source_date=qualifiers.pop("scope_date", None) or None,
                    position=f"llm:{index}:{relation_key(fact['relation'])}",
                    qualifiers=qualifiers,
                )
                store.insert_evidence(conn, (made,))
                total += 1
            conn.execute(
                "INSERT INTO evidence_coverage(parent_memory_id,user_id,version) VALUES(?,?,?)",
                (pair_id, user_id, EVIDENCE_VERSION),
            )
    return total, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=Path("eval/reports/runs/counting-20261010/baseline.sqlite"))
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="仅抽取前 N 份文档；0=全量")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--user-id", default="corp-kb_qa")
    args = parser.parse_args()
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.output_db.exists():
        shutil.copy2(args.source_db, args.output_db)
    with sqlite3.connect(args.output_db) as db:
        rows = db.execute("SELECT id,event_time,question FROM qa_pairs WHERE user_id=? ORDER BY local_index", (args.user_id,)).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    bases, keys, model = load_env()
    cached = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                cached[value["parent_memory_id"]] = value
    pending = [row for row in rows if row[0] not in cached]
    print(f"LLM Add extraction: total={len(rows)} cached={len(rows)-len(pending)} pending={len(pending)} workers={args.workers}", flush=True)
    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(extract_one, row, i, bases, keys, model) for i, row in enumerate(pending)]
            with args.cache.open("a", encoding="utf-8") as handle:
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    value = future.result()
                    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                    handle.flush()
                    cached[value["parent_memory_id"]] = value
                    print(f"  extracted {i}/{len(pending)} facts={len(value['facts'])} error={bool(value.get('error'))}", flush=True)
    selected = [cached[row[0]] for row in rows if row[0] in cached]
    facts, errors = write_facts(args.output_db, selected, user_id=args.user_id)
    print(json.dumps({"prompt_version": PROMPT_VERSION, "model": model, "documents": len(selected), "facts": facts, "failed_calls": errors, "db": str(args.output_db)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
