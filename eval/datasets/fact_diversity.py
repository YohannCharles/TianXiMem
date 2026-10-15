"""Build a deterministic, annotation-ready fact extraction sample.

The source datasets are read-only.  This module deliberately keeps answers,
gold labels, and evidence annotations out of ``text``.  It is an evaluation
fixture builder, not a production data loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
DEFAULT_OUTPUT = ROOT / "eval/facts/common-facts-diverse-sample.jsonl"
DEFAULT_REPORT = ROOT / "eval/reports/common-facts-gold/common-facts-diverse-sample.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _shorten(text: str, limit: int = 6000) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.62))
    tail = limit - head
    return f"{text[:head]}\n[...context shortened for the diversity sample...]\n{text[-tail:]}"


def _lines(messages: list[dict[str, Any]], *, limit: int = 6000) -> str:
    rendered = []
    for message in messages:
        role = message.get("speaker") or message.get("role") or "message"
        content = message.get("text") or message.get("content") or ""
        if content:
            rendered.append(f"{role}: {content}")
    return _shorten("\n".join(rendered), limit)


def _window(messages: list[dict[str, Any]], index: int, radius: int = 1) -> list[dict[str, Any]]:
    return messages[max(0, index - radius) : min(len(messages), index + radius + 1)]


def _tags(text: str, *extra: str) -> list[str]:
    tags = set(extra)
    if len(text) >= 900:
        tags.add("long_text")
    if re.search(
        r"\b(?:yesterday|last|before|after|from|until|in 19|in 20|\d{4}[-/]\d{1,2})", text, re.I
    ):
        tags.add("temporal")
    if re.search(r"(?:not|never|without|no evidence|insufficient|不|未|没有|并非)", text, re.I):
        tags.add("negation_or_uncertainty")
    if re.search(r"[A-Z][a-z]+(?:[-'’][A-Za-z]+)?(?:\s+[A-Z][a-z]+){1,4}", text):
        tags.add("entity_boundary")
    if re.search(r"[À-ÖØ-öø-ÿĀ-ž]|[\u4e00-\u9fff]", text):
        tags.add("non_ascii_entity")
    return sorted(tags)


def _row(
    dataset: str, source_ref: dict[str, Any], text: str, *tags: str, origin: str = "raw"
) -> dict[str, Any]:
    text = _shorten(text)
    return {
        "dataset": dataset,
        "text_origin": origin,
        "source_ref": source_ref,
        "text": text,
        "coverage": _tags(text, *tags),
        "annotation_status": "pending",
        "entities": None,
        "relations": None,
    }


def _locomo(path: Path, dataset: str, *, old: bool = False) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for conv_index, item in enumerate(rows[:2] if old else rows[:1]):
        conversation = item["conversation"]
        chosen = [
            (1, 2, "natural_language"),
            (1, 4, "entity_boundary"),
            (2, 0, "temporal"),
            (3, 0, "long_text"),
        ]
        for session_index, message_index, focus in chosen:
            messages = conversation.get(f"session_{session_index}", [])
            if not messages or message_index >= len(messages):
                continue
            selected = _window(messages, message_index, 1 if focus != "long_text" else 2)
            text = _lines(selected)
            out.append(
                _row(
                    dataset,
                    {
                        "sample_id": item.get("sample_id", conv_index),
                        "session": session_index,
                        "message": messages[message_index].get("dia_id"),
                    },
                    text,
                    focus,
                    "dialogue",
                    origin="raw_conversation",
                )
            )
    return out


def _longmemeval(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    wanted = {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    }
    out = []
    for item in rows:
        kind = item.get("question_type")
        if kind not in wanted or any(r["source_ref"].get("question_type") == kind for r in out):
            continue
        sessions = item.get("haystack_sessions", [])
        selected = []
        answer_ids = set(item.get("answer_session_ids", []))
        ids = item.get("haystack_session_ids", [])
        for i, session in enumerate(sessions):
            if i < len(ids) and ids[i] in answer_ids:
                selected.append(session)
        if not selected and sessions:
            selected = sessions[:2]
        flat = [turn for session in selected[:2] for turn in session[:8]]
        text = _lines(flat, limit=6500)
        out.append(
            _row(
                "longmemeval-s",
                {"question_id": item["question_id"], "question_type": kind},
                text,
                "dialogue",
                "multi_session" if kind == "multi-session" else kind,
                origin="raw_context_window",
            )
        )
    return out


def _musique(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    out = []
    seen = set()
    for item in rows:
        hops = len(item.get("question_decomposition", []))
        variant = "answerable" if item.get("answerable") else "unanswerable"
        key = (hops, variant)
        if key in seen or hops < 2:
            continue
        seen.add(key)
        text = "\n\n".join(
            f"Title: {p['title']}\n{p['paragraph_text']}" for p in item.get("paragraphs", [])
        )
        out.append(
            _row(
                "musique",
                {"id": item["id"], "hops": hops, "variant": variant},
                text,
                "multi_hop",
                "paragraphs",
                "unseen_sentence",
                origin="raw_paragraphs",
            )
        )
        if len(out) >= 4:
            break
    return out


def _hybridqa(path: Path, corpus_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in rows:
        table_id = item.get("table_id")
        source = corpus_dir / f"{table_id}.json"
        if not source.exists():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        text = "\n\n".join(f"{key}: {value}" for key, value in payload.items())
        out.append(
            _row(
                "hybridqa",
                {"question_id": item["question_id"], "table_id": table_id},
                text,
                "table_like",
                "entity_boundary",
                "multi_hop",
                origin="raw_linked_table_text",
            )
        )
        if len(out) >= 3:
            break
    return out


def _feverous(path: Path) -> list[dict[str, Any]]:
    out = []
    for item in _read_jsonl(path)[1:]:
        claim = item.get("claim", "").strip()
        if not claim:
            continue
        out.append(
            _row(
                "feverous",
                {"id": item.get("id"), "challenge": item.get("challenge")},
                claim,
                "claim",
                "negation_or_uncertainty",
                origin="raw_claim",
            )
        )
        if len(out) >= 4:
            break
    return out


def _halumem(path: Path) -> list[dict[str, Any]]:
    out = []
    for item in _read_jsonl(path)[:3]:
        sessions = item.get("sessions", [])
        if not sessions:
            continue
        dialogue = sessions[min(1, len(sessions) - 1)].get("dialogue", [])
        text = _lines(dialogue[:10], limit=5500)
        out.append(
            _row(
                "halumem",
                {"uuid": item["uuid"], "session": min(1, len(sessions) - 1) + 1},
                text,
                "dialogue",
                "persona",
                "structured_profile",
                origin="raw_dialogue",
            )
        )
    return out


def _medmemory(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path).to_pylist()
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        groups[(row["persona_id"], row["session_id"])].append(row)
    out = []
    for (persona, session), turns in sorted(groups.items()):
        if len(turns) < 3:
            continue
        text = _lines(turns[:8], limit=5000)
        out.append(
            _row(
                "medmemorybench",
                {"persona_id": persona, "session_id": session},
                text,
                "chinese",
                "dialogue",
                "long_text",
                origin="raw_dialogue",
            )
        )
        if len(out) >= 4:
            break
    return out


def _memtrap(root: Path) -> list[dict[str, Any]]:
    out = []
    source_dir = root / "memtrapbench/cognitive_bias"
    for path in sorted(source_dir.glob("*.json")):
        pattern = r"(?:Inertia|Hallucination|Hurt|Number_game|Poison|Unclear)_\d+\.json"
        if not re.fullmatch(pattern, path.name, re.I):
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not entries:
            continue
        item = entries[0]
        text = (
            _lines(item.get("context_history", [])[-6:], limit=5500)
            + f"\nuser: {item.get('final_trigger', '')}"
        )
        out.append(
            _row(
                "memtrapbench",
                {
                    "file": str(path.relative_to(DATASET)),
                    "id": item.get("id"),
                    "domain": item.get("domain"),
                },
                text,
                "long_text",
                "distractor_history",
                "unseen_sentence",
                origin="raw_context_window",
            )
        )
        if len(out) >= 4:
            break
    return out


def _mquake(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(
        path, columns=["case_id", "orig_triples_labeled", "new_triples_labeled"]
    ).to_pylist()
    out = []
    for item in table[:4]:
        old = item.get("orig_triples_labeled") or []
        new = item.get("new_triples_labeled") or []
        lines = [f"Corpus: {s} — {r} — {o}" for s, r, o in old]
        for (s, r, o), old_trip in zip(new, old, strict=False):
            if list(old_trip) != [s, r, o]:
                lines.append(f"Corpus: {s} — {r} — {o} (this replaces the earlier value)")
        out.append(
            _row(
                "mquake-remastered",
                {"file": path.name, "case_id": item["case_id"]},
                "\n".join(lines),
                "multi_hop",
                "update",
                "structured_triples",
                origin="rendered_from_raw_triples",
            )
        )
    return out


def _tempreason(root: Path) -> list[dict[str, Any]]:
    out = []
    for level in ("test_l2.json", "test_l3.json"):
        path = root / level
        with path.open(encoding="utf-8") as handle:
            item = json.loads(next(line for line in handle if line.strip()))
        context = item.get("fact_context", "")
        if item.get("context"):
            context += "\n\n" + _shorten(item["context"], 2600)
        out.append(
            _row(
                "tempreason",
                {"file": level, "id": item.get("id")},
                context,
                "temporal",
                "multi_hop" if "l3" in level else "interval_reasoning",
                origin="raw_context",
            )
        )
    return out


def _corporatebench(path: Path) -> list[dict[str, Any]]:
    out = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            n for n in archive.namelist() if n.startswith("documents/") and n.endswith(".txt")
        )
        groups = ("calendar_invite", "meeting_minute", "email_")
        for group in groups:
            name = next(
                (n for n in names if f"/{group}" in n or n.split("/")[-1].startswith(group)), None
            )
            if not name:
                continue
            text = archive.read(name).decode("utf-8", "replace")
            out.append(
                _row(
                    "corporatebench",
                    {"archive_member": name},
                    text,
                    "natural_language",
                    "structured_profile",
                    "entity_boundary",
                    origin="raw_email_or_calendar",
                )
            )
    return out


def _beam(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["conversation_id", "user_profile", "chat"]).to_pylist()
    out = []
    for item in table[:2]:
        chat = item.get("chat") or []
        if chat and isinstance(chat[0], list):
            chat = [turn for session in chat for turn in session]
        text = (
            str(item.get("user_profile", {}).get("user_info", ""))
            + "\n"
            + _lines(chat[:6], limit=5200)
        )
        out.append(
            _row(
                "beam",
                {"conversation_id": item["conversation_id"]},
                text,
                "long_text",
                "natural_language",
                "structured_profile",
                origin="raw_profile_and_chat",
            )
        )
    return out


def _clbench(path: Path) -> list[dict[str, Any]]:
    out = []
    for index, item in enumerate(_read_jsonl(path)[:2]):
        messages = item.get("messages", [])
        text = _lines(messages[:2], limit=6000)
        out.append(
            _row(
                "clbench",
                {"row": index},
                text,
                "long_text",
                "rules_and_process",
                "negation_or_uncertainty",
                origin="raw_messages",
            )
        )
    return out


def _personamem(path: Path, benchmark: Path) -> list[dict[str, Any]]:
    out = []
    with benchmark.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows[:2]:
        chat_path = path / row["chat_history_32k_link"]
        if not chat_path.exists():
            continue
        raw = json.loads(chat_path.read_text(encoding="utf-8"))
        text = _lines(raw.get("chat_history", [])[:8], limit=6000)
        out.append(
            _row(
                "personamem-v2",
                {"persona_id": row["persona_id"], "chat_file": str(chat_path.relative_to(ROOT))},
                text,
                "long_text",
                "persona",
                "natural_language",
                origin="raw_chat_history",
            )
        )
    return out


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    builders = [
        lambda: _locomo(DATASET / "locomo-refined/locomo_refined.json", "locomo-refined"),
        lambda: _locomo(DATASET / "locomo/locomo10.json", "locomo", old=True),
        lambda: _longmemeval(DATASET / "longmemeval-s/lme_s_cleaned.json"),
        lambda: _musique(DATASET / "musique/musique_full_v1.0_dev.jsonl"),
        lambda: _hybridqa(DATASET / "hybridqa/dev.json", DATASET / "hybridqa/request_tok"),
        lambda: _feverous(DATASET / "feverous/feverous_dev_challenges.jsonl"),
        lambda: _halumem(DATASET / "halumem/HaluMem-Medium.jsonl"),
        lambda: _medmemory(DATASET / "medmemorybench/data/zh/dialogues.parquet"),
        lambda: _memtrap(DATASET / "memtrapbench"),
        lambda: _mquake(DATASET / "mquake-remastered/data/CF3k-00000-of-00001.parquet"),
        lambda: _tempreason(DATASET / "tempreason"),
        lambda: _corporatebench(DATASET / "corporatebench/data/kb/zenith.kb"),
        lambda: _beam(DATASET / "beam/data/100K-00000-of-00001.parquet"),
        lambda: _clbench(DATASET / "clbench/clbench.jsonl"),
        lambda: _personamem(
            DATASET / "personamem-v2", DATASET / "personamem-v2/benchmark/text/benchmark.csv"
        ),
    ]
    rows: list[dict[str, Any]] = []
    for builder in builders:
        rows.extend(builder())
    unique = []
    seen = set()
    for row in rows:
        key = (row["dataset"], hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
        if key not in seen and row["text"]:
            seen.add(key)
            unique.append(row)
    for index, row in enumerate(unique, 1):
        row["id"] = f"diverse-{index:04d}"
    counts = Counter(row["dataset"] for row in unique)
    tags = Counter(tag for row in unique for tag in row["coverage"])
    report = {
        "version": "common-facts-diverse-sample-v1",
        "count": len(unique),
        "datasets": dict(sorted(counts.items())),
        "coverage": dict(sorted(tags.items())),
        "annotation_contract": (
            "text is source-only; entities/relations are pending nulls; "
            "do not score until annotated"
        ),
        "rendered_sources": ["mquake-remastered"],
        "excluded": {
            "scriptmem": "public story/source text is unavailable",
            "doc-pp": "source is PDF/image-oriented and not a clean text fact-extraction input",
            "personamem-v1": (
                "CSV has questions/options but no public chat history in this checkout"
            ),
            "tempreason-l1": "no memory context; arithmetic question only",
        },
    }
    return unique, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    rows, report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
