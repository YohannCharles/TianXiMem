"""Local evidence gate for offline LLM fact candidates.

The gate is intentionally conservative and evaluation-only.  It rejects a
candidate when its endpoints do not occur in the same local record or when no
relation trigger is close enough to license the proposed object.  It is not
wired into Add/Search.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tianximem.facts.evidence import key, relation_key

try:  # Direct script execution.
    from fact_entity_smoke import _score_triples, load_cases
except ModuleNotFoundError:  # Package import from pytest.
    from tools.fact_entity_smoke import _score_triples, load_cases


_TRIGGERS: dict[str, tuple[str, ...]] = {
    "employee": ("works for", "works at", "employed by", "employee of", "job at", "lost my job at"),
    "occupation": ("job as", "works as", "occupation", "profession"),
    "author": ("author", "written by"),
    "dedicated": ("dedicated to",),
    "place of birth": ("born in", "born at"),
    "place of death": ("died in", "died at"),
    "country of citizenship": ("citizen of", "citizenship"),
    "head of state": ("head of state", "current head of state"),
    "head of government": ("head of government", "current head of government"),
    "officeholder": ("officeholder", "secretary-general is", "is the secretary-general"),
    "capital": ("capital of",),
    "continent": ("continent", "located in the continent"),
    "creator": ("created by", "creator"),
    "developer": ("developed by", "developer"),
    "manufacturer": ("produced by", "manufacturer", "company that produced"),
    "performer": ("performed by", "performer"),
    "educated": ("educated at", "attended", "studied at"),
    "attended": ("attended",),
    "played": ("played for",),
    "team": ("plays for", "played for"),
    "participated": ("ran", "took part", "participated", "participate"),
    "started": ("started", "starting", "launched"),
    "passionate about": ("passionate about", "my passion", "been into"),
    "plans to open": ("want to open", "wanna open", "plan to open", "searching for a place to open", "looking for a place to open"),
    "plans to expand": ("plan to expand", "want to expand", "expand my", "expand the"),
    "position": ("holds the position of",),
    "political party": ("member of",),
    "spouse": ("married to", "spouse"),
    "organized": ("organized", "organiser", "organizer"),
    "attendee": ("attendees", "attendee"),
    "live": ("location", "lives in", "live in"),
    "profession": ("profession",),
}


def _scopes(source: str) -> tuple[str, ...]:
    text = source.strip()
    if not text:
        return ()
    lines = text.splitlines()
    corpus = [part.strip() for part in re.split(r"(?=Corpus:)", text) if part.strip().startswith("Corpus:")]
    if len(corpus) >= 2:
        return tuple(corpus)
    if "USER PROFILE:" in text:
        head = text.split("PERSONALITY OVERVIEW:", 1)[0].strip()
        return (head,) if head else (text,)
    speaker_indices = [
        i for i, line in enumerate(lines)
        if re.match(r"^\s*(?:user|assistant|[A-Za-z][A-Za-z0-9 _.'’-]{0,40})\s*:\s*\S", line)
    ]
    if len(speaker_indices) >= 2:
        scopes = []
        for pos, start in enumerate(speaker_indices):
            end = speaker_indices[pos + 1] if pos + 1 < len(speaker_indices) else len(lines)
            value = "\n".join(lines[start:end]).strip()
            if value:
                scopes.append(value)
        return tuple(scopes)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) > 1:
        return tuple(paragraphs)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    return tuple(sentences) if sentences else (text,)


def _profile_scope(source: str, subject: str, object_: str) -> bool:
    if "USER PROFILE:" not in source:
        return False
    folded = source.casefold()
    # A profession field is not employment.  This blocks the most common
    # profile hallucination while retaining the schema-specific profession fact.
    return subject.casefold() in folded and object_.casefold() in folded


def _same_scope(source: str, subject: str, object_: str) -> str | None:
    sf = subject.casefold()
    of = object_.casefold()
    for scope in _scopes(source):
        folded = scope.casefold()
        if sf in folded and of in folded:
            return scope
    return None


def _trigger_supported(scope: str, relation: str, object_: str) -> bool:
    folded = scope.casefold()
    relation = relation_key(relation)
    triggers = _TRIGGERS.get(relation)
    if not triggers:
        # Corpus rows are authoritative and their relation label is already
        # explicitly present in the row.
        return scope.lstrip().startswith("Corpus:") and relation in folded
    object_folded = object_.casefold()
    for trigger in triggers:
        start = 0
        while True:
            index = folded.find(trigger, start)
            if index < 0:
                break
            tail = folded[index + len(trigger) : index + len(trigger) + 180]
            if object_folded in tail:
                return True
            start = index + 1
    return False


def _keep(source: str, subject: str, relation: str, object_: str) -> tuple[bool, str]:
    if subject.casefold() not in source.casefold() or object_.casefold() not in source.casefold():
        return False, "endpoint-not-in-source"
    scope = _same_scope(source, subject, object_)
    if scope is None:
        return False, "cross-record-endpoints"
    rel = relation_key(relation)
    if "USER PROFILE:" in source and rel == "employee":
        return False, "profile-profession-is-not-employment"
    if not _trigger_supported(scope, rel, object_):
        return False, "missing-local-trigger"
    return True, "local-evidence"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.gold)
    candidate_rows = {
        row["id"]: row
        for row in (json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines())
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    rows: list[dict[str, Any]] = []
    predicted = []
    dropped = 0
    for case in cases:
        input_row = candidate_rows.get(case["id"], {})
        accepted = []
        decisions = []
        for item in input_row.get("relations", []):
            if not isinstance(item, dict):
                continue
            values = tuple(item.get(name) for name in ("subject", "relation", "object"))
            if not all(isinstance(value, str) and value.strip() for value in values):
                continue
            keep, reason = _keep(case["text"], *values)
            triple = dict(zip(("subject", "relation", "object"), values, strict=True))
            decisions.append({"input": triple, "keep": keep, "reason": reason})
            if keep:
                accepted.append(triple)
            else:
                dropped += 1
        rows.append({"id": case["id"], "text": case["text"], "relations": accepted, "gate_decisions": decisions})
        predicted.append({(key(x["subject"]), relation_key(x["relation"]), key(x["object"])) for x in accepted})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"cases": len(cases), "dropped": dropped, "score": _score_triples(cases, predicted), "out": str(args.out)}
    print(json.dumps(report, ensure_ascii=False, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
