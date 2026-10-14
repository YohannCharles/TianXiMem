"""Validate and stream a portable curated QA bundle, without original archives.

This is a loading/verification API, not a scoring runner. Preserve the per-question
source_dataset and gold for the corresponding existing evaluation adapter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(s) for s in handle if s.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(sessions: list[dict]) -> str:
    clean = [
        {"session_id": s["session_id"], "messages": [add_messages_one(m) for m in s["messages"]]}
        for s in sessions
    ]
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def add_messages_one(message: dict) -> dict:
    """Strip diagnostic source addresses; retain original role, body and timestamp."""
    return {k: message[k] for k in ("role", "content", "timestamp") if k in message}


def add_messages(session: dict) -> list[dict]:
    """Whole session, in source order. This function does not chunk or call Add."""
    return [add_messages_one(m) for m in session["messages"]]


def iter_space_records(bundle: Path):
    """Yield (whole isolated space, all its questions); no gold enters memories."""
    questions_by_space = defaultdict(list)
    for question in read_jsonl(bundle / "questions.jsonl"):
        questions_by_space[question["space_id"]].append(question)
    with gzip.open(bundle / "spaces.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            space = json.loads(line)
            yield space, questions_by_space[space["space_id"]]


def validate(bundle: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"file missing or fingerprint differs: {name}")
    questions = read_jsonl(bundle / "questions.jsonl")
    audits = read_jsonl(bundle / "question-audits.jsonl")
    if len(questions) != manifest["counts"]["questions"] or len(audits) != len(questions):
        raise ValueError("question or audit count differs")
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("duplicate question ID")
    if len({(q["source_dataset"], q["source_question_id"]) for q in questions}) != len(questions):
        raise ValueError("duplicate source question")
    tasks = Counter(q["task_type"] for q in questions)
    if tasks != manifest["counts"]["task_counts"]:
        raise ValueError("task counts differ")
    proofs = []
    for question, audit in zip(questions, audits, strict=True):
        if question["id"] != audit["id"] or question["space_id"] != question["user_id"]:
            raise ValueError("question/audit or user binding differs")
        if question["is_multi_hop"] != (question["task_type"] == "multi_hop"):
            raise ValueError("multi-hop flag differs")
        if question["is_multi_hop"] and (
            not question["answerable"] or question["is_abstention"] or len(audit["steps"]) < 2
        ):
            raise ValueError("multi-hop missing answer or chain")
        for proof in audit["evidence"] + [p for s in audit["steps"] for p in s["evidence"]]:
            proofs.append((question["space_id"], proof))
    wanted_ids = {mid for _, p in proofs for mid in p["source_message_ids"]}
    seen_ids, seen_spaces, addressed = set(), set(), {}
    scopes = manifest["space_scope_fingerprints"]
    messages, sessions, linked_questions = 0, 0, 0
    for space, bound_questions in iter_space_records(bundle):
        sid = space["space_id"]
        if sid in seen_spaces or space["user_id"] != sid or not bound_questions:
            raise ValueError("space collision, wrong isolation or orphan space")
        seen_spaces.add(sid)
        linked_questions += len(bound_questions)
        if source_fingerprint(space["sessions"]) != scopes[sid]["sha256"]:
            raise ValueError("whole original memory scope differs")
        for session in space["sessions"]:
            sessions += 1
            for message in session["messages"]:
                if (
                    set(message)
                    - {"role", "content", "timestamp", "source_message_id", "source_message_index"}
                    or not message["content"].strip()
                ):
                    raise ValueError("blank body or unexpected annotation in memory")
                mid = message["source_message_id"]
                if mid in seen_ids:
                    raise ValueError("source message ID collision")
                seen_ids.add(mid)
                if mid in wanted_ids:
                    addressed[mid] = (sid, message["content"])
                messages += 1
    if (
        seen_spaces != set(scopes)
        or len(seen_spaces) != manifest["counts"]["spaces"]
        or messages != manifest["counts"]["messages"]
        or linked_questions != len(questions)
    ):
        raise ValueError("whole memory or question binding count differs")
    for sid, proof in proofs:
        for mid in proof["source_message_ids"]:
            if mid not in addressed or addressed[mid][0] != sid:
                raise ValueError("missing or cross-user proof")
            text = addressed[mid][1]
            if proof.get("match_normalization"):
                text = text.replace("\\n", "\n")
            if proof.get("quote") and proof["quote"] not in text:
                raise ValueError("quote missing from addressed source")
    return {
        "status": "passed",
        "questions": len(questions),
        "task_counts": dict(tasks),
        "spaces": len(seen_spaces),
        "sessions": sessions,
        "messages": messages,
        "multi_hop_labeled": tasks["multi_hop"],
        "source_proofs_verified": len(proofs),
        "audit_status": manifest["audit_status"],
        "api_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.bundle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
