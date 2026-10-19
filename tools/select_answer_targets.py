#!/usr/bin/env python3
"""Select a reproducible, diverse set of v1.1.1 answer failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _kind(question: str, gold: object) -> str:
    q = question.lower()
    answer_type = gold.get("answer_type") if isinstance(gold, dict) else None
    if answer_type == "bool" or q.startswith(("did ", "were ", "was ", "is ")):
        return "boolean"
    if answer_type in {"int", "float"} or q.startswith(("how many", "what is the total number", "how often")):
        return "count"
    if answer_type == "List[str]" or q.startswith(("which ", "who ", "what are", "what were")):
        return "list_or_entity"
    return "scalar"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text())
    failed = {
        row["id"]: row
        for row in catalog
        if row.get("run") == "native24h-20261010-v1.1.1-corporatebench"
        and row.get("correct") is False
        and row.get("gold") is not None
    }
    source_rows = {
        row["id"]: row
        for row in (json.loads(line) for line in args.source.read_text().split("\n") if line.strip())
    }
    groups: dict[str, list[dict]] = {}
    for row in failed.values():
        source = source_rows.get(row["id"])
        if source is None:
            continue
        kind = _kind(source["question"], source.get("gold_answer"))
        groups.setdefault(kind, []).append(source)
    # Round-robin categories keeps the target set balanced while preserving source order.
    picked: list[dict] = []
    while len(picked) < args.limit and any(groups.values()):
        for kind in sorted(groups):
            if groups[kind] and len(picked) < args.limit:
                picked.append(groups[kind].pop(0))
    selected = [row["id"] for row in picked]
    manifest = {
        "top_k": 100,
        "selection": "v1.1.1 incorrect CorporateBench questions; round-robin answer types",
        "groups": [
            {
                "dataset": "corporatebench",
                "user_id": "corp-kb_qa",
                "source": str(args.source),
                "qids": selected,
                "group": "v1.1.1-failed-corporate-50",
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": len(selected), "kinds": {k: sum(_kind(source_rows[q]["question"], source_rows[q].get("gold_answer")) == k for q in selected) for k in sorted({ _kind(source_rows[q]["question"], source_rows[q].get("gold_answer")) for q in selected})}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
