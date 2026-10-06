"""离线核对采集套件、提问时序与当前共同取证查询计划；不调用服务或模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from tianximem.facts.query import compile_query  # noqa: E402


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def audit(capture: Path) -> dict:
    last_add: dict[str, int] = {}
    search_sequences: set[int] = set()
    with (capture / "official-timeline.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["kind"] == "add":
                last_add[row["user_id"]] = max(
                    last_add.get(row["user_id"], -1), int(row["seq"])
                )
            else:
                search_sequences.add(int(row["seq"]))

    families: dict[str, Counter] = defaultdict(Counter)
    users: dict[str, set[str]] = defaultdict(set)
    plans: dict[str, Counter] = defaultdict(Counter)
    dimension_keys: set[str] = set()
    seen: set[int] = set()
    with (capture / "official-eval-kit.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            seq = int(row["seq"])
            if seq in seen or seq not in search_sequences:
                raise ValueError(f"套件题号重复或不在采集 Search 内：{seq}")
            seen.add(seq)
            dimension_keys.update(
                key for key in row if "dimension" in key.lower() or "ability" in key.lower()
            )
            family = row["dataset"]
            families[family]["queries"] += 1
            if row["judge_kind"] == "none":
                continue
            families[family]["scorable"] += 1
            families[family]["before_user_last_add"] += seq < last_add.get(row["user_id"], -1)
            users[family].add(row["user_id"])
            plan = compile_query(row["query"])
            plans[family][
                f"{plan.operator}:{plan.projection}" if plan is not None else "no_plan"
            ] += 1

    source_paths = (
        "src/tianximem/facts/query.py",
        "src/tianximem/facts/input.py",
        "src/tianximem/facts/grammar.py",
        "src/tianximem/facts/evidence.py",
        "src/tianximem/facts/conversation.py",
    )
    return {
        "scope": (
            "Pure query-compiler applicability on captured scorable queries; "
            "not extraction coverage, executed evidence selection, retrieval or QA score. "
            "No query text or gold answers are exported."
        ),
        "capture_files": {
            name: fingerprint(capture / name)
            for name in ("official-eval-kit.jsonl", "official-timeline.jsonl")
        },
        "compiler_files": {name: fingerprint(ROOT / name) for name in source_paths},
        "queries": len(seen),
        "scorable_queries": sum(value["scorable"] for value in families.values()),
        "dimension_field_names": sorted(dimension_keys),
        "families": {
            family: {
                **dict(value),
                "scorable_users": len(users[family]),
                "query_plans": dict(plans[family]),
            }
            for family, value in sorted(families.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "official-dataset-2026-09-29")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("audit.json"))
    args = parser.parse_args()
    result = audit(args.capture_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{result['queries']} captured queries / {result['scorable_queries']} scorable; "
        f"dimension fields: {result['dimension_field_names']}; saved to {args.output}"
    )


if __name__ == "__main__":
    main()
