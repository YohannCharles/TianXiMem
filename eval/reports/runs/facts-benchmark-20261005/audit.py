"""Progress and provenance audit for the frozen seven-dataset HTTP replay."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NAME = "facts-benchmark-20261005"
OUT = ROOT / "eval/reports/runs" / NAME
VAR = ROOT / "var" / NAME


def rows(path: Path):
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def progress() -> dict:
    status = json.loads((OUT / "status.json").read_text())
    runs = []
    for entry in status["runs"]:
        directory = OUT.parent / entry["run_id"]
        counters = {}
        for stage in ("input", "answers", "labels"):
            counters[stage] = sum(
                sum(1 for line in path.open() if line.strip())
                for path in directory.glob(f"*/{stage}.jsonl")
            )
        runs.append(
            {
                "dataset": entry["dataset"],
                "facts": entry["facts"],
                "state": entry["state"],
                **counters,
                "overall": entry.get("scores", {}).get("overall"),
            }
        )
    return {"at": datetime.now(UTC).isoformat(), "state": status["state"], "runs": runs}


def audit() -> dict:
    manifest = json.loads((OUT / "manifest.json").read_text())
    state = json.loads((OUT / "status.json").read_text())
    expected = {d["dataset"]: set(d["qids"]) for d in manifest["datasets"]}
    db = VAR / "tianxi.db"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        parents = {r["id"]: dict(r) for r in conn.execute("select * from qa_pairs")}
        facts = {r["id"]: dict(r) for r in conn.execute("select * from memory_facts")}
        coverage = list(
            conn.execute("select parent_memory_id,user_id,version from evidence_coverage")
        )
    for fact in facts.values():
        parent = parents[fact["parent_memory_id"]]
        assert parent["user_id"] == fact["user_id"], fact["id"]
        assert fact["source_quote"] in (parent[fact["source_side"]] or ""), fact["id"]
        assert fact["version"] == "memory-facts-v3", fact["version"]
    summary = []
    for entry in state["runs"]:
        directory = OUT.parent / entry["run_id"]
        inputs = [r for path in directory.glob("*/input.jsonl") for r in rows(path)]
        answers = [r for path in directory.glob("*/answers.jsonl") for r in rows(path)]
        labels = [r for path in directory.glob("*/labels.jsonl") for r in rows(path)]
        ids = {}
        for stage, records in (("input", inputs), ("answers", answers), ("labels", labels)):
            seen = [r["id"] for r in records]
            assert len(seen) == len(set(seen)), (entry["run_id"], stage, "duplicate ids")
            ids[stage] = set(seen)
            assert ids[stage] <= expected[entry["dataset"]], (entry["run_id"], stage)
            if entry["state"] == "complete":
                assert ids[stage] == expected[entry["dataset"]], (entry["run_id"], stage)
        fact_queries = empty_queries = fact_occurrences = searches = 0
        for result in rows(OUT / f"searches-{entry['facts']}-{entry['dataset']}.jsonl"):
            items = result["response"]["data"]
            assert len(items) <= result["top_k"], "top_k contract"
            searches += 1
            used_fact = False
            for item in items:
                if item["id"] in facts:
                    assert facts[item["id"]]["user_id"] == result["user_id"]
                    assert item["content"] == facts[item["id"]]["content"]
                    fact_occurrences += 1
                    used_fact = True
                else:
                    assert item["id"] in parents, (entry["run_id"], item["id"])
                    assert parents[item["id"]]["user_id"] == result["user_id"]
            fact_queries += used_fact
            empty_queries += not items
        failures = sum("ERROR" in str(r.get("label", "")) for r in labels)
        summary.append(
            {
                "dataset": entry["dataset"],
                "facts_enabled": entry["facts"] == "on",
                "state": entry["state"],
                "n_expected": len(expected[entry["dataset"]]),
                "n_inputs": len(inputs),
                "n_answers": len(answers),
                "n_labels": len(labels),
                "judge_errors": failures,
                "search_requests": searches,
                "queries_returning_atomic_facts": fact_queries,
                "queries_returning_empty_memory": empty_queries,
                "returned_fact_occurrences": fact_occurrences,
                "scores": entry.get("scores"),
            }
        )
    result = {
        "at": datetime.now(UTC).isoformat(),
        "commit": manifest["commit"],
        "raw_pairs": len(parents),
        "derived_facts": len(facts),
        "coverage_rows": len(coverage),
        "all_stored_fact_quotes_and_users_verified": True,
        "all_returned_fact_contents_and_users_verified": True,
        "runs": summary,
    }
    (OUT / "audit-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(progress() if args.progress else audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
