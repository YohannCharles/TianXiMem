"""复现手工证据预检及仅原文的 HTTP Add；不导入产品模块。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402


def manual() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    frozen = freeze(manifest)
    path = OUT / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    facts = json.loads((OUT / "manual-facts.json").read_text())
    sources = json.loads((OUT / "sources.json").read_text())
    for domain, records in facts.items():
        for record in records:
            assert record["quote"] in sources[domain][record["source_slot"]]
    results = []
    for line in (OUT / "manual-input.jsonl").read_text().splitlines():
        item = json.loads(line)
        result = run_judge(
            pipeline_for(benchmark_dir(), "corporatebench"),
            [item],
            OUT / "manual" / item["id"],
            dataset="corporatebench",
        )[0]
        results.append(asdict(result))
        (OUT / "manual-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(item["id"], result.is_correct, flush=True)


def ingest(base_url: str) -> None:
    sources = json.loads((OUT / "sources.json").read_text())
    timestamp = json.loads((OUT / "protocol.json").read_text())["timestamp"]
    with httpx.Client(timeout=30) as health_client:
        health_client.get(base_url.rstrip("/") + "/health").raise_for_status()
    with ServiceClient(base_url, timeout=180) as client:
        for domain, texts in sources.items():
            request_id = f"generalization-original-{domain}"
            user_id = f"generalization-20261005-{domain}"
            session_id = f"original-{domain}"
            response = client.add(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                messages=[
                    {"role": "user", "content": text, "timestamp": timestamp} for text in texts
                ],
            )
            assert response == {
                "success": True,
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
            }
            print(domain, "source Add completed", flush=True)
    conn = sqlite3.connect(
        f"file:{ROOT / 'var/generalization-20261005/tianxi.db'}?mode=ro", uri=True
    )
    conn.row_factory = sqlite3.Row
    snapshot, domains = [], {}
    with httpx.Client(timeout=30) as vector_client:
        for domain, texts in sources.items():
            user = f"generalization-20261005-{domain}"
            stored = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM qa_pairs WHERE user_id=? ORDER BY local_index", (user,)
                )
            ]
            assert [row["question"] for row in stored] == texts
            assert all(row["answer"] is None for row in stored)
            assert all(row["event_time"] == timestamp for row in stored)
            response = vector_client.post(
                "http://127.0.0.1:6333/collections/memories_generalization_20261005/points/count",
                json={
                    "exact": True,
                    "filter": {"must": [{"key": "user_id", "match": {"value": user}}]},
                },
            )
            response.raise_for_status()
            count = response.json()["result"]["count"]
            assert count == len(texts)
            domains[domain] = {
                "raw_records": len(stored),
                "vectors": count,
                "original_text_equal": True,
            }
            snapshot.extend(stored)
    batches = [
        dict(row) for row in conn.execute("SELECT * FROM applied_batches ORDER BY request_id")
    ]
    assert len(batches) == 4 and len(snapshot) == 28
    record = {
        "domains": domains,
        "raw_records": len(snapshot),
        "vectors": 28,
        "applied_batches": len(batches),
        "all_checks_passed": True,
        "raw_snapshot": snapshot,
        "batches_snapshot": batches,
    }
    path = OUT / "ingest-verification.json"
    encoded = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        assert path.read_text() == encoded
    path.write_text(encoded)
    conn.close()
    print("original source ingestion completed 4 domains", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manual", "ingest"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8093")
    args = parser.parse_args()
    if args.stage == "manual":
        manual()
    else:
        ingest(args.base_url)


if __name__ == "__main__":
    main()
