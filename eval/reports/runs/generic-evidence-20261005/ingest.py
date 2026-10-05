"""两臂只喂同一冻结原文，核对全部原文/向量后才能评分。"""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.harness import ServiceClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    sources = json.loads((OUT / "sources.json").read_text())
    with httpx.Client(timeout=10, trust_env=False) as health:
        health.get(args.base_url + "/health").raise_for_status()
    with ServiceClient(args.base_url, timeout=180) as client:
        for domain, texts in sources.items():
            response = client.add(
                request_id=f"generic-original-{domain}",
                user_id=f"generic-evidence-20261005-{domain}",
                session_id=f"original-{domain}",
                messages=[
                    {"role": "user", "content": text, "timestamp": 1756684800000} for text in texts
                ],
            )
            assert response["success"]
            print(args.arm, domain, "Add completed", flush=True)
    conn = sqlite3.connect(f"var/generic-evidence-20261005/{args.arm}/tianxi.db")
    conn.row_factory = sqlite3.Row
    snapshot = []
    with httpx.Client(timeout=30, trust_env=False) as vector:
        for domain, texts in sources.items():
            user = f"generic-evidence-20261005-{domain}"
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM qa_pairs WHERE user_id=? ORDER BY local_index", (user,)
                )
            ]
            assert [row["question"] for row in rows] == texts
            assert all(row["answer"] is None for row in rows)
            assert all(row["event_time"] == 1756684800000 for row in rows)
            response = vector.post(
                f"http://127.0.0.1:6333/collections/memories_generic_{args.arm}_20261005/points/count",
                json={
                    "exact": True,
                    "filter": {"must": [{"key": "user_id", "match": {"value": user}}]},
                },
            )
            response.raise_for_status()
            assert response.json()["result"]["count"] == len(texts)
            snapshot.extend(rows)
    users = [f"generic-evidence-20261005-{domain}" for domain in sources]
    placeholders = ",".join("?" for _ in users)
    batches = [
        dict(row)
        for row in conn.execute(
            "SELECT request_id,user_id,session_id,payload_hash FROM applied_batches "
            f"WHERE user_id IN ({placeholders}) ORDER BY request_id",
            users,
        )
    ]
    assert len(snapshot) == 42 and len(batches) == 6
    snapshot.sort(key=lambda row: row["id"])
    record = {"qa_pairs": snapshot, "applied_batches": batches}
    record["content_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    (OUT / f"ingest-{args.arm}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    if args.arm == "candidate":
        assert record == json.loads((OUT / "ingest-baseline.json").read_text())
    print(args.arm, "verified 42 original records and vectors", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
