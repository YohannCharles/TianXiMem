"""只读 HTTP/SQLite 核对：冻结后上下文、同库开关对照及原始来源。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.harness import ServiceClient  # noqa: E402
from eval.harness.judge import render_memories  # noqa: E402


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def context_hash(hits) -> str:
    return hashlib.sha256(render_memories(hits).encode()).hexdigest()


def regression() -> None:
    manifest = json.loads((OUT / "regression-manifest.json").read_text())
    items = {
        item["id"]: item
        for item in map(json.loads, (OUT / "regression-input.jsonl").read_text().splitlines())
    }
    previous = {
        item["qid"]: item for item in json.loads((OUT / "regression-contexts.json").read_text())
    }
    checks = []
    with (
        ServiceClient("http://127.0.0.1:8098", timeout=180) as on,
        ServiceClient("http://127.0.0.1:8099", timeout=180) as off,
    ):
        for group in manifest["groups"]:
            qid, user = group["qids"][0], group["user_id"]
            query = items[qid]["question"]
            active = on.search(user_id=user, query=query, top_k=100)
            inactive = off.search(user_id=user, query=query, top_k=100)
            check = {
                "qid": qid,
                "dataset": group["dataset"],
                "user_id": user,
                "on_hash": context_hash(active),
                "off_hash": context_hash(inactive),
                "historical_hash": previous[qid]["expected"],
                "identical": context_hash(active) == context_hash(inactive),
                "historical_identical": context_hash(active) == previous[qid]["expected"],
                "on_hits": len(active),
                "off_hits": len(inactive),
            }
            checks.append(check)
            folder = OUT / "regression-corrected" / user / qid
            folder.mkdir(parents=True, exist_ok=True)
            for arm, hits in (("on", active), ("off", inactive)):
                (folder / f"search-{arm}.json").write_text(
                    json.dumps(
                        {"query": query, "hits": [asdict(hit) for hit in hits]},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
            write("regression-controlled-parity.json", checks)
            print(qid, check["identical"], check["historical_identical"], flush=True)
    assert len(checks) == 24 and all(row["identical"] for row in checks)


def parity() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    items = {
        item["id"]: item for item in map(json.loads, (OUT / "input.jsonl").read_text().splitlines())
    }
    checks = []
    with ServiceClient("http://127.0.0.1:8097", timeout=180) as client:
        for group in manifest["groups"]:
            contexts = {}
            for qid in group["qids"]:
                original = json.loads(
                    (OUT / "candidate" / group["user_id"] / qid / "search.json").read_text()
                )
                hits = client.search(
                    user_id=group["user_id"], query=items[qid]["question"], top_k=100
                )
                old_contents = [hit["content"] for hit in original["hits"]]
                contents = [hit.content for hit in hits]
                assert old_contents == contents, qid
                contexts[qid] = contents
                checks.append({"qid": qid, "identical": True, "hit_count": len(hits)})
            qids = group["qids"]
            assert contexts[qids[0]] == contexts[qids[1]]
            assert contexts[qids[2]] == contexts[qids[3]]
        assert not client.search(
            user_id="generic-evidence-20261005-missing-user",
            query="Who are the patients at Aspen Clinic?",
            top_k=100,
        )
    write("candidate-final-context-parity.json", checks)
    print("final candidate context parity", len(checks), flush=True)


def returns() -> None:
    path = ROOT / "var/generic-evidence-20261005/candidate/tianxi.db"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        facts = {}
        for table in ("grounded_evidence", "employment_facts"):
            facts.update({row["id"]: dict(row) for row in conn.execute(f"SELECT * FROM {table}")})
    checks = []
    with ServiceClient("http://127.0.0.1:8097", timeout=180) as client:
        for manifest_name, arm in (
            ("manifest.json", "candidate"),
            ("heldout-manifest.json", "heldout-candidate"),
        ):
            manifest = json.loads((OUT / manifest_name).read_text())
            for group in manifest["groups"]:
                contents = []
                for qid in group["qids"]:
                    saved = json.loads(
                        (OUT / arm / group["user_id"] / qid / "search.json").read_text()
                    )
                    hits = client.search(user_id=group["user_id"], query=saved["query"], top_k=100)
                    assert [asdict(hit) for hit in hits] == saved["hits"], qid
                    assert len(hits) == 3
                    for index, hit in enumerate(hits):
                        fact = facts[hit.id]
                        assert fact["user_id"] == group["user_id"]
                        assert hit.content == fact["content"]
                        assert hit.created_at == "2025-09-01"
                        assert hit.score == 1 / (index + 1)
                    contents.append([hit.content for hit in hits])
                    single = client.search(user_id=group["user_id"], query=saved["query"], top_k=1)
                    assert [asdict(hit) for hit in single] == [asdict(hits[0])]
                    checks.append({"qid": qid, "stored_atomic_facts": True, "top_k": True})
                assert contents[0] == contents[1]
                assert contents[2] == contents[3]
    assert len(checks) == 32
    write("returned-evidence-audit.json", checks)
    print("32 HTTP contexts match stored atomic facts, pairs agree, top_k obeyed", flush=True)


def sources() -> None:
    arms = {}
    with httpx.Client(timeout=30, trust_env=False) as vector:
        for arm in ("baseline", "candidate"):
            path = ROOT / "var/generic-evidence-20261005" / arm / "tianxi.db"
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(row) for row in conn.execute("SELECT * FROM qa_pairs ORDER BY id")]
                batches = [
                    dict(row)
                    for row in conn.execute(
                        "SELECT request_id,user_id,session_id,payload_hash FROM applied_batches "
                        "ORDER BY request_id"
                    )
                ]
                expected_users = {}
                for filename, infix in (("sources.json", ""), ("heldout-sources.json", "heldout-")):
                    for domain, texts in json.loads((OUT / filename).read_text()).items():
                        expected_users[f"generic-evidence-20261005-{infix}{domain}"] = texts
                for user, texts in expected_users.items():
                    selected = sorted(
                        (row for row in rows if row["user_id"] == user),
                        key=lambda row: row["local_index"],
                    )
                    assert [row["question"] for row in selected] == texts
                    assert all(row["answer"] is None for row in selected)
                    assert all(row["event_time"] == 1756684800000 for row in selected)
                    response = vector.post(
                        "http://127.0.0.1:6333/collections/"
                        f"memories_generic_{arm}_20261005/points/count",
                        json={
                            "exact": True,
                            "filter": {"must": [{"key": "user_id", "match": {"value": user}}]},
                        },
                    )
                    response.raise_for_status()
                    assert response.json()["result"]["count"] == len(texts)
                assert len(rows) == 56 and len(batches) == 8
                initial = json.loads((OUT / f"ingest-{arm}.json").read_text())
                original_users = {row["user_id"] for row in initial["qa_pairs"]}
                assert [row for row in rows if row["user_id"] in original_users] == initial[
                    "qa_pairs"
                ]
                if arm == "candidate":
                    evidence = [
                        dict(row) for row in conn.execute("SELECT * FROM grounded_evidence")
                    ]
                    parents = {row["id"]: row for row in rows}
                    for fact in evidence:
                        parent = parents[fact["parent_memory_id"]]
                        assert fact["user_id"] == parent["user_id"]
                        assert fact["source_quote"] in parent["question"]
                        assert fact["subject"] in parent["question"]
                        assert fact["event_time"] == parent["event_time"]
                    write(
                        "grounding-audit.json",
                        {"facts": len(evidence), "exact_source_quotes": True, "same_user": True},
                    )
                arms[arm] = {"qa_pairs": rows, "applied_batches": batches}
    assert arms["baseline"] == arms["candidate"]
    write("final-source-audit.json", {"original_rows": 56, "batches": 8, "arms_identical": True})
    print("both arms: 56 identical original memories and vectors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("regression", "parity", "sources", "returns"))
    args = parser.parse_args()
    {"regression": regression, "parity": parity, "sources": sources, "returns": returns}[
        args.stage
    ]()
