"""只读库取证和 HTTP 复查；不导入产品模块，不生成或修改记忆。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.harness import ServiceClient  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402

FACT_TABLES = (
    "employment_facts",
    "source_statements",
    "source_replacements",
    "temporal_statements",
    "inventory_facts",
    "work_observations",
    "participation_observations",
    "status_observations",
    "personal_references",
    "friend_contexts",
)


def rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


def fingerprint(conn: sqlite3.Connection) -> str:
    snapshot = {
        table: sorted(rows(conn, table), key=lambda row: json.dumps(row, sort_keys=True))
        for table in ("qa_pairs", "applied_batches", *FACT_TABLES, "fact_index_coverage")
    }
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8093")
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert freeze(manifest) == json.loads((OUT / "frozen.json").read_text())
    conn = sqlite3.connect(
        f"file:{ROOT / 'var/generalization-20261005/tianxi.db'}?mode=ro", uri=True
    )
    conn.row_factory = sqlite3.Row
    raw = {row["id"]: row for row in rows(conn, "qa_pairs")}
    ingested = json.loads((OUT / "ingest-verification.json").read_text())
    assert raw == {row["id"]: row for row in ingested["raw_snapshot"]}
    assert (
        sorted(rows(conn, "applied_batches"), key=lambda row: row["request_id"])
        == (ingested["batches_snapshot"])
    )
    facts, extraction = {}, {}
    for table in FACT_TABLES:
        for fact in rows(conn, table):
            parent = raw[fact["parent_memory_id"]]
            assert parent["user_id"] == fact["user_id"]
            domain = fact["user_id"].removeprefix("generalization-20261005-")
            extraction.setdefault(domain, {}).setdefault(table, []).append(fact)
            if "id" in fact:
                facts[fact["id"]] = {**fact, "table": table}
            # 本次只有任职/志愿地点事实；计划、非成员、虚构引用不应成为事实。
            assert table in {"employment_facts", "friend_contexts"}, table
            assert parent["local_index"] in {0, 1, 2, 3}
            assert parent["question"].startswith(
                f"document: Message-ID: generalization-{parent['local_index']}"
                if table == "employment_facts"
                else fact["person"] + ": "
            )
            if table == "employment_facts":
                for value in (fact["person"], fact["organization"], fact["role"]):
                    assert value in parent["question"]
                assert (
                    f"\n{fact['person']}\n{fact['role']}\n{fact['organization']}"
                    in parent["question"]
                )
            else:
                assert fact["kind"] == "volunteer-place"
                quote = fact["content"].split("Source quotation: ", 1)[1]
                assert quote in parent["question"]
                assert "I volunteer at Harbor Shelter." in quote
    assert len(facts) == 8
    before_replays = fingerprint(conn)
    audited, pairs = [], []
    with ServiceClient(args.base_url, timeout=180) as client:
        for group in manifest["groups"]:
            captured = {}
            for qid in group["qids"]:
                path = OUT / "http" / group["user_id"] / qid / "search.json"
                saved = json.loads(path.read_text())
                hits = saved["hits"]
                assert 0 < len(hits) <= manifest["top_k"]
                origins = []
                for hit in hits:
                    assert set(hit) == {"id", "content", "created_at", "score"}
                    source = facts.get(hit["id"], raw.get(hit["id"]))
                    assert source is not None
                    assert source["user_id"] == group["user_id"]
                    day = datetime.fromtimestamp(source["event_time"] / 1000, UTC).date()
                    assert hit["created_at"] == day.isoformat()
                    if hit["id"] in facts:
                        assert hit["content"] == source["content"]
                        origins.append("atomic-fact")
                    else:
                        # 本批原文没有触发相对日期注解；返回正文可逐字还原原 Add。
                        assert hit["content"] == f"[{day}] Q: {source['question']}"
                        origins.append("original-memory")
                repeated = client.search_raw(
                    user_id=group["user_id"], query=saved["query"], top_k=100
                )["data"]
                assert repeated == hits
                captured[qid] = {hit["id"]: hit["content"] for hit in hits}
                audited.append(
                    {"qid": qid, "hit_count": len(hits), "origins": origins, "stable": True}
                )
            for variant in ("canonical", "paraphrase"):
                prefix = f"generalization-{group['group']}"
                assert (
                    captured[f"{prefix}-list-{variant}"] == (captured[f"{prefix}-count-{variant}"])
                )
                pairs.append({"domain": group["group"], "variant": variant, "equal": True})
            sample = json.loads(
                (OUT / "http" / group["user_id"] / group["qids"][0] / "search.json").read_text()
            )
            limited = client.search_raw(user_id=group["user_id"], query=sample["query"], top_k=1)[
                "data"
            ]
            assert len(limited) <= 1
            assert all(hit["id"] in captured[group["qids"][0]] for hit in limited)
            empty = client.search_raw(
                user_id="generalization-20261005-unknown", query=sample["query"], top_k=100
            )["data"]
            assert empty == []
    assert fingerprint(conn) == before_replays
    record = {
        "scope": "local source/shape/isolation audit, not an official AML certification",
        "original_records_and_batches_unchanged_since_add": True,
        "db_unchanged_by_http_replays": True,
        "source_ownership_valid": True,
        "all_contents_match_stored_original_or_atomic_fact": True,
        "no_query_generated_answer_content": True,
        "list_count_pairs_same_memory_set": pairs,
        "unknown_user_empty": True,
        "top_k_one_checked": True,
        "frozen_pipeline_unchanged": True,
        "db_sha256_before_after": before_replays,
        "extraction": extraction,
        "search_audit": audited,
    }
    (OUT / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("16 captured returns verified; 8 list/count pairs; 8 facts; isolation/top_k pass")
    conn.close()


if __name__ == "__main__":
    main()
