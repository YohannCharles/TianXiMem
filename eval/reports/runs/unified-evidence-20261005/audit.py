"""只读核对原文、旧派生表、共同事实出处、HTTP 返回和冻结答案链。"""

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
VAR = ROOT / "var/unified-evidence-20261005"
OLD = ROOT / "var/generic-evidence-20261005"
sys.path.insert(0, str(ROOT))

from eval.harness import ServiceClient  # noqa: E402


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_snapshot(conn: sqlite3.Connection) -> dict:
    result = {}
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        assert name.replace("_", "").isalnum()
        rows = sorted(
            json.dumps(dict(r), sort_keys=True) for r in conn.execute(f"SELECT * FROM {name}")
        )
        result[name] = {
            "n": len(rows),
            "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
        }
    return result


def open_read(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def product() -> dict:
    paths = list((ROOT / "src/tianximem").rglob("*.py"))
    paths += [ROOT / "src/tianximem/store/schema.sql"]
    paths += list((ROOT / "configs/runs/unified-evidence-20261005").rglob("*.yaml"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def sources() -> None:
    report = {}
    for arm, source in (("candidate", "regression"), ("generalization", "candidate")):
        before = open_read(OLD / source / "tianxi-snapshot.db")
        after = open_read(VAR / arm / "tianxi.db")
        old, new = table_snapshot(before), table_snapshot(after)
        for name, value in old.items():
            assert new[name] == value, (arm, name)
        parents = {r["id"]: dict(r) for r in after.execute("SELECT * FROM qa_pairs")}
        facts = [dict(r) for r in after.execute("SELECT * FROM memory_facts")]
        for fact in facts:
            parent = parents[fact["parent_memory_id"]]
            assert fact["user_id"] == parent["user_id"]
            assert fact["source_side"] in {"question", "answer"}
            assert fact["source_quote"] and fact["source_quote"] in (
                parent[fact["source_side"]] or ""
            )
            assert fact["event_time"] == parent["event_time"]
            qualifiers = json.loads(fact["qualifiers"])
            if "quantity" in qualifiers:
                assert isinstance(qualifiers["quantity"], int)
                assert "Source quotation:" not in fact["content"]
        report[arm] = {
            "original_tables": {
                k: v for k, v in old.items() if k in {"qa_pairs", "applied_batches"}
            },
            "legacy_derived_tables": {
                k: v for k, v in old.items() if k not in {"qa_pairs", "applied_batches"}
            },
            "all_previous_rows_unchanged": True,
            "common_facts": len(facts),
            "same_user_and_exact_source_side_quote": True,
        }
        snapshot = VAR / arm / "tianxi-snapshot.db"
        with sqlite3.connect(snapshot) as target:
            after.backup(target)
        report[arm]["snapshot_sha256"] = digest(snapshot)
        before.close()
        after.close()
    with httpx.Client(timeout=30, trust_env=False) as client:
        for collection in ("memories_goal_20261005", "memories_generic_candidate_20261005"):
            response = client.get(f"http://127.0.0.1:6333/collections/{collection}")
            response.raise_for_status()
            info = response.json()["result"]
            report[collection] = {
                k: info[k] for k in ("points_count", "indexed_vectors_count", "status")
            }
    write("source-audit.json", report)
    print("Source and legacy tables unchanged; all common facts have exact, same-user sources")


def returns(phase: str) -> None:
    checks = []
    for scope, arm, port in (
        ("regression", "candidate", 8101),
        ("generalization", "generalization", 8103),
    ):
        rows = json.loads((OUT / phase / scope / "results.json").read_text())
        assert len(rows) == {"regression": 24, "generalization": 32}[scope], (scope, len(rows))
        conn = open_read(VAR / arm / "tianxi.db")
        facts = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM memory_facts")}
        parents = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM qa_pairs")}
        with ServiceClient(f"http://127.0.0.1:{port}", timeout=180) as client:
            for row in rows:
                hits = client.search(user_id=row["user_id"], query=row["query"], top_k=100)
                assert [asdict(hit) for hit in hits] == row["hits"], row["qid"]
                for index, hit in enumerate(hits):
                    assert hit.score == 1 / (index + 1)
                    if hit.id in facts:
                        fact = facts[hit.id]
                        assert hit.content == fact["content"]
                        assert fact["user_id"] == row["user_id"]
                    else:
                        assert hit.id in parents and parents[hit.id]["user_id"] == row["user_id"]
                if scope == "generalization":
                    assert len(hits) == 3 and all(hit.id in facts for hit in hits)
                    single = client.search(user_id=row["user_id"], query=row["query"], top_k=1)
                    assert [asdict(hit) for hit in single] == [asdict(hits[0])]
                checks.append(
                    {"qid": row["qid"], "scope": scope, "n": len(hits), "source_audited": True}
                )
            assert not client.search(
                user_id="unified-evidence-missing-user", query=rows[0]["query"], top_k=100
            )
        conn.close()
    baseline = json.loads((OUT / "baseline/regression/frozen.json").read_text())
    candidate = json.loads((OUT / phase / "regression/frozen.json").read_text())
    assert baseline == candidate, "答案模型、提示词、题目或评分口径变了"
    assert len(checks) == 56
    write("returned-evidence-audit.json", checks)
    print(
        "56 HTTP contexts match frozen results and original/atomic memories; answer chain unchanged"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "sources", "returns", "final"))
    parser.add_argument("--phase", default="candidate-repaired")
    args = parser.parse_args()
    current = product()
    frozen_path = OUT / "product-frozen.json"
    if args.stage == "freeze":
        assert not frozen_path.exists()
        write("product-frozen.json", current)
    else:
        assert current == json.loads(frozen_path.read_text()), (
            "产品代码或实验配置变化，必须另开 phase"
        )
        if args.stage in {"sources", "final"}:
            sources()
        if args.stage in {"returns", "final"}:
            returns(args.phase)


if __name__ == "__main__":
    main()
