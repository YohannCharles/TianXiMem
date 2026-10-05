"""核对实际 HTTP 片段出处、冻结答案链和原始记忆，并保存英文保留版本对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.harness import ServiceClient  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text())


def write(name: str, data: object) -> None:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="retained")
    args = parser.parse_args()
    manifest = read(OUT / "manifest.json")
    assert (
        hashlib.sha256((ROOT / "eval/harness/corporatebench_pipeline.py").read_bytes()).hexdigest()
        == manifest["answer_pipeline_sha256"]
    )
    first = {r["qid"]: r for r in read(OUT / "candidate/results.json")}
    quantities = {r["qid"]: r for r in read(OUT / "candidate-final/results.json")}
    conn = sqlite3.connect(f"file:{ROOT}/var/chinese-evidence-20261005/tianxi.db?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    baseline = read(ROOT / "var/chinese-evidence-20261005/baseline-raw-fingerprint.json")
    raw = {}
    for table in ("qa_pairs", "applied_batches"):
        rows = [tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
        raw[table] = {
            "n": len(rows),
            "sha256": hashlib.sha256(
                json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
        }
        assert raw[table] == baseline[table]
    audited, scored = [], []
    for group in manifest["groups"]:
        group_contexts = []
        for tag, query, dtype, _ in group["questions"]:
            qid = f"zh-{group['name']}-{tag}"
            receipt = read(OUT / args.phase / group["name"] / tag / "search.json")
            assert receipt["query"] == query and len(receipt["hits"]) == 3
            context = "\n".join(h["content"] for h in receipt["hits"])
            digest = hashlib.sha256(context.encode()).hexdigest()
            answer = quantities.get(qid, first[qid])
            assert digest == answer["context_sha256"], "不能复用上下文变化前的答案"
            assert answer["is_correct"]
            scored.append(
                {
                    "qid": qid,
                    "answer_type": dtype,
                    "is_correct": answer["is_correct"],
                    "answer_origin": "fresh-quantity" if qid in quantities else "unchanged-context",
                    "context_sha256": digest,
                }
            )
            group_contexts.append(context)
            for hit in receipt["hits"]:
                f = conn.execute("SELECT * FROM memory_facts WHERE id=?", (hit["id"],)).fetchone()
                assert f and hit["content"] == f["content"]
                parent = conn.execute(
                    "SELECT * FROM qa_pairs WHERE id=?", (f["parent_memory_id"],)
                ).fetchone()
                assert parent and f["user_id"] == parent["user_id"] == f"chinese-{group['name']}"
                assert f["source_quote"] in (parent[f["source_side"]] or "")
                attrs = json.loads(f["qualifiers"])
                if "item_quote" in attrs:
                    assert attrs["item_quote"] in f["source_quote"]
                audited.append(
                    {
                        "qid": qid,
                        "fact_id": f["id"],
                        "parent_memory_id": parent["id"],
                        "source_side": f["source_side"],
                        "quote_verified": True,
                    }
                )
        assert len(set(group_contexts)) == 1, "同范围的名单/计数必须使用相同证据"
    with ServiceClient("http://127.0.0.1:8062", timeout=180) as client:
        assert (
            len(client.search(user_id="chinese-school", query="北辰学校有哪些学生？", top_k=1)) == 1
        )
        assert not client.search(user_id="chinese-absent", query="北辰学校有哪些学生？", top_k=100)
    physical = dict(
        conn.execute("SELECT version,count(*) FROM evidence_coverage GROUP BY version").fetchall()
    )
    facts = dict(
        conn.execute("SELECT version,count(*) FROM memory_facts GROUP BY version").fetchall()
    )
    conn.close()
    write("source-audit.json", audited)
    write("retained-results.json", scored)
    summary = {
        "raw_fingerprints": raw,
        "contexts": len(scored),
        "fact_occurrences": len(audited),
        "coverage_by_version": physical,
        "facts_by_version": facts,
        "correct": sum(r["is_correct"] for r in scored),
        "top_k_and_missing_user": True,
        "answer_pipeline_unchanged": True,
    }
    # 初次选到整理中的 candidate-final；实际保留代码对应 candidate-repaired。
    controls = read(OUT / "english-controls.json")
    retained_controls = []
    for row in controls:
        paths = list(
            (
                ROOT / "eval/reports/runs/unified-evidence-20261005/candidate-repaired/regression"
            ).glob("*/" + row["qid"] + "/search.json")
        )
        assert len(paths) == 1
        old = read(paths[0])
        retained_controls.append(
            {
                **row,
                "comparison_phase": "candidate-repaired",
                "context_unchanged": row["context_sha256"] == old["context_sha256"],
            }
        )
    assert len(retained_controls) == 8 and all(r["context_unchanged"] for r in retained_controls)
    write("english-controls-retained.json", retained_controls)
    fresh_controls = read(OUT / "english-controls-fresh.json")
    assert len(fresh_controls) == 8 and all(r["context_unchanged"] for r in fresh_controls)
    write("audit-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
