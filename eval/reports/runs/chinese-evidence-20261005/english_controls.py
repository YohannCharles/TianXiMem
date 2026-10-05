"""在原文副本上重建共同索引，仅核对 8 个英文保留上下文；不调用答案模型。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tianximem.common.config import _read_env_file, load_config  # noqa: E402
from tianximem.pairing.apply import index_pair_facts  # noqa: E402
from tianximem.service.app import build_services  # noqa: E402
from tianximem.store.sqlite_store import SqliteStore  # noqa: E402

SELECTED = {
    "kb_qa-3",
    "kb_qa-7",
    "eeda8a6d",
    "conv-26#q0011",
    "conv-30#q0018",
    "conv-41#q0006",
    "conv-30#q0012",
    "conv-30#q0017",
}


def main() -> None:
    folder = ROOT / "var/chinese-evidence-20261005"
    db = folder / "english-controls-retained.db"
    assert not db.exists(), "不覆盖已有诊断库；重跑请先另行归档"
    original = sqlite3.connect(
        f"file:{ROOT}/var/unified-evidence-20261005/candidate/tianxi-snapshot.db?mode=ro", uri=True
    )
    copied = sqlite3.connect(db)
    original.backup(copied)
    original.close()
    copied.close()
    manifest = json.loads(
        (OUT.parent / "generic-evidence-20261005/regression-manifest.json").read_text()
    )
    groups = [g for g in manifest["groups"] if SELECTED.intersection(g["qids"])]
    store = SqliteStore.open(db)
    with store.transaction() as conn:
        for user in {g["user_id"] for g in groups}:
            for pair in store.iter_pairs(conn, user_id=user):
                index_pair_facts(store, conn, pair)
    env = _read_env_file(ROOT / ".env")
    env.update(
        TIANXIMEM_SQLITE_PATH=str(db),
        TIANXIMEM_PROFILE="local",
        TIANXIMEM_METRICS_PATH=str(folder / "english-retained-metrics.json"),
    )
    services = build_services(
        load_config(env, config_dir=ROOT / "configs/runs/unified-evidence-20261005/candidate")
    )
    rows = []
    for group in groups:
        items = {
            x["id"]: x for x in map(json.loads, Path(group["source"]).read_text().splitlines())
        }
        for qid in group["qids"]:
            if qid not in SELECTED:
                continue
            old = json.loads(
                (
                    OUT.parent
                    / "unified-evidence-20261005/candidate-repaired/regression"
                    / group["user_id"]
                    / qid
                    / "search.json"
                ).read_text()
            )
            hits = services.search.run(
                user_id=group["user_id"], query=items[qid]["question"], top_k=100
            ).items
            digest = hashlib.sha256("\n".join(h.content for h in hits).encode()).hexdigest()
            rows.append(
                {
                    "qid": qid,
                    "comparison_phase": "candidate-repaired",
                    "hits": len(hits),
                    "context_sha256": digest,
                    "context_unchanged": digest == old["context_sha256"],
                }
            )
            print(qid, rows[-1]["context_unchanged"], flush=True)
    (OUT / "english-controls-fresh.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    assert len(rows) == 8 and all(row["context_unchanged"] for row in rows)


if __name__ == "__main__":
    main()
