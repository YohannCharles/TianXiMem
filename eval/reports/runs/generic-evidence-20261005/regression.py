"""仅 24 个旧诊断查询的 HTTP 上下文对照；另重答 8 个先前修复案例。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "targeted-six-hours-20261005"
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402
from eval.harness.judge import EXTRA_DATASETS, render_memories  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    args = parser.parse_args()
    old = json.loads((OLD / "final-retained-context-parity.json").read_text())["records"]
    selected = []
    for run, qids in (
        (
            "cb-work-observations",
            {"kb_qa-3", "kb_qa-7", "kb_qa-12", "kb_qa-37", "kb_qa-49", "kb_qa-25"},
        ),
        ("mqk-covered", None),
        ("tr-temporal", None),
        ("lme-inventory-source-order", None),
        ("locomo-friends", {"conv-26#q0011", "conv-30#q0018", "conv-41#q0006"}),
        ("locomo-status", {"conv-30#q0012", "conv-30#q0017"}),
    ):
        rows = [row for row in old if row["run"] == run and (qids is None or row["qid"] in qids)]
        selected.extend(rows[:3] if run in {"mqk-covered", "tr-temporal"} else rows)
    mtb = json.loads((OLD / "final-mtb-context-parity.json").read_text())["records"]
    selected.extend([row for row in mtb if row["qid"] == "unclear_200-0163"][:1])
    assert len(selected) == 24, len(selected)
    items = []
    for row in selected:
        path = OLD / row["run"] / row["user_id"] / row["qid"] / "input.jsonl"
        item = json.loads(path.read_text().splitlines()[0])
        for field in ("retrieved_context", "speaker_1_memories", "speaker_2_memories"):
            item.pop(field, None)
        items.append(item)
    encoded = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    path = OUT / "regression-input.jsonl"
    if path.exists():
        assert path.read_text() == encoded
    path.write_text(encoded)
    manifest = {
        "top_k": 100,
        "groups": [
            {
                "dataset": row["dataset"],
                "user_id": row["user_id"],
                "group": row["run"],
                "source": str(path.relative_to(ROOT)),
                "qids": [row["qid"]],
            }
            for row in selected
        ],
    }
    write("regression-manifest.json", manifest)
    frozen = freeze(manifest)
    path = OUT / "regression-frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    write("regression-frozen.json", frozen)
    reanswer = {
        "kb_qa-3",
        "kb_qa-7",
        "kb_qa-12",
        "kb_qa-37",
        "CF3k-2-0",
        "test_l2-1888",
        "eeda8a6d",
        "conv-41#q0006",
    }
    checks, results = [], []
    with ServiceClient(args.base_url, timeout=180) as client:
        for row, item in zip(selected, items, strict=True):
            hits = client.search(user_id=row["user_id"], query=item["question"], top_k=100)
            context = render_memories(hits)
            digest = hashlib.sha256(context.encode()).hexdigest()
            expected = row.get("final_context_sha256", row.get("old_context_sha256"))
            check = {
                "qid": row["qid"],
                "dataset": row["dataset"],
                "run": row["run"],
                "expected": expected,
                "actual": digest,
                "identical": expected == digest,
                "hit_count": len(hits),
            }
            checks.append(check)
            write("regression-contexts.json", checks)
            print(row["run"], row["qid"], "same context", check["identical"], flush=True)
            folder = OUT / "regression" / row["user_id"] / row["qid"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "search.json").write_text(
                json.dumps(
                    {"query": item["question"], "hits": [asdict(hit) for hit in hits]},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            if row["qid"] in reanswer:
                field = (
                    "retrieved_context"
                    if row["dataset"] in EXTRA_DATASETS
                    else "speaker_1_memories"
                )
                item[field] = context
                if row["dataset"] not in EXTRA_DATASETS:
                    item["speaker_2_memories"] = ""
                result = run_judge(
                    pipeline_for(benchmark_dir(), row["dataset"]),
                    [item],
                    folder,
                    dataset=row["dataset"],
                )[0]
                results.append({"dataset": row["dataset"], **asdict(result)})
                write("regression-results.json", results)
                print("  answer", result.is_correct, result.generated_answer, flush=True)
    assert all(row["identical"] for row in checks)
    assert len(results) == 8


if __name__ == "__main__":
    main()
