"""共同路径整理的固定 56 题诊断；HTTP 取证，模型调用串行，不跑全量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "generic-evidence-20261005"
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402
from eval.harness.judge import EXTRA_DATASETS, render_memories  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--scope", choices=("regression", "generalization"), required=True)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()
    if args.scope == "regression":
        manifest = json.loads((OLD / "regression-manifest.json").read_text())
    else:
        manifests = [
            json.loads((OLD / name).read_text())
            for name in ("manifest.json", "heldout-manifest.json")
        ]
        manifest = {"top_k": 100, "groups": [g for m in manifests for g in m["groups"]]}
    folder = OUT / args.phase / args.scope
    frozen = freeze(manifest)
    path = folder / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    write(path, frozen)
    rows = []
    with ServiceClient(args.base_url, timeout=180) as client:
        for group in manifest["groups"]:
            path = Path(group["source"])
            items = {x["id"]: x for x in map(json.loads, path.read_text().splitlines())}
            for qid in group["qids"]:
                item = dict(items[qid])
                hits = client.search(user_id=group["user_id"], query=item["question"], top_k=100)
                context = render_memories(hits)
                digest = hashlib.sha256(context.encode()).hexdigest()
                row = {
                    "qid": qid,
                    "dataset": group["dataset"],
                    "user_id": group["user_id"],
                    "query": item["question"],
                    "hits": [asdict(x) for x in hits],
                    "context_sha256": digest,
                }
                case = folder / group["user_id"] / qid
                if (case / "search.json").exists():
                    previous = json.loads((case / "search.json").read_text())
                    assert previous["context_sha256"] == digest, (
                        "上下文变化须用新 phase，禁止复用旧答案"
                    )
                write(case / "search.json", row)
                if args.answer:
                    item[
                        "retrieved_context"
                        if group["dataset"] in EXTRA_DATASETS
                        else "speaker_1_memories"
                    ] = context
                    if group["dataset"] not in EXTRA_DATASETS:
                        item["speaker_2_memories"] = ""
                    result = run_judge(
                        pipeline_for(benchmark_dir(), group["dataset"]),
                        [item],
                        case,
                        dataset=group["dataset"],
                    )[0]
                    row["result"] = asdict(result)
                rows.append(row)
                write(folder / "results.json", rows)
                print(
                    args.phase,
                    qid,
                    len(hits),
                    row.get("result", {}).get("is_correct", "captured"),
                    flush=True,
                )
    write(
        folder / "summary.json",
        {
            "n": len(rows),
            "correct": sum(r.get("result", {}).get("is_correct", False) for r in rows)
            if args.answer
            else None,
        },
    )


if __name__ == "__main__":
    main()
