"""固定失败案例的小样本 HTTP 评测；答案与裁判复用已有 pipeline。

manifest 显式列出 dataset、user_id、source（历史 input.jsonl）、qids 和 group。
不加载金标图，不裁剪语料，不改变答案提示词。每题独立保存结果以便中断续跑，
deadline 为 UTC Unix 秒；到期不再发起检索或模型调用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.datasets.prepare import sha256_file  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402
from eval.harness.api_config import ANSWER_MODEL, JUDGE_MODEL  # noqa: E402
from eval.harness.judge import EXTRA_DATASETS, render_memories  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    """⚠ 按 `"\\n"` 切、**不用 `splitlines()`**（见 [`eval/jsonl_io.py`](../eval/jsonl_io.py)）。
    缺文件时**响亮报错**——不能返回空表，否则"产物没生成"会伪装成"跑了 0 题"。
    """
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def freeze(manifest: dict) -> dict:
    paths = {str(group["source"]) for group in manifest["groups"]}
    paths.update(
        str(pipeline_for(benchmark_dir(), group["dataset"])) for group in manifest["groups"]
    )
    paths.update(
        str(ROOT / "eval" / "harness" / name)
        for name in ("judge.py", "corporatebench_pipeline.py", "api_config.py")
    )
    return {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "files": {path: sha256_file(Path(path)) for path in sorted(paths)},
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "max_tokens": 1024,
        "temperature": 0,
        "top_k": manifest.get("top_k", 100),
        "add_shape": "official",
        "scope": "targeted diagnostic only; not a full benchmark score",
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    keys = {"all"} | {row["group"] for row in rows} | {row["dataset"] for row in rows}
    for key in sorted(keys):
        picked = [row for row in rows if key in ("all", row["group"], row["dataset"])]
        scores = [
            row["result"]["partial"] for row in picked if row["result"]["partial"] is not None
        ]
        summary[key] = {
            "n": len(picked),
            "correct": sum(row["result"]["is_correct"] for row in picked),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "mean_context_tokens": sum(row["context_tokens"] for row in picked) / len(picked),
            "mean_search_seconds": sum(row["search_seconds"] for row in picked) / len(picked),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--offline", action="store_true", help="禁止下载裁判依赖")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    from eval.datasets.prepare import ensure_dataset

    for dataset in sorted({group["dataset"] for group in manifest["groups"]}):
        ensure_dataset(dataset, benchmark_dir(), offline=args.offline, purpose="judge")
    frozen = freeze(manifest)
    if args.freeze.exists():
        if json.loads(args.freeze.read_text()) != frozen:
            raise SystemExit("冻结指纹发生变化：不能与旧结果比较")
    else:
        write_json(args.freeze, frozen)
    args.output.mkdir(parents=True, exist_ok=True)
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")
    results_path = args.output / "results.jsonl"
    results = read_rows(results_path) if results_path.exists() else []
    done = {(row["dataset"], row["user_id"], row["qid"]) for row in results}
    metadata = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "started_at": time.time(),
        "freeze": frozen,
        "base_url": args.base_url,
        "groups": args.group,
    }
    if not (args.output / "metadata.json").exists():
        write_json(args.output / "metadata.json", metadata)
    with ServiceClient(args.base_url, timeout=180) as client:
        for group in manifest["groups"]:
            if args.group and group["group"] not in args.group:
                continue
            items = {str(row["id"]): row for row in read_rows(Path(group["source"]))}
            for qid in group["qids"]:
                if (group["dataset"], group["user_id"], qid) in done:
                    continue
                if time.time() + 180 >= args.deadline:
                    print("到达停止边界，保留已完成结果", flush=True)
                    write_json(args.output / "summary.json", summarize(results))
                    return 0
                item = dict(items[qid])
                started = time.monotonic()
                hits = client.search(
                    user_id=group["user_id"], query=item["question"], top_k=frozen["top_k"]
                )
                search_seconds = time.monotonic() - started
                context = render_memories(hits)
                context_tokens = len(encoder.encode(context))
                item[
                    "retrieved_context"
                    if group["dataset"] in EXTRA_DATASETS
                    else "speaker_1_memories"
                ] = context
                if group["dataset"] not in EXTRA_DATASETS:
                    item["speaker_2_memories"] = ""
                out = args.output / group["user_id"] / qid
                write_json(
                    out / "search.json",
                    {"query": item["question"], "hits": [asdict(hit) for hit in hits]},
                )
                print(
                    f"{group['group']} {qid}: {len(hits)} 片段 / {context_tokens} tokens",
                    flush=True,
                )
                result = run_judge(
                    pipeline_for(benchmark_dir(), group["dataset"]),
                    [item],
                    out,
                    dataset=group["dataset"],
                )[0]
                row = {
                    "dataset": group["dataset"],
                    "user_id": group["user_id"],
                    "qid": qid,
                    "group": group["group"],
                    "result": asdict(result),
                    "context_tokens": context_tokens,
                    "hit_count": len(hits),
                    "search_seconds": search_seconds,
                    "total_seconds": time.monotonic() - started,
                }
                with results_path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.append(row)
                write_json(args.output / "summary.json", summarize(results))
                print(f"  correct={result.is_correct} score={result.partial}", flush=True)
    write_json(args.output / "summary.json", summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
