"""Serial replay of the seven historical frozen datasets through real HTTP.

The product, answer prompts and scorers remain frozen. A separate facts-off
CorporateBench run controls for its answer-prompt change since the old baseline.
Raw searches, failures and SQLite snapshots are retained outside tracked reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

NAME = "facts-benchmark-20261005"
OUT = ROOT / "eval/reports/runs" / NAME
VAR = ROOT / "var" / NAME
CONFIGS = ROOT / "configs/runs" / NAME
SETS = (
    "corporatebench",
    "mquake-remastered",
    "tempreason",
    "memtrapbench",
    "locomo-refined",
    "longmemeval-s",
    "medmemorybench",
)
HISTORY = {
    "corporatebench": "base-cb-rescore-20261004",
    "mquake-remastered": "mqk-fixtest",
    "tempreason": "tr-promptfix",
    "memtrapbench": "mtb-fixtest",
    "locomo-refined": "base-locomo0",
    "longmemeval-s": "base-lme2",
}
PORTS = {"on": 8063, "off": 8064}


def now() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    files = set((ROOT / "src/tianximem").rglob("*.py"))
    files.update((ROOT / "src/tianximem").rglob("*.sql"))
    files.update((ROOT / "eval/datasets").rglob("*.py"))
    files.update((ROOT / "eval/harness").rglob("*.py"))
    files.update((ROOT / "eval/experiments" / n) for n in ("run.py", "recipes.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare() -> dict:
    from eval.datasets import benchmark_dir, data_fingerprint
    from eval.experiments.recipes import recipe_for
    from eval.experiments.run import _load, judge_preconditions
    from eval.harness import pipeline_for
    from eval.harness.api_config import ANSWER_MODEL, JUDGE_MODEL

    if judge_preconditions():
        raise RuntimeError("Missing answer/judge configuration; run with uv --env-file .env")
    if (OUT / "manifest.json").exists():
        manifest = json.loads((OUT / "manifest.json").read_text())
        if manifest["code_hashes"] != code_hashes():
            raise RuntimeError("Product or evaluation code changed since manifest was frozen")
        return manifest
    VAR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    local = yaml.safe_load((ROOT / "configs/local.yaml").read_text())
    collection = "memories_facts_benchmark_20261005"
    for label in PORTS:
        parent = CONFIGS / label
        parent.mkdir(parents=True, exist_ok=True)
        arm = json.loads(json.dumps(cfg))
        arm["retrieval"]["grounded_evidence"] = label == "on"
        arm_local = json.loads(json.dumps(local))
        arm_local["storage"]["qdrant"]["collection"] = collection
        (parent / "default.yaml").write_text(
            yaml.safe_dump(arm, allow_unicode=True, sort_keys=False)
        )
        (parent / "local.yaml").write_text(
            yaml.safe_dump(arm_local, allow_unicode=True, sort_keys=False)
        )
    cache_source = Path(os.environ.get("TIANXIMEM_EMBED_CACHE_DIR", "var/embed_cache"))
    cache_target = VAR / "embed_cache"
    cache_target.mkdir(exist_ok=True)
    for path in cache_source.glob("*.db"):
        with (
            sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as source,
            sqlite3.connect(cache_target / path.name) as target,
        ):
            source.backup(target)
    metadata = []
    pipelines = {}
    for dataset in SETS:
        recipe = recipe_for(dataset)
        samples = _load(dataset, benchmark_dir(), recipe.limit, spread=recipe.spread)
        qids = [q.qid for sample in samples for q in sample.questions]
        if len(qids) != recipe.n_questions or len(set(qids)) != len(qids):
            raise RuntimeError(f"Unexpected question selection: {dataset}")
        old_name = HISTORY.get(dataset)
        old = None
        old_qids = []
        if old_name:
            old_path = ROOT / "eval/reports/runs" / (old_name + ".json")
            old = json.loads(old_path.read_text())
            # The rescore directory contains the same original input as base-cb.
            for path in sorted((old_path.parent / old_name).glob("*/input.jsonl")):
                old_qids += [r["id"] for r in jsonl(path)]
            if not old_qids and dataset == "corporatebench":
                for path in (old_path.parent / "base-cb").glob("*/input.jsonl"):
                    old_qids += [r["id"] for r in jsonl(path)]
            if set(old_qids) != set(qids):
                raise RuntimeError(f"Historical question set changed: {dataset}")
        fingerprint = data_fingerprint(
            benchmark_dir(),
            dataset,
            n_samples=len(samples),
            n_questions=len(qids),
            add_shape="official",
        )
        metadata.append(
            {
                "dataset": dataset,
                "recipe": {"limit": recipe.limit, "spread": recipe.spread},
                "n_questions": len(qids),
                "n_samples": len(samples),
                "user_ids": [s.user_id for s in samples],
                "qids": qids,
                "data_fingerprint": fingerprint,
                "historical_run": old_name,
                "historical_overall": old["scores"]["overall"] if old else None,
                "historical_input_ids_match": bool(old_qids),
            }
        )
        pipeline = pipeline_for(benchmark_dir(), dataset)
        pipelines[str(pipeline.resolve())] = sha(pipeline)
        print(f"Prepared {dataset}: {len(qids)} questions", flush=True)
    manifest = {
        "created_at": now(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Complete historical frozen question sets, not original-dataset full scale",
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "max_answer_tokens": 1024,
        "top_k": 100,
        "add_shape": "official",
        "collection": collection,
        "code_hashes": code_hashes(),
        "pipeline_hashes": pipelines,
        "config_hashes": {str(p.relative_to(ROOT)): sha(p) for p in CONFIGS.rglob("*.yaml")},
        "datasets": metadata,
    }
    write(OUT / "manifest.json", manifest)
    return manifest


def assert_frozen(manifest: dict) -> None:
    if code_hashes() != manifest["code_hashes"]:
        raise RuntimeError("Product/evaluation code changed during evaluation")
    for filename, digest in manifest["pipeline_hashes"].items():
        if sha(Path(filename)) != digest:
            raise RuntimeError(f"Pipeline changed: {filename}")
    for filename, digest in manifest["config_hashes"].items():
        if sha(ROOT / filename) != digest:
            raise RuntimeError(f"Frozen config changed: {filename}")


def child(args: list[str]) -> int:
    from eval.experiments import run
    from eval.harness import ServiceClient

    log_path = Path(os.environ["FACTS_SUITE_SEARCH_LOG"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {(r["user_id"], r["query"], r["top_k"]): r["response"] for r in jsonl(log_path)}

    class RecordingClient(ServiceClient):
        def search_raw(self, *, user_id: str, query: str, top_k: int) -> dict:
            previous = cached.get((user_id, query, top_k))
            if previous is not None:
                return previous
            payload = super().search_raw(user_id=user_id, query=query, top_k=top_k)
            with log_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "at": now(),
                            "user_id": user_id,
                            "query": query,
                            "top_k": top_k,
                            "response": payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return payload

    original_judge = run.run_judge

    def unchanged_input_judge(pipeline, items, out_dir, **kwargs):
        previous_input = Path(out_dir) / "input.jsonl"
        previous_answers = Path(out_dir) / "answers.jsonl"
        if (
            previous_answers.exists()
            and previous_answers.stat().st_size
            and jsonl(previous_input) != items
        ):
            raise RuntimeError("Cannot resume answers after their full input changed")
        return original_judge(pipeline, items, out_dir, **kwargs)

    run.ServiceClient = RecordingClient
    run.run_judge = unchanged_input_judge
    return run.main(args)


def run_suite(manifest: dict) -> int:
    services = {}
    handles = []
    status_path = OUT / "status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"started_at": now(), "state": "starting", "runs": []}
    )
    status.setdefault("starts", []).append(now())
    write(OUT / "status.json", status)
    env_base = os.environ.copy()
    env_base.update(
        TIANXIMEM_SQLITE_PATH=str(VAR / "tianxi.db"),
        TIANXIMEM_EMBED_CACHE_DIR=str(VAR / "embed_cache"),
        TIANXIMEM_PROFILE="local",
        TIANXIMEM_CAPTURE_PATH=str(VAR / "capture.jsonl"),
    )
    try:
        for label, port in PORTS.items():
            env = env_base | {
                "TIANXIMEM_CONFIG_DIR": str(CONFIGS / label),
                "TIANXIMEM_METRICS_PATH": str(VAR / f"metrics-{label}.json"),
            }
            handle = (VAR / f"service-{label}.log").open("a")
            handles.append(handle)
            services[label] = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "tianximem.service.app:create_app_from_env",
                    "--factory",
                    "--workers",
                    "1",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            ready = False
            for _ in range(120):
                if services[label].poll() is not None:
                    raise RuntimeError(f"Service {label} stopped during startup; see its log")
                try:
                    response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    response.raise_for_status()
                    ready = True
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            if not ready:
                raise RuntimeError(f"Service {label} did not become ready")
        write(VAR / "owned-services.json", {label: p.pid for label, p in services.items()})
        queue = [("off", "corporatebench"), *[("on", ds) for ds in SETS]]
        for label, dataset in queue:
            assert_frozen(manifest)
            rid = f"{NAME}-{label}-{dataset}"
            entry = next((r for r in status["runs"] if r["run_id"] == rid), None)
            if entry is not None and entry["state"] == "complete":
                print(f"Already complete: {label} {dataset}", flush=True)
                continue
            if entry is None:
                entry = {"dataset": dataset, "facts": label, "run_id": rid, "attempts": []}
                status["runs"].append(entry)
            entry.setdefault("attempts", []).append({"started_at": now()})
            entry.update(started_at=now(), state="running")
            status["state"] = "running"
            write(OUT / "status.json", status)
            env = env_base | {
                "TIANXIMEM_CONFIG_DIR": str(CONFIGS / label),
                "TIANXIMEM_METRICS_PATH": str(VAR / f"metrics-{label}.json"),
                "FACTS_SUITE_SEARCH_LOG": str(OUT / f"searches-{label}-{dataset}.jsonl"),
            }
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset-run",
                "--dataset",
                dataset,
                "--frozen",
                "--add-shape",
                "official",
                "--base-url",
                f"http://127.0.0.1:{PORTS[label]}",
                "--run-id",
                rid,
                "--configs-dir",
                str(CONFIGS / label),
                "--profile",
                "local",
                "--step",
                "add-search-facts-replay",
                "--embedder",
                "qwen3-embedding-8b",
                "--llm",
                manifest["answer_model"],
                "--reranker",
                "disabled (local profile)",
                "--metrics",
                str(VAR / f"metrics-{label}.json"),
                "--switches",
                json.dumps(
                    {
                        "retrieval.grounded_evidence": label == "on",
                        "neighbor.radius": 0,
                        "rerank.enabled": False,
                    }
                ),
                "--notes",
                (
                    "Complete historical frozen selection; new answer generation. "
                    "Product/prompt/scorer hashes pinned in suite manifest. "
                    "CorporateBench facts-off control uses the current answer prompt."
                ),
            ]
            entry["argv"] = argv
            write(OUT / "status.json", status)
            print(f"RUN {label} {dataset} → {rid}", flush=True)
            with (VAR / f"run-{label}-{dataset}.log").open("a") as handle:
                completed = subprocess.run(
                    argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            entry.update(finished_at=now(), exit_code=completed.returncode)
            entry["attempts"][-1].update(
                finished_at=entry["finished_at"], exit_code=completed.returncode
            )
            record_path = ROOT / "eval/reports/runs" / (rid + ".json")
            if completed.returncode == 0 and record_path.exists():
                record = json.loads(record_path.read_text())
                entry.update(
                    state="complete",
                    scores=record["scores"],
                    n=record["data_fingerprint"]["n_questions"],
                )
            else:
                entry["state"] = "failed"
            write(OUT / "status.json", status)
            print(f"DONE {label} {dataset}: {entry['state']}", flush=True)
        assert_frozen(manifest)
        with (
            sqlite3.connect(VAR / "tianxi.db") as source,
            sqlite3.connect(VAR / "final-snapshot.db") as target,
        ):
            source.backup(target)
        status.update(state="finished", finished_at=now())
        write(OUT / "status.json", status)
        return int(any(r["state"] != "complete" for r in status["runs"]))
    finally:
        for process in services.values():
            if process.poll() is None:
                process.terminate()
        for process in services.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def main() -> int:
    if "--dataset-run" in sys.argv:
        return child(sys.argv[sys.argv.index("--dataset-run") + 1 :])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = prepare()
    if args.prepare_only:
        print(f"Manifest → {OUT / 'manifest.json'}", flush=True)
        return 0
    return run_suite(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
