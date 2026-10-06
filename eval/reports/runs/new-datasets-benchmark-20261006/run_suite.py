"""Run real HTTP smoke checks, then a serial frozen subset of four new datasets.

Product and evaluation code are copied into an immutable runtime snapshot. Only
the Qdrant collection and runtime paths differ from the current local profile.
Original datasets, pinned scoring functions, and prior runs stay untouched.
Native answer contract changes require a new recorded snapshot revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[4]
NAME = "new-datasets-benchmark-20261006"
OUT = ROOT / "eval/reports/runs" / NAME
VAR = ROOT / "var" / NAME
SNAPSHOT = VAR / "source"
CONFIGS = ROOT / "configs/runs" / NAME
DATASETS = ("halumem", "musique", "hybridqa", "feverous")
PORT = 8067


def now() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staged.replace(path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def emit(path: Path, value: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def frozen_imports() -> None:
    sys.path[:0] = [str(SNAPSHOT), str(SNAPSHOT / "src")]


def smoke_samples(dataset: str, samples: list) -> list:
    """Choose coverage cases before any requests, retaining each complete corpus."""
    chosen = []
    if dataset == "halumem":
        # Early and late checkpoints exercise substantially different context sizes.
        candidates = [
            min(samples, key=lambda s: len(s.sessions)),
            max(samples, key=lambda s: len(s.sessions)),
        ]
        for sample in candidates:
            if sample.user_id not in {s.user_id for s in chosen}:
                chosen.append(replace(sample, questions=sample.questions[:1]))
    else:
        seen = set()
        for sample in samples:
            for question in sample.questions:
                category = question.category
                if dataset == "musique":
                    category = question.is_abstention
                elif dataset == "feverous":
                    category = question.gold["label"]
                if category not in seen:
                    seen.add(category)
                    chosen.append(replace(sample, questions=(question,)))
    return chosen


def batch_count(samples: list) -> int:
    from eval.harness.add_shape import shape_batch
    from eval.harness.batching import batches

    return sum(
        sum(
            1
            for _ in batches(
                shape_batch(
                    session.messages,
                    dataset=sample.dataset,
                    speaker_names=sample.speaker_names,
                    shape="official",
                )
            )
        )
        for sample in samples
        for session in sample.sessions
    )


def prepare() -> dict:
    manifest_path = OUT / "manifest.json"
    if manifest_path.is_file():
        frozen_imports()
        manifest = json.loads(manifest_path.read_text())
        check_frozen(manifest)
        return manifest
    if SNAPSHOT.exists():
        raise RuntimeError("Incomplete source snapshot; inspect preparation before retrying")
    VAR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".sql"}]
    for directory in ("datasets", "harness", "experiments"):
        paths.extend((ROOT / "eval" / directory).glob("*.py"))
    paths.extend([ROOT / "eval/reports/schema.py", Path(__file__).resolve()])
    for path in paths:
        destination = SNAPSHOT / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "default.yaml").write_bytes((ROOT / "configs/default.yaml").read_bytes())
    local = yaml.safe_load((ROOT / "configs/local.yaml").read_text())
    local["storage"]["qdrant"]["collection"] = "memories_new_datasets_20261006"
    (CONFIGS / "local.yaml").write_text(yaml.safe_dump(local, sort_keys=False))
    cache_source = Path(os.environ.get("TIANXIMEM_EMBED_CACHE_DIR", "var/embed_cache"))
    cache_target = VAR / "embed_cache"
    cache_target.mkdir(exist_ok=True)
    for path in cache_source.glob("*.db"):
        with (
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source,
            sqlite3.connect(cache_target / path.name) as target,
        ):
            source.backup(target)
    frozen_imports()
    from eval.datasets import benchmark_dir, data_fingerprint
    from eval.datasets.prepare import ensure_dataset
    from eval.experiments.recipes import recipe_for
    from eval.experiments.run import _load, judge_preconditions
    from eval.harness.api_config import ANSWER_MODEL, JUDGE_MODEL

    if judge_preconditions():
        raise RuntimeError("Answer/judge configuration missing; use uv --env-file .env")
    metadata = []
    for dataset in DATASETS:
        ensure_dataset(dataset, benchmark_dir(), offline=True)
        recipe = recipe_for(dataset)
        samples = _load(dataset, benchmark_dir(), recipe.limit, spread=recipe.spread)
        qids = [q.qid for s in samples for q in s.questions]
        if len(qids) != recipe.n_questions or len(set(qids)) != len(qids):
            raise RuntimeError(f"Frozen selection drifted: {dataset}")
        small = smoke_samples(dataset, samples)
        with (VAR / f"plan-{dataset}.pickle").open("wb") as handle:
            pickle.dump({"smoke": small, "subset": samples}, handle)
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
                "n_questions": len(qids),
                "n_samples": len(samples),
                "qids": qids,
                "recipe": {"limit": recipe.limit, "spread": recipe.spread},
                "full_batches": batch_count(samples),
                "smoke_batches": batch_count(small),
                "smoke_qids": [q.qid for s in small for q in s.questions],
                "category_counts": dict(Counter(q.category for s in samples for q in s.questions)),
                "data_fingerprint": fingerprint,
            }
        )
        print(
            f"PREPARED {dataset}: {len(qids)} subset questions, "
            f"{len(metadata[-1]['smoke_qids'])} smoke questions",
            flush=True,
        )
    frozen_files = [
        *SNAPSHOT.rglob("*.py"),
        *SNAPSHOT.rglob("*.sql"),
        *CONFIGS.glob("*.yaml"),
        *VAR.glob("plan-*.pickle"),
    ]
    manifest = {
        "created_at": now(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "top_k": 100,
        "max_answer_tokens": 1024,
        "profile": "local",
        "collection": local["storage"]["qdrant"]["collection"],
        "source_snapshot": str(SNAPSHOT),
        "code_config_plan_hashes": {str(p): sha(p) for p in frozen_files},
        "datasets": metadata,
    }
    write(manifest_path, manifest)
    write(OUT / "status.json", {"created_at": now(), "state": "prepared", "runs": []})
    return manifest


def check_frozen(manifest: dict) -> None:
    for name, digest in manifest["code_config_plan_hashes"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Frozen source/config/plan changed: {name}")


def artifact_tag(manifest: dict, phase: str, dataset: str) -> str:
    if phase == "smoke":
        return manifest.get("smoke_artifact_tags", {}).get(dataset, phase)
    return phase


def latest_run(status: dict, phase: str, dataset: str) -> dict:
    return next(
        (r for r in reversed(status["runs"]) if r["phase"] == phase and r["dataset"] == dataset),
        {},
    )


def child(phase: str, dataset: str) -> int:
    frozen_imports()
    from eval.experiments import run
    from eval.harness import ServiceClient

    manifest = json.loads((OUT / "manifest.json").read_text())
    check_frozen(manifest)
    tag = artifact_tag(manifest, phase, dataset)
    meta = next(item for item in manifest["datasets"] if item["dataset"] == dataset)
    with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
        selected = pickle.load(handle)[phase]
    run._load = lambda *args, **kwargs: selected
    telemetry = OUT / f"http-{tag}-{dataset}.jsonl"
    searches = OUT / f"searches-{tag}-{dataset}.jsonl"
    previous = {(r["user_id"], r["query"], r["top_k"]): r["response"] for r in rows(searches)}

    class RecordingClient(ServiceClient):
        def add(self, **kwargs):
            start = time.monotonic()
            response = super().add(**kwargs)
            emit(
                telemetry,
                {
                    "at": now(),
                    "kind": "add",
                    "seconds": time.monotonic() - start,
                    "messages": len(kwargs["messages"]),
                    "user_id": kwargs["user_id"],
                },
            )
            return response

        def search_raw(self, *, user_id, query, top_k):
            key = (user_id, query, top_k)
            if key in previous:
                return previous[key]
            start = time.monotonic()
            response = super().search_raw(user_id=user_id, query=query, top_k=top_k)
            emit(
                telemetry,
                {
                    "at": now(),
                    "kind": "search",
                    "seconds": time.monotonic() - start,
                    "hits": len(response["data"]),
                    "user_id": user_id,
                },
            )
            emit(
                searches,
                {
                    "at": now(),
                    "user_id": user_id,
                    "query": query,
                    "top_k": top_k,
                    "response": response,
                },
            )
            return response

    original_judge = run.run_judge

    def guarded_judge(pipeline, items, out_dir, **kwargs):
        check_frozen(manifest)
        previous_input = Path(out_dir) / "input.jsonl"
        previous_answers = Path(out_dir) / "answers.jsonl"
        if rows(previous_answers) and rows(previous_input) != items:
            raise RuntimeError("Answer resume input changed")
        start = time.monotonic()
        results = original_judge(pipeline, items, out_dir, **kwargs)
        emit(
            telemetry,
            {
                "at": now(),
                "kind": "judge",
                "seconds": time.monotonic() - start,
                "questions": len(items),
            },
        )
        return results

    run.ServiceClient = RecordingClient
    run.run_judge = guarded_judge
    argv = [
        "--dataset",
        dataset,
        "--frozen",
        "--offline",
        "--base-url",
        f"http://127.0.0.1:{PORT}",
        "--run-id",
        f"{NAME}-{tag}-{dataset}",
        "--configs-dir",
        str(CONFIGS),
        "--reports-dir",
        str(ROOT / "eval/reports"),
        "--profile",
        "local",
        "--step",
        "new-datasets-subset",
        "--embedder",
        "qwen3-embedding-8b",
        "--llm",
        manifest["answer_model"],
        "--reranker",
        "disabled (local profile)",
        "--metrics",
        str(VAR / f"metrics-{tag}-{dataset}.json"),
        "--notes",
        f"{phase}; current product/eval snapshot frozen in suite manifest; "
        "smoke qids are declared separately.",
    ]
    if phase == "smoke":
        # _load uses the declared smoke plan; record that selection explicitly.
        argv.remove("--frozen")
        argv += [
            "--limit",
            str(len(selected)),
            "--notes",
            "Real HTTP smoke selection from frozen subset; selected qids in suite manifest; "
            "complete per-question corpus retained.",
        ]
    code = run.main(argv)
    if code == 0:
        directory = ROOT / "eval/reports/runs" / f"{NAME}-{tag}-{dataset}"
        labels = [r for p in directory.glob("*/labels.jsonl") for r in rows(p)]
        actual = [r["id"] for r in labels]
        expected = meta["smoke_qids"] if phase == "smoke" else meta["qids"]
        errors = [r for r in labels if r.get("label", "").endswith("ERROR")]
        write(
            OUT / f"audit-{tag}-{dataset}.json",
            {"expected": len(expected), "actual": len(actual), "errors": errors},
        )
        if (
            len(actual) != len(set(actual))
            or set(actual) != set(expected)
            or (phase == "smoke" and errors)
        ):
            raise RuntimeError(
                f"{dataset}: question coverage or answer/judge errors; inspect audit"
            )
    return code


def execute(manifest: dict, phase: str) -> int:
    status = json.loads((OUT / "status.json").read_text())
    if phase == "subset" and not all(
        latest_run(status, "smoke", ds).get("state") == "complete" for ds in DATASETS
    ):
        raise RuntimeError("All four real smoke checks must succeed before subset run")
    status.update(state="running", phase=phase, supervisor_pid=os.getpid(), started_at=now())
    write(OUT / "status.json", status)
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(SNAPSHOT), str(SNAPSHOT / "src")]),
        "PYTHONUNBUFFERED": "1",
        "TIANXIMEM_BENCHMARK_DIR": str(
            Path(os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset")).resolve()
        ),
        "TIANXIMEM_CONFIG_DIR": str(CONFIGS),
        "TIANXIMEM_PROFILE": "local",
        "TIANXIMEM_SQLITE_PATH": str(VAR / "tianxi.db"),
        "TIANXIMEM_EMBED_CACHE_DIR": str(VAR / "embed_cache"),
        "TIANXIMEM_CAPTURE_PATH": str(VAR / "capture.jsonl"),
    }
    for dataset in DATASETS:
        if latest_run(status, phase, dataset).get("state") == "complete":
            continue
        check_frozen(manifest)
        tag = artifact_tag(manifest, phase, dataset)
        entry = {
            "phase": phase,
            "dataset": dataset,
            "artifact_tag": tag,
            "run_id": f"{NAME}-{tag}-{dataset}",
            "state": "starting",
            "started_at": now(),
        }
        status["runs"].append(entry)
        write(OUT / "status.json", status)
        service_env = env | {"TIANXIMEM_METRICS_PATH": str(VAR / f"metrics-{tag}-{dataset}.json")}
        service = None
        start = time.monotonic()
        try:
            with (VAR / f"service-{tag}-{dataset}.log").open("a") as log:
                service = subprocess.Popen(
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
                        str(PORT),
                    ],
                    cwd=VAR,
                    env=service_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            entry["service_pid"] = service.pid
            for _ in range(90):
                if service.poll() is not None:
                    raise RuntimeError("Owned service failed at startup; inspect service log")
                try:
                    httpx.get(
                        f"http://127.0.0.1:{PORT}/health", timeout=2, trust_env=False
                    ).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            else:
                raise RuntimeError("Owned service failed readiness check")
            entry["state"] = "running"
            with (VAR / f"run-{tag}-{dataset}.log").open("a") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--child", phase, dataset],
                    cwd=VAR,
                    env=service_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                entry["runner_pid"] = process.pid
                write(OUT / "status.json", status)
                print(
                    f"RUN {phase} {dataset}: service={service.pid} runner={process.pid}", flush=True
                )
                code = process.wait()
            entry.update(exit_code=code, state="complete" if code == 0 else "failed")
        except Exception as error:
            entry.update(state="failed", error=str(error))
        finally:
            if service and service.poll() is None:
                service.terminate()
                try:
                    service.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    service.kill()
                    service.wait()
            entry.update(finished_at=now(), seconds=round(time.monotonic() - start, 3))
            write(OUT / "status.json", status)
        print(f"DONE {phase} {dataset}: {entry['state']} ({entry['seconds']}s)", flush=True)
        if entry["state"] != "complete" and phase == "smoke":
            status.update(state="smoke_failed", finished_at=now())
            write(OUT / "status.json", status)
            return 1
    with (
        sqlite3.connect(VAR / "tianxi.db") as source,
        sqlite3.connect(VAR / f"snapshot-{phase}.db") as target,
    ):
        source.backup(target)
    failed = any(latest_run(status, phase, ds).get("state") != "complete" for ds in DATASETS)
    status.update(state="failed" if failed else f"{phase}_complete", finished_at=now())
    write(OUT / "status.json", status)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--phase", choices=("smoke", "subset"), default="smoke")
    parser.add_argument("--child", nargs=2)
    args = parser.parse_args()
    if args.child:
        return child(*args.child)
    manifest = prepare()
    return 0 if args.prepare_only else execute(manifest, args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
