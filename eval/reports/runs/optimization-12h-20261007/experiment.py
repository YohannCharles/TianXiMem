"""Frozen native HTTP experiments with isolated truth copies and serial judging.

The service reads an immutable product source snapshot. Python loads the native
eval package from the repository; its code must match the captured snapshot byte
for byte. Existing vectors can be shared because qa_pairs are exact copies and
this experiment never calls Add. Derived facts are rebuilt in private SQLite.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[4]
NAME = "optimization-12h-20261007"
OUT = ROOT / "eval/reports/runs" / NAME
VAR = ROOT / "var" / NAME
SETS = (
    "corporatebench",
    "mquake-remastered",
    "tempreason",
    "memtrapbench",
    "locomo-refined",
    "longmemeval-s",
    "medmemorybench",
    "halumem",
    "musique",
    "hybridqa",
    "feverous",
)
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(UTC).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staged.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def truth_hash(db):
    h = hashlib.sha256()
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        for table in ("qa_pairs", "applied_batches"):
            for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):
                h.update(json.dumps(row, ensure_ascii=False).encode() + b"\n")
    return h.hexdigest()


def select(samples, count):
    """Round-robin categories, using a fixed qid hash before any arm results."""
    buckets = defaultdict(list)
    for sample in samples:
        for question in sample.questions:
            buckets[question.category].append((sample, question))
    for bucket in buckets.values():
        bucket.sort(key=lambda pair: hashlib.sha256(pair[1].qid.encode()).hexdigest())
    picked = set()
    while len(picked) < count and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(picked) < count:
                picked.add(buckets[category].pop(0)[1].qid)
    return [
        replace(s, questions=tuple(q for q in s.questions if q.qid in picked))
        for s in samples
        if any(q.qid in picked for q in s.questions)
    ]


def prepare():
    from eval.datasets import benchmark_dir
    from eval.experiments.recipes import recipe_for
    from eval.experiments.run import _load

    path = OUT / "plans.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        for file, digest in manifest["hashes"].items():
            if sha(Path(file)) != digest:
                raise RuntimeError(f"Frozen plan changed: {file}")
        return manifest
    VAR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for dataset in SETS:
        recipe = recipe_for(dataset)
        full = _load(dataset, benchmark_dir(), recipe.limit, spread=recipe.spread)
        qids = [q.qid for s in full for q in s.questions]
        if len(qids) != recipe.n_questions or len(set(qids)) != len(qids):
            raise RuntimeError(f"Frozen qids drifted: {dataset}")
        plan = {"small": select(full, 6), "stage": select(full, 24), "full": full}
        file = VAR / f"plan-{dataset}.pickle"
        with file.open("wb") as handle:
            pickle.dump(plan, handle)
        group = "seven" if dataset in SETS[:7] else "four"
        previous = (
            f"facts-benchmark-20261005-on-{dataset}"
            if group == "seven"
            else f"new-datasets-benchmark-20261006-subset-{dataset}"
        )
        prior_ids = [
            str(r["id"])
            for p in (ROOT / "eval/reports/runs" / previous).glob("*/input.jsonl")
            for r in rows(p)
        ]
        if set(prior_ids) != set(qids):
            raise RuntimeError(f"Prior frozen input qids differ: {dataset}")
        metadata.append(
            {
                "dataset": dataset,
                "group": group,
                "prior_run": previous,
                "qids": qids,
                "n": len(qids),
                "small_qids": [q.qid for s in plan["small"] for q in s.questions],
                "stage_qids": [q.qid for s in plan["stage"] for q in s.questions],
            }
        )
    if sum(d["n"] for d in metadata) != 3001:
        raise RuntimeError("Full selection must contain exactly 3,001 questions")
    manifest = {
        "created_at": now(),
        "selection": "category round-robin, sha256(qid)",
        "datasets": metadata,
        "hashes": {str(p): sha(p) for p in VAR.glob("plan-*.pickle")},
    }
    write(path, manifest)
    return manifest


def freeze_arm(label, source, truth_parent=None):
    manifest_path = OUT / f"arm-{label}.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    arm = VAR / label
    snapshot = arm / "source"
    paths = [
        p
        for directory in ("src", "eval/datasets", "eval/harness", "eval/experiments")
        for p in (source / directory).rglob("*")
        if p.suffix in {".py", ".sql"}
    ]
    paths += [source / "eval/reports/schema.py"]
    for path in paths:
        destination = snapshot / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    metadata = {}
    parent = json.loads((OUT / f"arm-{truth_parent}.json").read_text()) if truth_parent else None
    if parent:
        check_arm(parent)
        for relative in (
            "src/tianximem/facts/evidence.py",
            "src/tianximem/facts/grammar.py",
            "src/tianximem/facts/conversation.py",
        ):
            if sha(source / relative) != sha(Path(parent["snapshot"]) / relative):
                raise RuntimeError("Cannot reuse derived facts with a different extractor")
    for group, previous in (
        ("seven", "facts-benchmark-20261005"),
        ("four", "new-datasets-benchmark-20261006"),
    ):
        cfg = arm / group / "configs"
        cfg.mkdir(parents=True)
        default = yaml.safe_load((source / "configs/default.yaml").read_text())
        local = yaml.safe_load((source / "configs/local.yaml").read_text())
        local["storage"]["qdrant"]["collection"] = (
            "memories_facts_benchmark_20261005"
            if group == "seven"
            else "memories_new_datasets_20261006"
        )
        (cfg / "default.yaml").write_text(
            yaml.safe_dump(default, sort_keys=False, allow_unicode=True)
        )
        (cfg / "local.yaml").write_text(yaml.safe_dump(local, sort_keys=False, allow_unicode=True))
        original = (
            Path(parent["groups"][group]["db"]) if parent else ROOT / "var" / previous / "tianxi.db"
        )
        db = arm / group / "tianxi.db"
        with (
            sqlite3.connect(original.resolve().as_uri() + "?mode=ro", uri=True) as src,
            sqlite3.connect(db) as dst,
        ):
            src.backup(dst)
        metadata[group] = {
            "db": str(db),
            "truth_sha256": truth_hash(db),
            "configs": str(cfg),
            "cache": str(ROOT / "var" / previous / "embed_cache"),
        }
        if parent and metadata[group]["truth_sha256"] != parent["groups"][group]["truth_sha256"]:
            raise RuntimeError("Reused truth clone differs from its frozen parent")
    frozen = [*snapshot.rglob("*.py"), *snapshot.rglob("*.sql"), *arm.glob("*/configs/*.yaml")]
    manifest = {
        "label": label,
        "created_at": now(),
        "snapshot": str(snapshot),
        "groups": metadata,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "truth_parent": truth_parent,
        "hashes": {str(p): sha(p) for p in frozen},
    }
    write(manifest_path, manifest)
    return manifest


def check_arm(manifest):
    snapshot = Path(manifest["snapshot"])
    for path, digest in manifest["hashes"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Frozen source/config drift: {path}")
        file = Path(path)
        if file.is_relative_to(snapshot):
            relative = file.relative_to(snapshot)
            if relative.parts[0] == "eval" and sha(ROOT / relative) != digest:
                raise RuntimeError(f"Live native harness differs from frozen source: {relative}")


def clone_arm(label, parent, overrides):
    """Single-variable configuration arm, retaining its parent's exact source."""
    path = OUT / f"arm-{label}.json"
    if path.exists():
        return json.loads(path.read_text())
    original = json.loads((OUT / f"arm-{parent}.json").read_text())
    check_arm(original)
    arm = VAR / label
    snapshot = arm / "source"
    shutil.copytree(original["snapshot"], snapshot)
    groups = {}
    for name, group in original["groups"].items():
        cfg = arm / name / "configs"
        shutil.copytree(group["configs"], cfg)
        local = yaml.safe_load((cfg / "local.yaml").read_text())
        for field, value in overrides.items():
            section, key = field.split(".", 1)
            local.setdefault(section, {})[key] = value
        (cfg / "local.yaml").write_text(yaml.safe_dump(local, sort_keys=False))
        db = arm / name / "tianxi.db"
        with (
            sqlite3.connect(Path(group["db"]).resolve().as_uri() + "?mode=ro", uri=True) as src,
            sqlite3.connect(db) as dst,
        ):
            src.backup(dst)
        groups[name] = group | {"db": str(db), "configs": str(cfg)}
        if truth_hash(db) != group["truth_sha256"]:
            raise RuntimeError("Cloned arm truth differs")
    files = [*snapshot.rglob("*.py"), *snapshot.rglob("*.sql"), *arm.glob("*/configs/*.yaml")]
    manifest = original | {
        "label": label,
        "created_at": now(),
        "snapshot": str(snapshot),
        "groups": groups,
        "parent": parent,
        "overrides": overrides,
        "env_overrides": {"TIANXIMEM_RERANKER_MODEL": "qwen3-reranker-4b"},
        "hashes": {str(p): sha(p) for p in files},
    }
    write(path, manifest)
    return manifest


def index_facts(label, group):
    from tianximem.facts.evidence import EVIDENCE_VERSION
    from tianximem.pairing.apply import index_pair_facts
    from tianximem.store.sqlite_store import SqliteStore

    manifest = json.loads((OUT / f"arm-{label}.json").read_text())
    store = SqliteStore.open(manifest["groups"][group]["db"])
    with store.read() as conn:
        users = sorted({p.user_id for p in store.iter_pairs(conn)})
    n = 0
    for user in users:
        while True:
            with store.transaction() as conn:
                pairs = store.fetch_unindexed_pairs(conn, user, version=EVIDENCE_VERSION, limit=512)
                for pair in pairs:
                    index_pair_facts(store, conn, pair)
                n += len(pairs)
            if not pairs:
                break
    if (
        truth_hash(Path(manifest["groups"][group]["db"]))
        != manifest["groups"][group]["truth_sha256"]
    ):
        raise RuntimeError("Derived indexing changed truth")
    print(f"Indexed {label}/{group}: {n} sources, version={EVIDENCE_VERSION}", flush=True)


def child(label, phase, dataset, port):
    manifest = json.loads((OUT / f"arm-{label}.json").read_text())
    check_arm(manifest)
    sys.path[:0] = [manifest["snapshot"], str(Path(manifest["snapshot"]) / "src")]
    from eval.experiments import run
    from eval.harness import ServiceClient

    plans = prepare()
    meta = next(d for d in plans["datasets"] if d["dataset"] == dataset)
    group = manifest["groups"][meta["group"]]
    with (VAR / f"plan-{dataset}.pickle").open("rb") as handle:
        saved = pickle.load(handle)
    if phase in saved:
        samples = saved[phase]
    else:
        target = json.loads((OUT / f"targets-{phase}.json").read_text())
        if target["plan_hashes"] != plans["hashes"]:
            raise RuntimeError("Targeted selection no longer matches frozen native inputs")
        qids = set(target["qids"][dataset])
        samples = [
            replace(s, questions=tuple(q for q in s.questions if q.qid in qids))
            for s in saved["full"]
            if any(q.qid in qids for q in s.questions)
        ]
        if {q.qid for s in samples for q in s.questions} != qids:
            raise RuntimeError("Targeted native qids differ from the frozen selection")
    run._load = lambda *args, **kwargs: samples
    log = OUT / f"search-{label}-{phase}-{dataset}.jsonl"
    cached = {(r["user_id"], r["query"]): r["response"] for r in rows(log)}

    class RecordingClient(ServiceClient):
        def ingest(self, sample):
            raise RuntimeError("This experiment uses verified truth clones, never Add")

        def search_raw(self, *, user_id, query, top_k):
            if (user_id, query) in cached:
                return cached[user_id, query]
            start = time.monotonic()
            response = super().search_raw(user_id=user_id, query=query, top_k=top_k)
            with log.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "query": query,
                            "top_k": top_k,
                            "response": response,
                            "seconds": time.monotonic() - start,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return response

    run.ServiceClient = RecordingClient
    rid = f"{NAME}-{label}-{phase}-{dataset}"
    return run.main(
        [
            "--dataset",
            dataset,
            "--frozen",
            "--offline",
            "--skip-ingest",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--run-id",
            rid,
            "--configs-dir",
            group["configs"],
            "--profile",
            "local",
            "--reports-dir",
            str(ROOT / "eval/reports"),
            "--step",
            "native-12h-optimization",
            "--embedder",
            "qwen3-embedding-8b",
            "--reranker",
            manifest.get("env_overrides", {}).get(
                "TIANXIMEM_RERANKER_MODEL", "disabled (local profile)"
            ),
            "--metrics",
            str(VAR / label / meta["group"] / "metrics.json"),
            "--notes",
            f"{phase}; immutable product source snapshot; byte-checked native harness; "
            "native frozen qids and unchanged pipeline; exact truth clone "
            "and existing matching vector index; phase subset is not a full benchmark score",
        ]
    )


def run_phase(label, phase, datasets):
    rejected = OUT / f"rejected-{label}.json"
    if rejected.exists():
        reason = json.loads(rejected.read_text()).get("reason", "failed candidate gate")
        raise RuntimeError(f"Candidate {label} is quarantined: {reason}")
    # One answer/judge chain for the entire experiment, including other arms.
    with (VAR / "serial.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another native experiment chain is already running") from error
        return _run_phase(label, phase, datasets)


def _run_phase(label, phase, datasets):
    manifest = json.loads((OUT / f"arm-{label}.json").read_text())
    check_arm(manifest)
    plans = prepare()
    env = os.environ.copy()
    env.update(manifest.get("env_overrides", {}))
    env["PYTHONPATH"] = str(Path(manifest["snapshot"]) / "src")
    status_path = OUT / f"status-{label}-{phase}.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {"started_at": now(), "runs": []}
    )
    services = {}
    handles = []
    ports = {"seven": 8070, "four": 8071}
    try:
        for name in {
            next(d["group"] for d in plans["datasets"] if d["dataset"] == ds) for ds in datasets
        }:
            group = manifest["groups"][name]
            service_env = env | {
                "TIANXIMEM_SQLITE_PATH": group["db"],
                "TIANXIMEM_EMBED_CACHE_DIR": group["cache"],
                "TIANXIMEM_CONFIG_DIR": group["configs"],
                "TIANXIMEM_PROFILE": "local",
                "TIANXIMEM_ENV_FILE": str(ROOT / ".env"),
                "TIANXIMEM_METRICS_PATH": str(VAR / label / name / "metrics.json"),
            }
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--index", label, name],
                env=service_env,
                check=True,
            )
            handle = (VAR / label / name / "service.log").open("a")
            handles.append(handle)
            process = subprocess.Popen(
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
                    str(ports[name]),
                ],
                cwd=ROOT,
                env=service_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            services[name] = process
            ready = False
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(f"Service exited: {name}; see log")
                try:
                    httpx.get(
                        f"http://127.0.0.1:{ports[name]}/health", timeout=2
                    ).raise_for_status()
                    ready = True
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            if not ready:
                raise RuntimeError(f"Service not ready: {name}")
        for dataset in datasets:
            if (OUT / f"rejected-{label}.json").exists():
                raise RuntimeError(f"Candidate {label} was rejected during this phase")
            check_arm(manifest)
            meta = next(d for d in plans["datasets"] if d["dataset"] == dataset)
            entry = next((r for r in status["runs"] if r["dataset"] == dataset), None)
            if entry and entry.get("state") == "complete":
                continue
            if entry is None:
                entry = {"dataset": dataset}
                status["runs"].append(entry)
            entry.update(state="running", started_at=now())
            status["state"] = "running"
            write(status_path, status)
            print(f"RUN {label}/{phase}/{dataset}", flush=True)
            with (VAR / label / f"{phase}-{dataset}.log").open("a") as handle:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        label,
                        phase,
                        dataset,
                        str(ports[meta["group"]]),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            entry.update(
                state="complete" if result.returncode == 0 else "failed",
                finished_at=now(),
                exit_code=result.returncode,
            )
            record = ROOT / "eval/reports/runs" / f"{NAME}-{label}-{phase}-{dataset}.json"
            if entry["state"] == "complete" and record.exists():
                entry["scores"] = json.loads(record.read_text())["scores"]
            write(status_path, status)
            print(f"DONE {dataset}: {entry['state']}", flush=True)
            if result.returncode:
                return result.returncode
        status.update(state="finished", finished_at=now())
        write(status_path, status)
        return 0
    finally:
        for process in services.values():
            process.terminate()
        for process in services.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--freeze-arm", nargs=2, metavar=("LABEL", "SOURCE"))
    parser.add_argument("--truth-parent", help="Reuse an exact truth and unchanged-extractor copy")
    parser.add_argument("--clone-arm", nargs=3, metavar=("LABEL", "PARENT", "OVERRIDES_JSON"))
    parser.add_argument("--index", nargs=2)
    parser.add_argument("--child", nargs=4)
    parser.add_argument("--run", nargs=2, metavar=("LABEL", "PHASE"))
    parser.add_argument("--datasets", nargs="+", choices=SETS)
    args = parser.parse_args()
    if args.child:
        return child(*args.child)
    if args.index:
        index_facts(*args.index)
        return 0
    if args.freeze_arm:
        freeze_arm(args.freeze_arm[0], Path(args.freeze_arm[1]).resolve(), args.truth_parent)
        return 0
    if args.clone_arm:
        clone_arm(args.clone_arm[0], args.clone_arm[1], json.loads(args.clone_arm[2]))
        return 0
    if args.run:
        return run_phase(*args.run, args.datasets or SETS)
    if args.prepare:
        prepare()
        return 0
    parser.error("Choose an operation")


if __name__ == "__main__":
    raise SystemExit(main())
