"""Two independent endpoint workers, with native datasets still run serially.

Only scheduling changes: each worker invokes the unchanged native run_round and
pipeline. User directories are disjoint, completed rows are preserved, and the
parent aggregates JudgeResult objects in original sample order. Run IDs, source,
questions, prompts, scoring and generation settings remain unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import queue
import time
import traceback
from pathlib import Path

import experiment
from experiment import NAME, OUT, check_arm, now, rows, write

SECONDARY = "https://memory3.021130.xyz/v1"
TRANSPORT = OUT / "dual-endpoint-a13.json"


def complete_results(items, out_dir):
    """Read exact completed checkpoints; never judge them a second time."""
    from eval.harness.judge import JudgeResult

    def keyed(values):
        mapping = {str(x["id"]): x for x in values}
        if len(mapping) != len(values):
            raise RuntimeError("Duplicate native checkpoint IDs")
        return mapping

    incoming = keyed(items)
    inputs = keyed(rows(out_dir / "input.jsonl"))
    answers = keyed(rows(out_dir / "answers.jsonl"))
    labels = keyed(rows(out_dir / "labels.jsonl"))
    if inputs and inputs != incoming:
        raise RuntimeError(f"Existing native scoring input changed: {out_dir.name}")
    if not (incoming.keys() == answers.keys() == labels.keys()):
        return None
    return [
        JudgeResult(
            qid=str(item["id"]),
            is_correct=labels[str(item["id"])]["is_correct"],
            label=labels[str(item["id"])]["label"],
            judge_response=labels[str(item["id"])]["judge_response"],
            generated_answer=answers[str(item["id"])].get("generated_answer", ""),
            partial=labels[str(item["id"])].get("partial"),
            metrics=labels[str(item["id"])].get("metrics"),
        )
        for item in items
    ]


def preserve_partial(native_judge, pipeline, items, out_dir, **options):
    """Judge only missing labels; preserve already committed rows exactly."""
    from eval.harness import judge

    old_labels = rows(out_dir / "labels.jsonl")
    if not old_labels:
        return native_judge(pipeline, items, out_dir, **options), items
    done = {str(row["id"]) for row in old_labels}
    pending = [item for item in items if str(item["id"]) not in done]
    pending_ids = {str(item["id"]) for item in pending}
    digest = hashlib.sha256(json.dumps(sorted(pending_ids)).encode()).hexdigest()[:16]
    temporary = out_dir / f".dual-pending-{digest}"
    temporary.mkdir(parents=True, exist_ok=True)
    checkpoint = temporary / "answers.jsonl"
    if not checkpoint.exists():
        checkpoint.write_text(
            "".join(
                judge._jsonl_line(row)
                for row in rows(out_dir / "answers.jsonl")
                if str(row["id"]) in pending_ids
            )
        )
    native_judge(pipeline, pending, temporary, **options)
    for kind in ("answers", "labels"):
        combined = {}
        for directory in (out_dir, temporary):
            path = directory / f"{kind}.jsonl"
            for line in path.read_text().split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = str(row["id"])
                if qid in combined and json.loads(combined[qid]) != row:
                    raise RuntimeError(f"Completed native {kind} row changed: {qid}")
                combined[qid] = combined.get(qid, line)
        if set(combined) != {str(item["id"]) for item in items}:
            raise RuntimeError(f"Partial native {kind} merge has incomplete coverage")
        path = out_dir / f"{kind}.jsonl"
        staged = path.with_suffix(".jsonl.tmp")
        staged.write_text("".join(combined[str(item["id"])] + "\n" for item in items))
        staged.replace(path)
    result = complete_results(items, out_dir)
    if result is None:
        raise RuntimeError("Merged native checkpoint is incomplete")
    return result, pending


def worker(slot, tasks, results, kwargs, original_round, dataset, samples):
    from eval.experiments import run
    from eval.harness import judge

    if slot == 1:
        key = os.environ["AML_EMB_API_KEY"]
        os.environ.update(
            AML_BASE_URL=SECONDARY,
            AML_API_KEY=key,
            AML_JUDGE_BASE_URL=SECONDARY,
            AML_JUDGE_API_KEY=key,
        )
    else:
        # Let an interrupted primary request finish at its server before the
        # first resumed primary request. The secondary can start immediately.
        time.sleep(30)
    endpoint = os.environ["AML_BASE_URL"]
    native_judge = run.run_judge
    logfile = OUT / f"transport-a13-full-{dataset}-{slot}.jsonl"

    def checkpoint_judge(pipeline, items, out_dir, **options):
        out_dir = Path(out_dir)
        # Native sanitation retains completed rows and repairs a killed final
        # answer line using the same routine used by the native harness.
        judge._sanitize_jsonl(out_dir / "answers.jsonl")
        judge._sanitize_jsonl(out_dir / "labels.jsonl")
        existing = complete_results(items, out_dir)
        if existing is not None:
            return existing
        before = {str(x["id"]) for x in rows(out_dir / "answers.jsonl")}
        started = now()
        result, judged_items = preserve_partial(native_judge, pipeline, items, out_dir, **options)
        with logfile.open("a") as handle:
            handle.write(
                json.dumps(
                    dict(
                        user_id=out_dir.name,
                        qids=[str(x["id"]) for x in judged_items],
                        newly_answered_qids=[
                            str(x["id"]) for x in judged_items if str(x["id"]) not in before
                        ],
                        answer_base=endpoint,
                        judge_base=os.environ.get("AML_JUDGE_BASE_URL") or endpoint,
                        model=os.environ["AML_MODEL"],
                        started_at=started,
                        finished_at=now(),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
        return result

    run.run_judge = checkpoint_judge
    try:
        while True:
            task = tasks.get()
            if task is None:
                return
            index = task
            sample = samples[index]
            run._load = lambda *args, _sample=sample, **options: [_sample]
            try:
                _, native_results = original_round(**kwargs)
            except Exception:
                results.put(("error", index, traceback.format_exc(), slot))
            else:
                results.put(("ok", index, native_results, slot))
    except Exception:
        results.put(("error", -1, traceback.format_exc(), slot))


def parallel_round(original_round, **kwargs):
    from eval.experiments import run

    dataset = kwargs["dataset"]
    samples = run._load(
        dataset, kwargs["bench_dir"], kwargs.get("limit"), spread=kwargs.get("spread", False)
    )
    if len({sample.user_id for sample in samples}) != len(samples):
        raise RuntimeError("Parallel workers require disjoint native user directories")
    if not kwargs.get("skip_ingest") or kwargs.get("input_contract", "native") != "native":
        raise RuntimeError("Parallel worker requires the frozen, read-only native experiment")
    context = multiprocessing.get_context("fork")
    tasks, results = context.Queue(), context.Queue()
    # Fork already inherits immutable samples; queue only indices, avoiding
    # copies of large corpora and a feeder blocked on unread payloads.
    for index in range(len(samples)):
        tasks.put(index)
    for _ in range(2):
        tasks.put(None)
    workers = [
        context.Process(
            target=worker,
            args=(slot, tasks, results, kwargs, original_round, dataset, samples),
        )
        for slot in range(2)
    ]
    collected = {}
    failures = []
    try:
        for process in workers:
            process.start()
        while len(collected) < len(samples):
            try:
                kind, index, payload, slot = results.get(timeout=5)
            except queue.Empty:
                if not any(process.is_alive() for process in workers):
                    raise RuntimeError(
                        "Endpoint workers exited before completing native samples"
                    ) from None
                continue
            if kind == "error":
                if index < 0:
                    raise RuntimeError(f"Endpoint worker {slot} failed:\n{payload}")
                failures.append(
                    dict(index=index, user_id=samples[index].user_id, slot=slot,
                         error=payload, failed_at=now())
                )
                write(OUT / f"execution-failures-a13-full-{dataset}.json", failures)
            if index in collected:
                raise RuntimeError(f"Duplicate native sample result: {index}")
            collected[index] = payload if kind == "ok" else None
            print(
                f"  dual {dataset}: {len(collected)}/{len(samples)} samples attempted; "
                f"{len(failures)} execution failures",
                flush=True,
            )
        for process in workers:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError(f"Endpoint worker did not exit cleanly: {process.exitcode}")
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        # A failed worker leaves queued tasks unread. Never wait for a feeder
        # to flush to dead readers during interpreter shutdown.
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
    if failures:
        raise RuntimeError(
            f"{len(failures)} native samples failed; completed checkpoints preserved; "
            f"see execution-failures-a13-full-{dataset}.json"
        )
    ordered = [result for index in range(len(samples)) for result in collected[index]]
    expected = {question.qid for sample in samples for question in sample.questions}
    if len(ordered) != len(expected) or {result.qid for result in ordered} != expected:
        raise RuntimeError("Parallel aggregation changed the frozen native question set")
    return samples, ordered


def child(label, phase, dataset, port):
    from eval.experiments import run

    original_round = run.run_round
    run.run_round = lambda **kwargs: parallel_round(original_round, **kwargs)
    # RecordingClient is installed inside experiment.child. Serialize its
    # append-only Search log while letting independent model calls overlap.
    native_main = run.main

    def locked_main(args):
        # experiment.child installs its RecordingClient before calling main.
        original_search = run.ServiceClient.search_raw

        def locked_search(client, **kwargs):
            with (OUT / f"search-{label}-{phase}-{dataset}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return original_search(client, **kwargs)

        run.ServiceClient.search_raw = locked_search
        return native_main(args)

    run.main = locked_main
    result = experiment.child(label, phase, dataset, port)
    if not result:
        record = OUT.parent / f"{NAME}-{label}-{phase}-{dataset}.json"
        value = json.loads(record.read_text())
        value["execution_backends"] = json.loads(TRANSPORT.read_text())["backends"]
        value["execution_transport"] = {
            "scheduling": "datasets serial; one native sample worker per independent endpoint",
            "routing_logs": [
                str(OUT / f"transport-a13-full-{dataset}-{slot}.jsonl") for slot in range(2)
            ],
            "preexisting_rows": str(TRANSPORT),
        }
        write(record, value)
    return result


def resume():
    manifest = json.loads((OUT / "arm-a13.json").read_text())
    check_arm(manifest)
    if not (OUT / "gate-a13.json").exists():
        raise RuntimeError("Associated candidate gates have not passed")
    if os.environ["AML_MODEL"] != "Qwen/Qwen3.5-9B":
        raise RuntimeError("Frozen logical model differs from the new deployment")
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(
            filter(None, (os.environ.get(name), "127.0.0.1", "localhost", "::1"))
        )
    native_run = experiment.subprocess.run

    def dispatch(args, *positional, **options):
        if isinstance(args, list) and "--child" in args and Path(args[1]).name == "experiment.py":
            args = [args[0], str(Path(__file__).resolve()), *args[2:]]
        return native_run(args, *positional, **options)

    experiment.subprocess.run = dispatch
    result = experiment.run_phase("a13", "full", experiment.SETS)
    if result:
        return result
    from final_audit import audit

    audit("a13")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=4)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if options.child:
        raise SystemExit(child(*options.child[:3], int(options.child[3])))
    if options.resume:
        raise SystemExit(resume())
    parser.error("Choose --resume or --child")
