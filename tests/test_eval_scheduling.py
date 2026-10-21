"""Concurrent users retain payloads/history, bounded queues and failure barriers."""

from __future__ import annotations

import json
from collections import defaultdict
from threading import Barrier, Event, Lock

import httpx
import pytest
from eval.datasets import Message, Question, Sample, Session
from eval.experiments import run as runner

#: run_round 与它调用的协作者住在 runner.py。
#: ⚠ 补丁必须打在「调用点解析的那个模块」上：打在 runner（= run.py）上的是名字快照，
#: 而真调用发生在 runner.py 里 ⇒ 补丁静默失效（表现是真去加载数据、真发请求，不是报错）。
from eval.experiments import runner as runner_impl
from eval.harness import ServiceClient
from eval.harness.judge import JudgeResult
from eval.harness.scheduling import run_pipeline, run_user_tasks, worker_counts


def test_independent_rerank_user_pools_reach_16_and_32_without_borrowing_slots():
    from eval.harness.scheduling import run_grouped_user_tasks
    limits = {"local": 16, "memory3": 32}
    barrier, lock = Barrier(48), Lock()
    active = dict.fromkeys(limits, 0)
    peak = dict(active)
    events = defaultdict(list)
    users = [(group, i) for group, count in limits.items() for i in range(2 * count)]

    def run(_index, user):
        group, index = user
        with lock:
            active[group] += 1
            peak[group] = max(peak[group], active[group])
        try:
            if index < limits[group]:
                barrier.wait(timeout=10)
            for event in ("add", "search", "add", "search"):
                events[user].append(event)
        finally:
            with lock:
                active[group] -= 1

    completed = list(run_grouped_user_tasks(
        users, user_id=str, run=run, group_of=lambda x: x[0],
        group_workers=limits, cancel=lambda: None,
    ))
    assert sorted(completed) == list(range(len(users)))
    assert peak == limits
    assert all(events[user] == ["add", "search", "add", "search"] for user in users)


def test_group_failure_cancels_and_joins_other_group():
    from eval.harness.scheduling import run_grouped_user_tasks
    stop, joined = Event(), Event()
    barrier = Barrier(2)
    def run(_index, group):
        barrier.wait(timeout=5)
        if group == "bad":
            raise RuntimeError("failed")
        assert stop.wait(timeout=5)
        joined.set()
    with pytest.raises(RuntimeError, match="failed"):
        list(run_grouped_user_tasks(["bad", "slow"], user_id=str, run=run,
             group_of=str, group_workers={"bad": 1, "slow": 1}, cancel=stop.set))
    assert joined.is_set()


def test_user_tasks_cancel_before_joining_other_replays():
    stop = Event()
    joined = Event()
    barrier = Barrier(2)

    def run(_index, user):
        barrier.wait(timeout=5)
        if user == "bad":
            raise RuntimeError("fatal HTTP failure")
        assert stop.wait(timeout=5), "cancellation was not signalled before join"
        joined.set()

    with pytest.raises(RuntimeError, match="fatal HTTP"):
        list(run_user_tasks(["bad", "other"], user_id=lambda x: x, run=run,
                            workers=2, cancel=stop.set))
    assert joined.is_set()


def test_duplicate_replay_users_are_rejected_before_http():
    calls = []
    with pytest.raises(ValueError, match="unique user_id"):
        list(run_user_tasks(["same", "same"], user_id=lambda x: x,
                            run=lambda *args: calls.append(args), workers=2, cancel=lambda: None))
    assert not calls


@pytest.mark.parametrize('workers', [6, 16])
def test_user_tasks_can_run_simultaneously(workers):
    barrier = Barrier(workers)
    lock = Lock()
    active = maximum = 0

    def run(_index, _user):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=5)
        with lock:
            active -= 1

    done = list(run_user_tasks(
        list(range(workers)), user_id=str, run=run, workers=workers, cancel=lambda: None
    ))
    assert maximum == workers and sorted(done) == list(range(workers))


def test_parallel_users_preserve_batches_search_barrier_and_result_order(tmp_path, monkeypatch):
    samples = [
        Sample(
            user_id=f"user-{i}",
            dataset="longmemeval-s",
            sessions=tuple(
                Session(
                    f"s-{j}",
                    tuple(
                        Message("user" if k % 2 == 0 else "assistant", f"user-{i}/s-{j}/m-{k}")
                        for k in range(25)
                    ),
                )
                for j in range(2)
            ),
            questions=(Question(f"q-{i}", "What?", "gold", "single-session-user"),),
        )
        for i in range(2)
    ]
    monkeypatch.setattr(runner_impl, "_load", lambda *a, **kw: samples)
    monkeypatch.setattr(runner_impl, "pipeline_for", lambda *a: tmp_path / "pipeline.py")
    monkeypatch.setattr(runner_impl, "ENDPOINTS", (("e0", "k0"), ("e1", "k1")))
    monkeypatch.setattr(
        runner_impl,
        "run_judge",
        lambda pipeline, items, out, **kw: [
            JudgeResult(item["id"], True, "CORRECT", "ok", "answer") for item in items
        ],
    )
    barrier = Barrier(2)

    def replay(workers):
        requests = defaultdict(list)

        def handler(request):
            body = json.loads(request.content)
            uid = body["user_id"]
            requests[uid].append((request.url.path, body))
            if request.url.path == "/add":
                if workers == 2 and len(requests[uid]) == 1:
                    barrier.wait(timeout=5)  # Both users must really overlap.
                return httpx.Response(
                    200,
                    json={
                        "success": True,
                        **{k: body[k] for k in ("request_id", "user_id", "session_id")},
                    },
                )
            assert sum(path == "/add" for path, _ in requests[uid]) == 4
            return httpx.Response(200, json={"data": []})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = ServiceClient("http://stub", client=http)
            _, results = runner.run_round(
                dataset="longmemeval-s",
                base_url="http://stub",
                bench_dir=tmp_path,
                out_dir=tmp_path / str(workers),
                client=client,
                ingest_workers=workers,
                judge_workers=2,
            )
        return dict(requests), results

    serial_requests, serial = replay(1)
    parallel_requests, parallel = replay(2)
    assert serial_requests == parallel_requests  # Exact IDs, shape, batches, order, queries.
    assert serial == parallel
    assert [row.qid for row in parallel] == ["q-0", "q-1"]
    timings = json.loads((tmp_path / "2/user-0/timings.json").read_text())
    assert timings["batches"] == 4 and timings["ingest_workers"] == 2


def test_fast_user_can_finish_while_previous_user_is_still_preparing():
    release = Event()
    first_started = Event()

    def prepare(index, value):
        if index == 0:
            first_started.set()
            assert release.wait(timeout=5)
        else:
            assert first_started.wait(timeout=5)
        return value

    def judge(index, value, prepared):
        if index == 1:
            release.set()
        return prepared

    got = dict(
        run_pipeline(
            ["slow", "fast"],
            user_id=str,
            prepare=prepare,
            judge=judge,
            ingest_workers=2,
            judge_workers=1,
        )
    )
    assert got == {0: "slow", 1: "fast"}


def test_pipeline_bounds_prepared_contexts_while_judges_are_busy():
    lock = Lock()
    prepared = set()
    judged = set()
    largest = 0

    def prepare(index, value):
        nonlocal largest
        with lock:
            prepared.add(index)
            largest = max(largest, len(prepared - judged))
        return value

    def judge(index, value, context):
        with lock:
            judged.add(index)
        return context

    assert (
        len(
            dict(
                run_pipeline(
                    list(range(60)),
                    user_id=str,
                    prepare=prepare,
                    judge=judge,
                    ingest_workers=3,
                    judge_workers=2,
                )
            )
        )
        == 60
    )
    assert largest <= 5


def test_duplicate_users_and_invalid_workers_fail_before_preparing():
    called = []
    with pytest.raises(ValueError, match="unique user_id"):
        list(
            run_pipeline(
                [1, 2],
                user_id=lambda _: "same",
                prepare=lambda *a: called.append(a),
                judge=lambda *a: None,
                ingest_workers=2,
                judge_workers=1,
            )
        )
    assert not called
    with pytest.raises(ValueError, match="nonnegative"):
        worker_counts(-1, 2, users=2, endpoints=2)


def test_failed_preparation_is_never_sent_to_judge():
    judged = []

    def prepare(index, value):
        raise RuntimeError("Add failed")

    with pytest.raises(RuntimeError, match="Add failed"):
        list(
            run_pipeline(
                ["u"],
                user_id=str,
                prepare=prepare,
                judge=lambda *a: judged.append(a),
                ingest_workers=1,
                judge_workers=1,
            )
        )
    assert not judged


def test_parallel_aml_users_preserve_historical_search_events(tmp_path, monkeypatch):
    from eval.datasets.aml.plan import AddEvent, InputPlan, SearchEvent

    plans = [
        InputPlan(
            Sample(
                f"u-{i}",
                "halumem",
                (),
                (
                    Question(f"early-{i}", "old?", {"answer": "gold"}, "fact"),
                    Question(f"late-{i}", "new?", {"answer": "gold"}, "fact"),
                ),
            ),
            (
                AddEvent("s", ({"role": "user", "content": "user: PAST"},)),
                SearchEvent(f"early-{i}", "old?"),
                AddEvent("s", ({"role": "user", "content": "user: FUTURE"},)),
                SearchEvent(f"late-{i}", "new?"),
            ),
        )
        for i in range(2)
    ]
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("# stub")
    monkeypatch.setattr(runner_impl, "load_plans", lambda *a, **kw: plans)
    monkeypatch.setattr(runner_impl, "pipeline_for", lambda *a: pipeline)
    seen = []

    def judge(path, items, out, **kw):
        seen.extend(items)
        return []

    monkeypatch.setattr(runner_impl, "run_judge", judge)
    barrier = Barrier(2)
    memory = defaultdict(list)
    requests = defaultdict(list)

    def handler(request):
        body = json.loads(request.content)
        uid = body["user_id"]
        requests[uid].append(request.url.path)
        if request.url.path == "/add":
            if len(requests[uid]) == 1:
                barrier.wait(timeout=5)
            memory[uid].extend(m["content"] for m in body["messages"])
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": str(i), "content": content, "created_at": "", "score": 1.0}
                    for i, content in enumerate(memory[uid])
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        runner.run_round(
            dataset="halumem",
            base_url="http://stub",
            bench_dir=tmp_path,
            out_dir=tmp_path / "aml",
            input_contract="aml-v1",
            ingest_workers=2,
            judge_workers=2,
            client=ServiceClient("http://stub", client=http),
        )
    assert len(seen) == 4
    for item in seen:
        assert ("FUTURE" in item["retrieved_context"]) == item["id"].startswith("late-")
    assert all(paths == ["/add", "/search", "/add", "/search"] for paths in requests.values())
