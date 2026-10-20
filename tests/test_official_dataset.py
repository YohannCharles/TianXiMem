"""分类、固定分母、版本可比性及历史 Search 续跑的行为验证，不调用真实模型。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from eval.datasets.official_capture import replay_plan
from eval.datasets.official_groups import (
    classified_rows,
    data_identity,
    load_manifest,
    prepare_groups,
    write_json,
)
from eval.experiments import official_dataset as pipeline
from eval.harness import JudgeResult, SearchHit
from eval.harness.plan_driver import execute_plan, freeze_run
from eval.jsonl_io import read_jsonl
from eval.reports.official_comparison import compare_runs, summarize_run


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_failure_progress_distinguishes_duplicate_history_and_recovered_work(tmp_path):
    write_rows(tmp_path / 'skipped.jsonl', [{'request_id': 'a'}, {'request_id': 'a'}, {'request_id': 'b'}])
    write_rows(tmp_path / 'recovered-adds.jsonl', [{'request_id': 'a', 'status': 'recovered'}])
    write_rows(tmp_path / 'errors.jsonl', [{'wire_user_id': 'u', 'n_questions': 2, 'question_id': None}])
    write_rows(tmp_path / 'u/labels.jsonl', [{'id': 'q1', 'label': 'WRONG'}, {'id': 'q2', 'label': 'CORRECT'}])
    status = pipeline._failure_status(tmp_path)
    assert '历史 3 次/2 批' in status
    assert '已恢复 1 批，待处理 1 批' in status
    assert '错误历史 1 次，待处理 0 项' in status
    write_rows(tmp_path / 'u/labels.jsonl', [{'id': 'q1', 'label': 'WRONG'}, {'id': 'q2', 'label': 'JUDGE_ERROR'}])
    assert '错误历史 1 次，待处理 1 项' in pipeline._failure_status(tmp_path)


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    root = tmp_path / "capture"
    kit = []
    for seq, user, dataset, answer, rubric in [
        (1, "a", "mquake-remastered", ["hello"], None),
        (3, "a", "beam", [], ["must say hi"]),
        (4, "a", "unknown", [], None),
        (6, "b", "mquake-remastered", ["hello"], None),
    ]:
        kit.append(
            {
                "seq": seq,
                "ts": str(seq),
                "user_id": user,
                "query": "q?",
                "question": "q?",
                "dataset": dataset,
                "task": "open",
                "attribution": "unknown" if dataset == "unknown" else "gold_extra",
                "judge_kind": "none" if dataset == "unknown" else "exact",
                "gold_answers": answer,
                "gold_rubric": rubric,
            }
        )
    searches = [{key: r[key] for key in ("seq", "ts", "user_id", "query")} for r in kit]
    adds = [
        {
            "seq": seq,
            "ts": str(seq),
            "user_id": user,
            "session_id": "s",
            "request_id": f"r{seq}",
            "messages": [{"role": "user", "content": content}],
        }
        for seq, user, content in [
            (0, "a", "  early  "),
            (2, "a", "future"),
            (5, "b", "other user"),
        ]
    ]
    timeline = [{"kind": "search", **r} for r in searches]
    timeline += [
        {"kind": "add", **{k: r[k] for k in ("seq", "ts", "user_id", "request_id")}} for r in adds
    ]
    write_rows(root / "official-searches.jsonl", searches)
    write_rows(root / "official-eval-kit.jsonl", kit)
    write_rows(root / "official-adds.jsonl", adds)
    write_rows(root / "official-timeline.jsonl", sorted(timeline, key=lambda r: r["seq"]))
    return root


def test_classification_keeps_repeated_queries_and_does_not_guess(capture: Path) -> None:
    manifest = prepare_groups(capture, capture / "grouped")
    assert manifest["counts"]["mquake-remastered"]["searches"] == 2
    assert manifest["counts"]["beam"]["rubric_only"] == 1
    assert manifest["counts"]["other"]["scorable"] == 0
    assert {r["seq"] for r in manifest["questions"]} == {1, 3, 6}
    assert len(read_jsonl(capture / "grouped/searches/other.jsonl")) == 1
    # Even a recognized name with only corpus heuristics must remain unresolved.
    kit = read_jsonl(capture / "official-eval-kit.jsonl")
    kit[3]["attribution"] = "corpus"
    write_rows(capture / "official-eval-kit.jsonl", kit)
    assert classified_rows(capture)[3]["dataset"] == "other"


def test_selection_is_deterministic_and_manifest_detects_source_changes(capture: Path) -> None:
    first = prepare_groups(capture, capture / "first", per_dataset=1)
    second = prepare_groups(capture, capture / "second", per_dataset=1)
    assert first == second
    assert [r["seq"] for r in first["questions"]] == [1, 3]
    load_manifest(capture / "first/manifest.json", verify_sources=True)
    with (capture / "official-adds.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="源文件已变化"):
        load_manifest(capture / "first/manifest.json", verify_sources=True)


def test_wrong_join_and_duplicate_seq_fail(capture: Path) -> None:
    kit = read_jsonl(capture / "official-eval-kit.jsonl")
    kit[0]["query"] = "wrong question"
    write_rows(capture / "official-eval-kit.jsonl", kit)
    with pytest.raises(ValueError, match="不匹配"):
        classified_rows(capture)
    kit.append(kit[-1])
    write_rows(capture / "official-eval-kit.jsonl", kit)
    with pytest.raises(ValueError, match="重复 seq"):
        classified_rows(capture)


def test_only_seqs_keeps_required_history_and_trims_future(capture: Path) -> None:
    plan = replay_plan(capture, only_seqs={1})[0]
    assert [e.seq for e in plan.events] == [0, 1]
    assert plan.events[0].messages[0]["content"] == "  early  "
    with pytest.raises(ValueError, match="不存在或不可评分"):
        replay_plan(capture, only_seqs={4})


class FakeService:
    def __init__(self, _base: str = ""):
        self.history = {}
        self.calls = []
        self.fail = False

    def add(self, *, user_id, messages, **kwargs):
        self.calls.append(("add", messages[0]["content"]))
        self.history.setdefault(user_id, []).extend(messages)

    def search(self, *, user_id, query, top_k):
        self.calls.append(("search", query))
        if self.fail:
            raise RuntimeError("search unavailable")
        text = "\n".join(m["content"] for m in self.history.get(user_id, []))
        return [SearchHit(id="m", content=text, score=1, created_at="")]

    def close(self):
        pass


def configure_official_runner(capture, monkeypatch):
    """Common offline setup for scheduler integration cases."""
    from eval.harness import official_judging

    prepare_groups(capture, capture / "grouped")
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda *a, **kw: None)
    for key in (
        "ANSWER_API_BASE", "ANSWER_API_KEY", "ANSWER_MODEL",
        "JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL",
    ):
        monkeypatch.setattr(pipeline.api_config, key, "configured")
    observed = {}

    def judge(_path, items, output, **kwargs):
        for item in items:
            observed[item["id"]] = item["retrieved_context"]
        write_rows(output / "answers.jsonl", [
            {"id": item["id"], "generated_answer": "hello"} for item in items
        ])
        write_rows(output / "labels.jsonl", [
            {"id": item["id"], "is_correct": True, "label": "CORRECT", "judge_response": "ok"}
            for item in items
        ])

    monkeypatch.setattr(pipeline, "run_judge", judge)
    monkeypatch.setattr(official_judging, "run_judge", judge)
    return observed, [
        "run", "--manifest", str(capture / "grouped/manifest.json"),
        "--run-id", "parallel", "--service-revision", "v1",
        "--reports-dir", str(capture / "reports"),
        "--ingest-workers", "2", "--judge-workers", "2",
    ]


def test_cross_user_http_overlaps_with_isolated_clients_and_ordered_history(capture, monkeypatch):
    import threading

    observed, argv = configure_official_runner(capture, monkeypatch)
    barrier = threading.Barrier(2)
    shared_history = {}
    instances = []
    first_adds = set()
    lock = threading.Lock()

    class Client(FakeService):
        def __init__(self, _base):
            super().__init__()
            self.history = shared_history
            self.closed = False
            self.owner = threading.get_ident()
            instances.append(self)

        def add(self, **kwargs):
            assert not self.closed and threading.get_ident() == self.owner
            with lock:
                first = kwargs["user_id"] not in first_adds
                first_adds.add(kwargs["user_id"])
            if first:
                barrier.wait(timeout=5)  # A serial replay would fail this barrier.
            super().add(**kwargs)

        def search(self, **kwargs):
            assert not self.closed and threading.get_ident() == self.owner
            return super().search(**kwargs)

        def close(self):
            self.closed = True

    monkeypatch.setattr(pipeline, "ServiceClient", Client)
    assert pipeline.main(argv) == 0
    assert len(instances) == 2 and all(c.closed for c in instances)
    assert sorted([[kind for kind, _ in c.calls] for c in instances], key=len) == [
        ["add", "search"], ["add", "search", "add", "search"]
    ]
    assert "future" not in observed["1"] and "other user" not in observed["1"]
    assert "future" in observed["3"] and "other user" not in observed["3"]
    assert "other user" in observed["6"] and "early" not in observed["6"]
    summary = summarize_run(capture / "reports/runs/parallel")
    assert summary["completed"] == 3 and not summary["errors"]


def test_after_replay_starts_models_after_last_search_and_resumes_without_http(capture, monkeypatch):
    from eval.harness import official_judging

    observed, argv = configure_official_runner(capture, monkeypatch)
    service = FakeService()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    original_judge = official_judging.run_judge

    def judge(*args, **kwargs):
        assert sum(kind == "search" for kind, _ in service.calls) == 3
        return original_judge(*args, **kwargs)

    monkeypatch.setattr(official_judging, "run_judge", judge)
    argv += ["--judge-scheduling", "after-replay"]
    assert pipeline.main(argv) == 0
    assert "future" not in observed["1"] and "future" in observed["3"]
    before = list(service.calls)
    assert pipeline.main(argv) == 0
    assert service.calls == before


def test_replay_only_saves_historical_hits_without_models_and_resumes_offline(capture, monkeypatch):
    observed, argv = configure_official_runner(capture, monkeypatch)
    service = FakeService()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay-only must not prepare or call answer/judge")

    monkeypatch.setattr(pipeline, "ensure_dataset", forbidden)
    monkeypatch.setattr(pipeline, "run_judge", forbidden)
    monkeypatch.setattr(pipeline, "StreamingOfficialJudge", forbidden)
    monkeypatch.setattr(pipeline.api_config, "ANSWER_API_KEY", "")
    argv += ["--replay-only", "--judge-scheduling", "after-replay"]
    assert pipeline.main(argv) == 0
    output = capture / "reports/runs/parallel"
    hits = {}
    for path in output.glob("*/checkpoint.json"):
        hits.update(json.loads(path.read_text(encoding="utf-8"))["hits"])
    assert set(hits) == {"1", "3", "6"}
    assert "future" not in hits["1"][0]["content"]
    assert "future" in hits["3"][0]["content"]
    assert not list(output.rglob("answers.jsonl"))
    assert not list(output.rglob("labels.jsonl"))
    assert not observed
    assert json.loads((output / "retrieval-status.json").read_text(encoding="utf-8"))["complete"] is True
    before = list(service.calls)
    assert pipeline.main(argv) == 0
    assert service.calls == before


@pytest.mark.parametrize("grouped", [False, True])
def test_non_prefix_user_finishes_while_prefix_user_waits(capture, monkeypatch, grouped):
    import threading

    observed, argv = configure_official_runner(capture, monkeypatch)
    prefix = capture / "prefix.json"
    write_json(prefix, [{"source_user_id": "a", "next_event": 2}])
    argv += ["--serial-prefix", str(prefix), "--judge-scheduling", "after-replay"]
    if grouped:
        argv += ["--rerank-local-workers", "1", "--rerank-memory3-workers", "1"]
    other_done = threading.Event()

    class Client(FakeService):
        def add(self, **kwargs):
            if kwargs["messages"][0]["content"] == "  early  ":
                assert other_done.wait(timeout=5), "Non-prefix user blocked by prefix"
            return super().add(**kwargs)

        def search(self, **kwargs):
            result = super().search(**kwargs)
            if any(m["content"] == "other user" for m in self.history[kwargs["user_id"]]):
                other_done.set()
            return result

    service = Client()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    assert pipeline.main(argv) == 0
    assert other_done.is_set()
    assert "future" not in observed["1"] and "future" in observed["3"]
    assert observed["6"] == "other user"


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_cold_arms_reproduce_same_serial_prefix_and_resume_without_http(capture, monkeypatch, staged, grouped):
    observed, argv = configure_official_runner(capture, monkeypatch)
    prefix = capture / "prefix.json"
    payload = [{"source_user_id": "a", "next_event": 2}]
    if staged:
        payload = {"stages": [
            {"workers": 1, "users": payload},
            {"workers": 2, "users": [
                {"source_user_id": "a", "next_event": 3},
                {"source_user_id": "b", "next_event": 1},
            ]},
        ]}
        argv[argv.index("--ingest-workers") + 1] = "6"
    write_json(prefix, payload)
    argv += ["--serial-prefix", str(prefix)]
    if grouped:
        argv[argv.index("--ingest-workers") + 1] = "48"
        argv += ["--rerank-local-workers", "16", "--rerank-memory3-workers", "32"]
    calls_by_user = {}

    class Client(FakeService):
        def add(self, **kwargs):
            calls_by_user.setdefault(kwargs["user_id"], []).append("add")
            return super().add(**kwargs)

        def search(self, **kwargs):
            calls_by_user.setdefault(kwargs["user_id"], []).append("search")
            return super().search(**kwargs)

    service = Client()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    for run_name in ("left", "right"):
        argv[argv.index("--run-id") + 1] = run_name
        before = len(service.calls)
        assert pipeline.main(argv) == 0
        runtime = json.loads((capture / "reports/runs" / run_name / "run-manifest.json").read_text(encoding="utf-8"))
        assert calls_by_user[runtime["users"]["a"]] == ["add", "search", "add", "search"]
        assert calls_by_user[runtime["users"]["b"]] == ["add", "search"]
        assert len(service.calls) - before == 6
        assert "future" not in observed["1"]
        calls = list(service.calls)
        assert pipeline.main(argv) == 0
        assert service.calls == calls
    comparison = compare_runs(
        capture / "reports/runs/left", capture / "reports/runs/right",
        capture / "comparison", "left", "right"
    )
    assert comparison["left"]["evaluation"]["serial_prefix"] == payload
    if grouped:
        mappings = []
        for run_name in ("left", "right"):
            directory = capture / "reports/runs" / run_name
            groups = json.loads((directory / "rerank-user-groups.json").read_text(encoding="utf-8"))
            runtime = json.loads((directory / "run-manifest.json").read_text(encoding="utf-8"))
            mappings.append({source: groups["users"][wire] for source, wire in runtime["users"].items()})
        assert mappings[0] == mappings[1]


@pytest.mark.parametrize("grouped", [False, True])
def test_independent_prefix_users_overlap_stages_keep_routing_and_resume(capture, monkeypatch, grouped):
    import threading

    observed, argv = configure_official_runner(capture, monkeypatch)
    prefix = capture / "prefix.json"
    write_json(prefix, {"stages": [
        {"workers": 1, "users": [{"source_user_id": "a", "next_event": 2}]},
        {"workers": 1, "rerank_routing": "query-bucket", "users": [
            {"source_user_id": "a", "next_event": 3},
            {"source_user_id": "b", "next_event": 2},
        ]},
    ]})
    argv += ["--serial-prefix", str(prefix), "--prefix-scheduling", "independent"]
    if grouped:
        argv[argv.index("--ingest-workers") + 1] = "4"
        argv += ["--rerank-local-workers", "2", "--rerank-memory3-workers", "2"]
    other_done = threading.Event()
    output = capture / "reports/runs/parallel"

    class Client(FakeService):
        def search(self, **kwargs):
            history = self.history[kwargs["user_id"]]
            if history[0]["content"] == "  early  ":
                assert other_done.wait(timeout=5), "Prefix users blocked across stages"
            else:
                active = json.loads((output / "ingest-progress.json").read_text())["active_users"]
                assert active[kwargs["user_id"]]["prefix_index"] == 1
                assert active[kwargs["user_id"]]["prefix_routing"] == "query-bucket"
                other_done.set()
            return super().search(**kwargs)

    service = Client()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    assert pipeline.main(argv) == 0
    assert "future" not in observed["1"] and "future" in observed["3"]
    assert observed["6"] == "other user"
    assert not (output / "errors.jsonl").exists()
    before = list(service.calls)
    assert pipeline.main(argv) == 0
    assert service.calls == before


def test_independent_prefix_failure_does_not_advance_to_later_stages(capture, monkeypatch):
    _, argv = configure_official_runner(capture, monkeypatch)
    prefix = capture / "prefix.json"
    write_json(prefix, {"stages": [
        {"workers": 1, "users": [{"source_user_id": "a", "next_event": 2}]},
        {"workers": 1, "users": [{"source_user_id": "a", "next_event": 3}]},
    ]})
    argv += ["--serial-prefix", str(prefix), "--prefix-scheduling", "independent", "--replay-only"]
    service = FakeService()
    service.fail = True
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    assert pipeline.main(argv) == 0
    assert ("add", "future") not in service.calls
    assert service.calls.count(("add", "  early  ")) == 1
    assert service.calls.count(("search", "q?")) == 2
    output = capture / "reports/runs/parallel"
    assert json.loads((output / "retrieval-status.json").read_text())["complete"] is False
    service.fail = False
    assert pipeline.main(argv) == 0
    assert json.loads((output / "retrieval-status.json").read_text())["complete"] is True
    assert service.calls.count(("add", "  early  ")) == 1
    assert service.calls.count(("add", "future")) == 1


def test_search_failure_blocks_future_add_and_resume_keeps_historical_hits(capture: Path) -> None:
    source = replay_plan(capture, only_seqs={1, 3})[0]
    plan = pipeline.checkpoint_plan(source, "test-run")
    output = capture / "run"
    freeze_run(output, [plan], execution={"top_k": 100})
    service = FakeService()
    service.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        execute_plan(plan, client=service, out_dir=output / plan.sample.user_id)
    assert service.calls == [("add", "  early  "), ("search", "q?")]
    service.fail = False
    hits = execute_plan(plan, client=service, out_dir=output / plan.sample.user_id)
    assert hits["1"][0].content == "  early  "
    assert hits["3"][0].content == "  early  \nfuture"
    n_calls = len(service.calls)
    assert execute_plan(plan, client=service, out_dir=output / plan.sample.user_id) == hits
    assert len(service.calls) == n_calls
    assert "hello" not in json.dumps(plan.wire())  # reference answers never enter HTTP


def test_search_callback_only_publishes_durable_history_and_replays_without_http(capture):
    source = replay_plan(capture, only_seqs={1, 3})[0]
    plan = pipeline.checkpoint_plan(source, "streaming-test")
    output = capture / "streaming-run"
    freeze_run(output, [plan], execution={"top_k": 100})
    user_dir = output / plan.sample.user_id
    service = FakeService()
    published = []

    def publish(qid, hits):
        checkpoint = json.loads((user_dir / "checkpoint.json").read_text(encoding="utf-8"))
        assert checkpoint["hits"][qid][0]["content"] == hits[0].content
        published.append((qid, hits[0].content, list(service.calls)))

    execute_plan(plan, client=service, out_dir=user_dir, on_search=publish)
    assert [r[:2] for r in published] == [
        ("1", "  early  "), ("3", "  early  \nfuture")
    ]
    assert ("add", "future") not in published[0][2]
    calls = list(service.calls)
    published.clear()
    execute_plan(plan, client=service, out_dir=user_dir, on_search=publish)
    assert service.calls == calls
    assert [r[:2] for r in published] == [
        ("1", "  early  "), ("3", "  early  \nfuture")
    ]


def test_serial_prefix_boundary_and_cooperative_cancel_keep_history(capture):
    import threading

    source = replay_plan(capture, only_seqs={1, 3})[0]
    plan = pipeline.checkpoint_plan(source, "prefix-test")
    output = capture / "prefix-run"
    freeze_run(output, [plan], execution={"top_k": 100})
    service = FakeService()
    user_dir = output / plan.sample.user_id
    hits = execute_plan(plan, client=service, out_dir=user_dir, stop_at_event=2)
    assert list(hits) == ["1"]
    assert service.calls == [("add", "  early  "), ("search", "q?")]
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        execute_plan(plan, client=service, out_dir=user_dir, stop_requested=stop.is_set)
    assert len(service.calls) == 2
    hits = execute_plan(plan, client=service, out_dir=user_dir)
    assert hits["1"][0].content == "  early  "
    assert hits["3"][0].content == "  early  \nfuture"
    assert service.calls[2:] == [("add", "future"), ("search", "q?")]


class FailingAddService(FakeService):
    """第一条 add 稳定 500（模拟官方采集里那条"单块超过 embedding 上限"）。"""

    def __init__(self, _base: str = ""):
        super().__init__(_base)
        self.failed = False

    def add(self, *, user_id, messages, **kwargs):
        if not self.failed and messages[0]["content"] == "  early  ":
            self.failed = True
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "http://service/add"),
                response=httpx.Response(500, text="boom"),
            )
        super().add(user_id=user_id, messages=messages, **kwargs)


def test_add_error_handler_skips_the_batch_and_advances_cursor(capture: Path) -> None:
    """官方采集的长跑不能死在一条注定 500 的 add 上——跳过并记账，游标继续前进。

    跳过后重跑**不得重投**它（游标已经越过）；这正是"线上那一批也丢了"的复现。
    """
    source = replay_plan(capture, only_seqs={1, 3})[0]
    plan = pipeline.checkpoint_plan(source, "skip-run")
    output = capture / "run-skip"
    freeze_run(output, [plan], execution={"top_k": 100})
    service = FailingAddService()
    skipped: list[tuple[str, str]] = []

    hits = execute_plan(
        plan,
        client=service,
        out_dir=output / plan.sample.user_id,
        on_add_error=lambda event, error: skipped.append((event.request_id, type(error).__name__)),
    )
    assert skipped == [("r0", "HTTPStatusError")]
    assert hits["1"][0].content == ""  # 那一批确实没进库
    assert hits["3"][0].content == "future"
    n_calls = len(service.calls)
    assert (
        execute_plan(
            plan,
            client=service,
            out_dir=output / plan.sample.user_id,
            on_add_error=lambda event, error: skipped.append(
                (event.request_id, type(error).__name__)
            ),
        )
        == hits
    )
    assert len(service.calls) == n_calls
    assert len(skipped) == 1  # 续跑没有重投，也没有重复记账


def test_results_from_labels_reads_the_dispatch_shape(tmp_path: Path) -> None:
    """续跑时 run record 必须从磁盘读回**所有**用户，而不是只统计本轮返回值。"""
    from eval.harness.judge_results import results_from_labels

    write_rows(
        tmp_path / "answers.jsonl",
        [{"id": "1", "generated_answer": "alpha"}, {"id": "3", "generated_answer": "beta"}],
    )
    write_rows(
        tmp_path / "labels.jsonl",
        [
            {"id": "1", "is_correct": True, "label": "CORRECT", "judge_response": "ok"},
            {"id": "3", "is_correct": False, "label": "WRONG", "judge_response": "no"},
        ],
    )
    results = results_from_labels(tmp_path)
    assert [(r.qid, r.is_correct, r.generated_answer) for r in results] == [
        ("1", True, "alpha"),
        ("3", False, "beta"),
    ]


def fake_run(root: Path, name: str, manifest: dict, labels: list[dict]) -> Path:
    run = root / name
    write_json(run / "selection-manifest.json", manifest)
    write_json(
        run / "run-manifest.json",
        {
            "run_id": name,
            "service_revision": name,
            "data_identity": data_identity(manifest),
            "evaluation": {"model": "fixed"},
            "users": {"a": "wire-a", "b": "wire-b"},
        },
    )
    users = {str(r["seq"]): r["user_id"] for r in manifest["questions"]}
    for user in ("a", "b"):
        write_rows(
            run / f"wire-{user}/labels.jsonl", [r for r in labels if users[str(r["id"])] == user]
        )
    return run


def test_comparison_fixed_denominator_errors_and_missing(capture: Path) -> None:
    manifest = prepare_groups(capture, capture / "grouped")
    left = fake_run(
        capture,
        "v1",
        manifest,
        [
            {"id": "1", "is_correct": True, "label": "CORRECT"},
            {"id": "3", "is_correct": False, "label": "JUDGE_ERROR"},
        ],
    )
    right = fake_run(
        capture,
        "v2",
        manifest,
        [{"id": str(seq), "is_correct": True, "label": "CORRECT"} for seq in (1, 3, 6)],
    )
    summary = summarize_run(left)
    assert (summary["expected"], summary["completed"], summary["missing"]) == (3, 2, 1)
    assert summary["accuracy"] is None
    assert summary["datasets"]["beam"]["errors"] == {"JUDGE_ERROR": 1}
    assert summary["datasets"]["beam"]["accuracy"] == 0
    assert summary["datasets"]["other"]["expected"] == 0
    result = compare_runs(left, right, capture / "comparison", "v1", "v2")
    rows = {r["dataset"]: r for r in result["rows"]}
    assert rows["mquake-remastered"]["delta_pp"] is None
    assert rows["beam"]["delta_pp"] == 100
    assert (capture / "comparison/comparison.csv").exists()


def test_comparison_rejects_different_inputs_or_models(capture: Path) -> None:
    manifest = prepare_groups(capture, capture / "grouped")
    left = fake_run(capture, "left", manifest, [])
    right = fake_run(capture, "right", manifest, [])
    path = right / "run-manifest.json"
    runtime = json.loads(path.read_text(encoding="utf-8"))
    runtime["evaluation"] = {"model": "changed"}
    write_json(path, runtime)
    with pytest.raises(ValueError, match="模型"):
        compare_runs(left, right, capture / "out", "v1", "v2")
    manifest["questions"] = manifest["questions"][:1]
    write_json(right / "selection-manifest.json", manifest)
    with pytest.raises(ValueError, match="题目清单已变化"):
        summarize_run(right)


def test_runtime_drift_allows_code_changes_but_not_scoring_changes() -> None:
    """长跑期间改代码/配置只记账；改模型/端点/预算/服务版本必须换 run-id。"""
    import copy

    base = {
        "data_identity": "x",
        "users": {"u": "wire-u"},
        "top_k": 100,
        "service_revision": "v1",
        "service_base": "http://127.0.0.1:8000",
        "evaluation": {
            "answer_model": "m",
            "judge_model": "j",
            "answer_base": "a",
            "judge_base": "b",
            "max_tokens": 1024,
            "memory_date": "none",
            "annotate_mark": "paren",
            "code_sha256": {"a.py": "1"},
            "upstream_sha256": {"p.py": "1"},
        },
        "config_fingerprint": {"snapshot_hashes": {"default.yaml": "h1"}},
    }
    changed = copy.deepcopy(base)
    changed["evaluation"]["code_sha256"] = {"a.py": "2"}
    changed["config_fingerprint"]["snapshot_hashes"]["default.yaml"] = "h2"
    assert pipeline._runtime_drift(base, changed) == (
        [],
        ["config_fingerprint", "evaluation.code_sha256"],
    )

    for path in (
        ("evaluation", "answer_model"),
        ("evaluation", "max_tokens"),
        ("service_revision",),
        ("data_identity",),
        ("users",),
    ):
        changed = copy.deepcopy(base)
        cursor = changed
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = "changed"
        fatal, _ = pipeline._runtime_drift(base, changed)
        assert ".".join(path) in fatal, path


def test_full_runner_uses_checkpoint_driver_and_refuses_changed_version(
    capture: Path, monkeypatch
) -> None:
    prepare_groups(capture, capture / "grouped")
    service = FakeService()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda *a, **kw: None)
    for key in (
        "ANSWER_API_BASE",
        "ANSWER_API_KEY",
        "ANSWER_MODEL",
        "JUDGE_API_BASE",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
    ):
        monkeypatch.setattr(pipeline.api_config, key, "configured")

    def fake_judge(_path, items, output, **kwargs):
        labels = [
            {"id": i["id"], "is_correct": True, "label": "CORRECT", "judge_response": "ok"}
            for i in items
        ]
        write_rows(output / "labels.jsonl", labels)
        return [
            JudgeResult(
                qid=r["id"],
                is_correct=True,
                label="CORRECT",
                judge_response="ok",
                generated_answer="ok",
            )
            for r in labels
        ]

    monkeypatch.setattr(pipeline, "run_judge", fake_judge)
    argv = [
        "run",
        "--manifest",
        str(capture / "grouped/manifest.json"),
        "--run-id",
        "test",
        "--service-revision",
        "v1",
        "--reports-dir",
        str(capture / "reports"),
    ]
    assert pipeline.main(argv) == 0
    summary = summarize_run(capture / "reports/runs/test")
    assert summary["completed"] == 3 and summary["accuracy"] == 1
    n_calls = len(service.calls)
    assert pipeline.main(argv) == 0
    assert len(service.calls) == n_calls
    argv[argv.index("v1")] = "v2"
    assert pipeline.main(argv) == 2


def test_streaming_runner_answers_historical_search_while_http_advances(capture, monkeypatch):
    import threading

    from eval.harness import official_judging

    prepare_groups(capture, capture / "grouped")
    answering = threading.Event()
    searched_future = threading.Event()

    class OverlappingService(FakeService):
        def search(self, **kwargs):
            if len([c for c in self.calls if c[0] == "search"]) == 1:
                assert answering.wait(timeout=5), "answer did not start during replay"
                searched_future.set()
            return super().search(**kwargs)

    service = OverlappingService()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda *a, **kw: None)
    for key in (
        "ANSWER_API_BASE", "ANSWER_API_KEY", "ANSWER_MODEL",
        "JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL",
    ):
        monkeypatch.setattr(pipeline.api_config, key, "configured")
    judged = []

    def fake_judge(_path, items, output, **kwargs):
        item = items[0]
        if item["id"] == "1":
            answering.set()
            assert searched_future.wait(timeout=5)
            assert "future" not in item["retrieved_context"]
        judged.append(item["id"])
        write_rows(output / "answers.jsonl", [{"id": item["id"], "generated_answer": "hello"}])
        write_rows(output / "labels.jsonl", [
            {"id": item["id"], "is_correct": True, "label": "CORRECT", "judge_response": "ok"}
        ])

    monkeypatch.setattr(official_judging, "run_judge", fake_judge)
    argv = [
        "run", "--manifest", str(capture / "grouped/manifest.json"),
        "--run-id", "streaming", "--service-revision", "v1",
        "--reports-dir", str(capture / "reports"), "--judge-workers", "2",
    ]
    assert pipeline.main(argv) == 0
    run_dir = capture / "reports/runs/streaming"
    assert summarize_run(run_dir)["completed"] == 3
    assert sorted(judged) == ["1", "3", "6"]
    calls = list(service.calls)
    assert pipeline.main(argv) == 0
    assert service.calls == calls
    assert len(judged) == 3


def test_full_runner_records_user_failure_and_continues(capture: Path, monkeypatch) -> None:
    """一个用户的裁判失败**不能打死整轮**：记账、继续下一个用户，重跑同 run-id 续。

    这是官方采集全量（8,783 题 / 1,374 用户 / 40,463 条 add）能跑完的前提——
    任何一次子进程失败都让整个跑批归零，就永远跑不完。
    """
    prepare_groups(capture, capture / "grouped")
    service = FakeService()
    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: service)
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda *a, **kw: None)
    for key in (
        "ANSWER_API_BASE",
        "ANSWER_API_KEY",
        "ANSWER_MODEL",
        "JUDGE_API_BASE",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
    ):
        monkeypatch.setattr(pipeline.api_config, key, "configured")

    calls: list[str] = []

    def flaky_judge(_path, items, output, **kwargs):
        calls.append(output.name)
        if len(calls) == 1:
            raise RuntimeError("judge subprocess died")
        labels = [
            {"id": i["id"], "is_correct": True, "label": "CORRECT", "judge_response": "ok"}
            for i in items
        ]
        write_rows(output / "labels.jsonl", labels)
        return [
            JudgeResult(
                qid=row["id"],
                is_correct=True,
                label="CORRECT",
                judge_response="ok",
                generated_answer="ok",
            )
            for row in labels
        ]

    monkeypatch.setattr(pipeline, "run_judge", flaky_judge)
    argv = [
        "run",
        "--manifest",
        str(capture / "grouped/manifest.json"),
        "--run-id",
        "resilient",
        "--service-revision",
        "v1",
        "--reports-dir",
        str(capture / "reports"),
    ]
    assert pipeline.main(argv) == 0
    run_dir = capture / "reports/runs/resilient"
    assert (run_dir / "errors.jsonl").is_file()
    assert len(read_jsonl(run_dir / "errors.jsonl")) == 1
    summary = summarize_run(run_dir)
    # a（2 题）失败、b（1 题）成功 ⇒ 分母仍是 3，缺 2，不冒充完成。
    assert (summary["expected"], summary["completed"], summary["missing"]) == (3, 1, 2)
    assert summary["accuracy"] is None


def test_metric_totals_keep_exact_and_f1_separate() -> None:
    """EM 决定 is_correct，F1 只作指标——两者分别汇总，不能互相替代。"""
    from eval.reports.official_comparison import _metric_totals

    totals = _metric_totals(
        [
            {"metrics": {"exact_match": 1.0, "f1": 0.5}},
            {"metrics": {"exact_match": 0.0, "f1": 0.25}},
        ]
    )
    assert totals["exact_match"] == {"sum": 1.0, "n": 2, "mean": 0.5}
    assert totals["f1"] == {"sum": 0.75, "n": 2, "mean": 0.375}
    # 缺席时不写 0.0（那会被读成"全错"）。
    assert _metric_totals([{"metrics": {}}])["f1"] == {"sum": 0, "n": 0, "mean": None}
def test_failed_add_preserves_full_payload_without_truncation(capture, monkeypatch):
    messages = [{"role": "user", "content": "  原文 " * 10000}]
    adds = read_jsonl(capture / "official-adds.jsonl")
    adds[0]["messages"] = messages
    write_rows(capture / "official-adds.jsonl", adds)
    _, argv = configure_official_runner(capture, monkeypatch)

    class Client(FakeService):
        def add(self, *, messages, **kwargs):
            if kwargs["request_id"] == "r0":
                raise httpx.HTTPStatusError(
                    "embedding context exceeded", request=httpx.Request("POST", "http://service/add"),
                    response=httpx.Response(500),
                )
            super().add(messages=messages, **kwargs)

    monkeypatch.setattr(pipeline, "ServiceClient", Client)
    assert pipeline.main(argv) == 0
    run_dir = capture / "reports/runs/parallel"
    failures = read_jsonl(run_dir / "skipped.jsonl")
    assert len(failures) == 1
    payload = json.loads((run_dir / failures[0]["payload_file"]).read_text(encoding="utf-8"))
    assert payload["messages"] == messages
    assert payload["request_id"] == "r0"
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert payload["user_id"] == manifest["users"]["a"]
    assert pipeline.main(argv) == 0
    assert len(read_jsonl(run_dir / "skipped.jsonl")) == 1


def test_add_4xx_is_fatal_not_skipped(capture: Path, monkeypatch) -> None:
    """4xx（尤其 409）是系统性我方 bug——**不能**记进 skipped 后继续跑完。"""
    prepare_groups(capture, capture / "grouped")
    monkeypatch.setattr(pipeline, "ensure_dataset", lambda *a, **kw: None)
    for key in (
        "ANSWER_API_BASE",
        "ANSWER_API_KEY",
        "ANSWER_MODEL",
        "JUDGE_API_BASE",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
    ):
        monkeypatch.setattr(pipeline.api_config, key, "configured")

    class RejectingService:
        def __init__(self, _base: str = ""):
            pass

        def add(self, **_kwargs):
            raise httpx.HTTPStatusError(
                "conflict",
                request=httpx.Request("POST", "http://service/add"),
                response=httpx.Response(409, text="request_id conflict"),
            )

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "ServiceClient", lambda _: RejectingService())
    argv = [
        "run",
        "--manifest",
        str(capture / "grouped/manifest.json"),
        "--run-id",
        "fatal",
        "--service-revision",
        "v1",
        "--reports-dir",
        str(capture / "reports"),
    ]
    assert pipeline.main(argv) == 2
    run_dir = capture / "reports/runs/fatal"
    assert not (run_dir / "skipped.jsonl").exists()
