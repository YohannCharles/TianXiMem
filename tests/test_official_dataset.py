"""分类、固定分母、版本可比性及历史 Search 续跑的行为验证，不调用真实模型。"""

from __future__ import annotations

import json
from pathlib import Path

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
    runtime = json.loads(path.read_text())
    runtime["evaluation"] = {"model": "changed"}
    write_json(path, runtime)
    with pytest.raises(ValueError, match="模型"):
        compare_runs(left, right, capture / "out", "v1", "v2")
    manifest["questions"] = manifest["questions"][:1]
    write_json(right / "selection-manifest.json", manifest)
    with pytest.raises(ValueError, match="题目清单已变化"):
        summarize_run(right)


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
