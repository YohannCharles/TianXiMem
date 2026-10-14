"""Fixed selection, source isolation, resume safety and unchanged native scopes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from eval.datasets import selection_suite as selection
from eval.datasets.aml.plan import digest
from eval.datasets.preprocess import Message, Question, Sample, Session
from eval.experiments import selection_suite as runner
from eval.harness import JudgeResult, ServiceClient
from eval.harness.plan_driver import execute_plan, freeze_run


def candidate(origin="memory600", origin_id="memory500-0001", world="updated"):
    return {
        "origin": origin,
        "origin_id": origin_id,
        "dataset": "mquake-remastered",
        "native_qid": "CF3k-1-0",
        "source_key": ["mquake-remastered", "CF3k-1-0", world],
        "route": "native",
        "audit": [],
        "original_scope": digest(origin),
        "original_evaluation": digest([origin, origin_id]),
    }


def fixture_selection(sample):
    rows = []
    for q in sample.questions:
        row = candidate(origin_id=q.qid)
        row.update(
            dataset=sample.dataset,
            native_qid=q.qid,
            source_key=[sample.dataset, q.qid, "default"],
            scope_sha256=selection.sample_scope(sample),
            evaluation_sha256=selection.question_identity(q),
        )
        rows.append(row)
    return {
        "version": selection.VERSION,
        "questions": selection.deduplicate(rows),
        "counts": {"memory600": len(rows), "official_sample": 0},
        "excluded": [],
    }


@pytest.fixture
def sample():
    return Sample(
        "original-user",
        "longmemeval-s",
        (
            Session(
                "s", (Message("user", "source fact", 1), Message("assistant", "source response", 1))
            ),
        ),
        (
            Question("q1", "question one?", "GOLD_ONLY_1", "fact"),
            Question("q2", "question two?", "GOLD_ONLY_2", "fact"),
        ),
    )


def test_tenth_has_exact_total_preserves_strata_and_is_order_independent():
    rows = [{"seq": i, "dataset": "a" if i < 71 else "b"} for i in range(101)]
    selected = selection.stratified_tenth(rows, "seed")
    assert len(selected) == 11
    assert sum(r["dataset"] == "a" for r in selected) == 8
    assert selected == selection.stratified_tenth(list(reversed(rows)), "seed")
    assert selected != selection.stratified_tenth(rows, "different")
    with pytest.raises(ValueError, match="duplicate"):
        selection.stratified_tenth(rows + [rows[0]], "seed")
    assert selection.stratified_tenth([], "seed") == []


def test_same_source_merges_members_but_worlds_do_not_merge():
    memory = candidate()
    memory["audit"] = [{"status": "needs_review"}]
    official = candidate("official", 27)
    original = candidate("official", 28, "original")
    rows = selection.deduplicate([memory, official, original])
    assert len(rows) == 2
    merged = next(r for r in rows if r["source_key"][-1] == "updated")
    assert {m["origin"] for m in merged["members"]} == {"memory600", "official"}
    assert any(m["audit"] for m in merged["members"])
    assert (
        merged["id"]
        == next(
            r
            for r in selection.deduplicate([original, official, memory])
            if r["source_key"][-1] == "updated"
        )["id"]
    )


def test_address_resolution_preserves_variants_and_halu_indexing():
    qid = selection.official_native_id(
        {"dataset": "halumem"},
        {"source_locator": {"uuid": "u", "session_index": 5, "question_index": 3}},
        {},
        Path("."),
    )
    assert qid == "halumem-u-s006-q003"
    qid = selection.official_native_id(
        {"dataset": "musique", "gold_native": {"id": "same", "answerable": False}},
        {},
        {},
        Path("."),
    )
    assert qid == "musique-same-unanswerable"
    assert selection.official_native_id({"dataset": "scriptmem"}, {}, {}, Path(".")) is None


def test_portable_manifest_rejects_duplicates_and_unsupported_execution(tmp_path):
    row = selection.deduplicate([candidate()])[0]
    manifest = {"version": selection.VERSION, "questions": [row]}
    path = tmp_path / "selection.json"
    selection.write_json(path, manifest)
    assert selection.load_selection(path) == manifest
    manifest["questions"].append(row)
    selection.write_json(path, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        selection.load_selection(path)
    manifest["questions"] = [dict(row, route="official")]
    selection.write_json(path, manifest)
    with pytest.raises(ValueError, match="execution route"):
        selection.load_selection(path)


def test_compact_ids_preserve_tasks_and_audits_without_per_question_hashes(tmp_path):
    row = selection.deduplicate([candidate()])[0]
    row["members"][0]["audit"] = [{"aspect": "multi_hop_strength", "status": "needs_review"}]
    path = tmp_path / "ids.tsv"
    selection.write_ids(path, {"questions": [row]})
    text = path.read_text()
    assert text.splitlines()[-1] == "mquake-remastered\tCF3k-1-0"
    assert "sha256" not in text and "original_scope" not in text
    loaded = selection.load_selection(path)
    assert loaded["questions"][0]["id"] == row["id"]
    assert loaded["questions"][0]["audit"] == row["members"][0]["audit"]
    assert "scope_sha256" not in loaded["questions"][0]
    assert "sources" not in loaded


@pytest.mark.parametrize(
    "text",
    [
        "longmemeval-s\tq1\nlongmemeval-s\tq1\n",
        "unknown\tq1\n",
        "longmemeval-s\tq1\textra\n",
        "# audit\tlongmemeval-s\tq2\tboundary\tneeds_review\nlongmemeval-s\tq1\n",
        "# audit\tlongmemeval-s\tq1\tboundary\tfailed\nlongmemeval-s\tq1\n",
        "# no questions\n",
    ],
)
def test_invalid_compact_ids_fail(text):
    with pytest.raises(ValueError):
        selection.parse_ids(text)


def test_compact_ids_snapshot_blocks_changed_resume(sample, monkeypatch, tmp_path):
    text = "longmemeval-s\tq1\nlongmemeval-s\tq2\n"
    first = selection.parse_ids(text)
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [sample])
    jobs = runner.build_plans(first, tmp_path, None, None, "run")
    assert all("scope_sha256" in r and "evaluation_sha256" in r for r in first["questions"])
    freeze_run(tmp_path / "run", [p for p, _ in jobs], execution={"top_k": 100})
    changed = replace(
        sample, questions=(replace(sample.questions[0], gold="changed"), sample.questions[1])
    )
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [changed])
    other = runner.build_plans(selection.parse_ids(text), tmp_path, None, None, "run")
    with pytest.raises(ValueError, match="input/execution changed"):
        freeze_run(tmp_path / "run", [p for p, _ in other], execution={"top_k": 100})


def test_plan_keeps_whole_memory_and_gold_out_of_add(sample, monkeypatch):
    manifest = fixture_selection(sample)
    # Only one question is selected; both original messages must still be ingested.
    manifest["questions"] = manifest["questions"][:1]
    selected = replace(sample, questions=(sample.questions[0],))
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [selected])
    plan, rows = runner.build_plans(manifest, Path("."), None, None, "one")[0]
    assert len(rows) == 1
    text = json.dumps(plan.wire())
    assert "source fact" in text and "source response" in text
    assert "GOLD_ONLY" not in text
    assert plan.sample.user_id != sample.user_id
    other = runner.build_plans(manifest, Path("."), None, None, "two")[0][0]
    assert other.sample.user_id != plan.sample.user_id
    assert {r["id"] for r in rows} == {r["id"] for r in manifest["questions"]}


@pytest.mark.parametrize("mutation", ["memory", "gold", "missing"])
def test_changed_or_missing_native_input_fails_before_http(sample, monkeypatch, mutation):
    manifest = fixture_selection(sample)
    if mutation == "memory":
        changed = replace(sample, sessions=(Session("s", (Message("user", "different"),)),))
    elif mutation == "gold":
        changed = replace(
            sample, questions=(replace(sample.questions[0], gold="other"), sample.questions[1])
        )
    else:
        changed = replace(sample, questions=(sample.questions[0],))
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [changed])
    with pytest.raises(ValueError, match="changed|missing"):
        runner.build_plans(manifest, Path("."), None, None, "run")


def test_prepare_drops_unsupported_and_failed_audits_without_question_text(
    sample, monkeypatch, tmp_path
):
    root = tmp_path / "bundle"
    root.mkdir()
    selection.write_json(
        root / "audit-status.json",
        {
            "questions_with_open_findings": [
                {"id": "memory500-bad", "aspect": "abstention_gold", "status": "failed"}
            ]
        },
    )
    selection.write_json(
        root / "manifest.json", {"space_scope_fingerprints": {"s": {"sha256": "scope"}}}
    )
    questions = [
        {
            "id": "memory500-good",
            "source_dataset": "longmemeval-s",
            "source_question_id": "q1",
            "space_id": "s",
        },
        {
            "id": "memory500-bad",
            "source_dataset": "longmemeval-s",
            "source_question_id": "q2",
            "space_id": "s",
        },
    ]
    (root / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions))
    monkeypatch.setattr(selection, "validate", lambda bundle: None)
    monkeypatch.setattr(selection, "sha256", lambda path: "sha")
    monkeypatch.setattr(selection, "capture_links", lambda path: ({}, {}))
    monkeypatch.setattr(
        selection,
        "load_native",
        lambda root, wanted: [replace(sample, questions=(sample.questions[0],))],
    )
    rows = [
        {
            "seq": i,
            "dataset": "scriptmem",
            "scorable": True,
            "user_id": "user",
            "prior_adds_for_user": 1,
            "ts": str(i),
        }
        for i in range(10)
    ]
    monkeypatch.setattr(selection, "classified_rows", lambda path: rows)
    manifest = selection.prepare(tmp_path, tmp_path, root)
    assert manifest["counts"]["official_sample"] == 1
    assert manifest["counts"]["unique_tasks"] == 1
    assert manifest["counts"]["dropped_unsupported"] == 1
    assert len(manifest["excluded"]) == 2
    assert {r["route"] for r in manifest["questions"]} == {"native"}
    text = json.dumps(manifest)
    assert "GOLD_ONLY" not in text and "question one?" not in text
    assert str(tmp_path) not in text
    # A failed source cannot re-enter through a matching captured request.
    monkeypatch.setattr(selection, "official_native_id", lambda *args: "q2")
    monkeypatch.setattr(
        selection,
        "classified_rows",
        lambda path: [dict(row, dataset="longmemeval-s") for row in rows],
    )
    other = selection.prepare(tmp_path, tmp_path, root)
    assert other["counts"]["unique_tasks"] == 1
    assert len(other["excluded"]) == 2
    assert all(any(f["status"] == "failed" for f in r["audit"]) for r in other["excluded"])


def test_http_checkpoint_uses_existing_driver_and_reuses_search_snapshots(
    sample, monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [sample])
    jobs = runner.build_plans(fixture_selection(sample), Path("."), None, None, "run")
    plan = jobs[0][0]
    freeze_run(tmp_path / "run", [plan], execution={"top_k": 100})
    calls = []
    history = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        assert body["user_id"] == plan.sample.user_id
        if request.url.path == "/add":
            assert "GOLD_ONLY" not in request.content.decode()
            history.extend(m["content"] for m in body["messages"])
            return httpx.Response(200, json={"success": True, **body})
        return httpx.Response(
            200,
            json={
                "data": [{"id": "m", "content": "\n".join(history), "created_at": "", "score": 1.0}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as wire:
        client = ServiceClient("http://test", client=wire)
        directory = tmp_path / "run" / plan.sample.user_id
        hits = execute_plan(plan, client=client, out_dir=directory)
        assert set(hits) == {"q1", "q2"}
        assert calls == ["/add", "/search", "/search"]
        execute_plan(plan, client=client, out_dir=directory)
        assert calls == ["/add", "/search", "/search"]


def test_summary_keeps_missing_and_technical_errors_in_fixed_denominator(sample, tmp_path):
    manifest = fixture_selection(sample)
    selection.write_json(tmp_path / "selection.json", manifest)
    row = {"id": manifest["questions"][0]["id"], "result": asdict_result(False, "JUDGE_ERROR")}
    selection.write_json(tmp_path / "u" / "task-results.json", [row])
    first = runner.summarize(tmp_path)["groups"]["overall"]
    assert first == {
        "expected": 2,
        "completed": 1,
        "missing": 1,
        "correct": 0,
        "accuracy": None,
        "technical_errors": 1,
    }
    other = {"id": manifest["questions"][1]["id"], "result": asdict_result(True, "CORRECT")}
    selection.write_json(tmp_path / "u" / "task-results.json", [row, other])
    final = runner.summarize(tmp_path)["groups"]["overall"]
    assert final["accuracy"] == 0.5 and final["technical_errors"] == 1


def asdict_result(correct, label):
    return {"is_correct": correct, "label": label}


def test_compare_rejects_different_models_and_does_not_score_missing(sample, tmp_path):
    manifest = fixture_selection(sample)
    left, right = tmp_path / "left", tmp_path / "right"
    runtime = {
        "selection_sha256": digest(manifest),
        "evaluation": {"model": "test"},
        "suite_code_sha256": "code",
    }
    for path in (left, right):
        selection.write_json(path / "selection.json", manifest)
        selection.write_json(path / "run-manifest.json", runtime)
    result = runner.compare(left, right, tmp_path / "comparison")
    assert result["groups"]["overall"]["change_pp"] is None
    selection.write_json(right / "run-manifest.json", dict(runtime, evaluation={"model": "other"}))
    with pytest.raises(ValueError, match="not comparable"):
        runner.compare(left, right, tmp_path / "comparison")


@pytest.mark.parametrize("compact", [False, True])
def test_run_dispatches_existing_pipeline_and_resumes_without_http(
    sample, monkeypatch, tmp_path, compact
):
    from eval.datasets import prepare as data_prepare
    from eval.harness import api_config

    manifest = fixture_selection(sample)
    if compact:
        manifest = selection.parse_ids("longmemeval-s\tq1\nlongmemeval-s\tq2\n")
    monkeypatch.setattr(runner, "load_native", lambda root, wanted: [sample])
    jobs = runner.build_plans(manifest, tmp_path, None, None, "test")
    for key in (
        "ANSWER_API_BASE",
        "ANSWER_API_KEY",
        "ANSWER_MODEL",
        "JUDGE_API_BASE",
        "JUDGE_API_KEY",
        "JUDGE_MODEL",
    ):
        monkeypatch.setattr(api_config, key, "test-only")
    monkeypatch.setattr(data_prepare, "ensure_dataset", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "evaluation_fingerprint", lambda args: {"model": "test"})
    monkeypatch.setattr(runner, "config_fingerprint", lambda *a, **kw: {"snapshot_files": []})
    monkeypatch.setattr(runner, "pipeline_for", lambda root, dataset: Path(dataset))
    calls, judged = [], []
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (
                calls.append(request.url.path)
                or httpx.Response(200, json={"success": True, **json.loads(request.content)})
                if request.url.path == "/add"
                else calls.append("/search") or httpx.Response(200, json={"data": []})
            )
        )
    )
    monkeypatch.setattr(runner, "ServiceClient", lambda base: ServiceClient(base, client=client))

    def judge(pipeline, items, out_dir, **kwargs):
        judged.append((pipeline, items))
        assert pipeline == Path("longmemeval-s")
        assert all(item["speaker_1_memories"] == "" for item in items)
        return [JudgeResult(item["id"], True, "CORRECT", "", "answer") for item in items]

    monkeypatch.setattr(runner, "run_judge", judge)
    args = SimpleNamespace(
        data_root=tmp_path,
        reports_dir=str(tmp_path),
        run_id="trial",
        configs_dir=tmp_path,
        profile="local",
        base_url="http://test",
        service_revision="test",
        max_tokens=128,
        memory_date="none",
        offline=True,
    )
    try:
        assert runner.run(args, manifest, jobs) == 0
        assert calls == ["/add", "/search", "/search"]
        assert len(judged) == 1
        assert runner.run(args, manifest, jobs) == 0
        assert calls == ["/add", "/search", "/search"]
        assert runner.summarize(tmp_path / "runs" / "trial")["groups"]["overall"]["accuracy"] == 1.0
    finally:
        client.close()
