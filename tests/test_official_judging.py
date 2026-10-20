"""逐题作答流水线：真正的重叠、并发上限、续用答案和聚合恢复，不调用模型。"""

import json
import threading
from pathlib import Path

from eval.harness import official_judging as module
from eval.jsonl_io import jsonl_line, read_jsonl


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(jsonl_line(row) for row in rows), encoding="utf-8")


def answer(qid, text="saved answer"):
    return {"id": qid, "generated_answer": text}


def label(qid, text="CORRECT"):
    return {"id": qid, "is_correct": text == "CORRECT", "label": text, "judge_response": "ok"}


def test_work_overlaps_submission_with_bounded_workers_and_single_aggregation(tmp_path, monkeypatch):
    started = threading.Barrier(3)
    release = threading.Event()
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake(_pipeline, items, out_dir, **kwargs):
        nonlocal active, maximum
        qid = items[0]["id"]
        with lock:
            active += 1
            maximum = max(maximum, active)
        if qid in ("1", "2"):
            started.wait(timeout=5)
        assert release.wait(timeout=5)
        write(out_dir / "answers.jsonl", [answer(qid)])
        write(out_dir / "labels.jsonl", [label(qid)])
        with lock:
            active -= 1

    monkeypatch.setattr(module, "run_judge", fake)
    errors = []
    queue = module.StreamingOfficialJudge(
        tmp_path, Path("fake"), workers=2, max_tokens=1024, on_error=lambda *args: errors.append(args)
    )
    try:
        for qid in ("1", "2", "3", "4"):
            queue.submit("user", {"id": qid, "retrieved_context": "frozen " + qid})
        started.wait(timeout=5)
        assert queue.active == 2
        assert not (tmp_path / "user/labels.jsonl").exists()
        queue.submit("user", {"id": "1"})  # duplicate dispatch must not overwrite its input
    finally:
        release.set()
        queue.close()
    assert maximum == 2
    assert not errors
    assert len(read_jsonl(tmp_path / "user/answers.jsonl")) == 4
    assert {r["id"] for r in read_jsonl(tmp_path / "user/labels.jsonl")} == {"1", "2", "3", "4"}
    assert read_jsonl(tmp_path / "user/questions/1/input.jsonl")[0]["retrieved_context"] == "frozen 1"


def test_existing_answers_are_reused_by_real_subprocess_and_completed_labels_skip(tmp_path):
    pipeline = tmp_path / "stub.py"
    pipeline.write_text('''
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('command'); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--answers'); p.add_argument('--max-tokens'); a=p.parse_args()
rows=lambda p: [json.loads(s) for s in Path(p).read_text().split('\\n') if s.strip()] if Path(p).exists() else []
if a.command=='answer':
    assert rows(a.output)==[{'id':'1','generated_answer':'keep existing'}], 'old answer was not seeded'
else:
    assert rows(a.answers)[0]['generated_answer']=='keep existing'
    Path(a.output).write_text(json.dumps({'id':'1','is_correct':True,'label':'CORRECT','judge_response':'ok'})+'\\n')
''', encoding="utf-8")
    write(tmp_path / "user/answers.jsonl", [answer("1", "keep existing")])
    write(tmp_path / "user/labels.jsonl", [label("2", "WRONG")])
    errors = []
    queue = module.StreamingOfficialJudge(
        tmp_path, pipeline, workers=2, max_tokens=1024, on_error=lambda *args: errors.append(args)
    )
    queue.submit("user", {"id": "1"})
    queue.submit("user", {"id": "2"})
    queue.close()
    assert not errors
    assert read_jsonl(tmp_path / "user/answers.jsonl") == [answer("1", "keep existing")]
    assert [r["id"] for r in read_jsonl(tmp_path / "user/labels.jsonl")] == ["2", "1"]
    assert not (tmp_path / "user/questions/2").exists()


def test_failed_job_preserves_partial_answer_and_resume_merges_once(tmp_path, monkeypatch):
    errors = []

    def fails(_pipeline, items, out_dir, **kwargs):
        write(out_dir / "answers.jsonl", [answer("1")])
        raise RuntimeError("temporary judge outage")

    monkeypatch.setattr(module, "run_judge", fails)
    queue = module.StreamingOfficialJudge(
        tmp_path, Path("fake"), workers=1, max_tokens=1024, on_error=lambda *args: errors.append(args)
    )
    queue.submit("user", {"id": "1"})
    queue.close()
    assert len(errors) == 1
    assert not read_jsonl(tmp_path / "user/labels.jsonl")

    def resumes(_pipeline, items, out_dir, **kwargs):
        assert read_jsonl(out_dir / "answers.jsonl") == [answer("1")]
        write(out_dir / "labels.jsonl", [label("1")])

    monkeypatch.setattr(module, "run_judge", resumes)
    for _ in range(2):
        queue = module.StreamingOfficialJudge(
            tmp_path, Path("fake"), workers=1, max_tokens=1024,
            on_error=lambda *args: errors.append(args),
        )
        queue.submit("user", {"id": "1"})
        queue.close()
    assert len(errors) == 1
    assert read_jsonl(tmp_path / "user/answers.jsonl") == [answer("1")]
    assert read_jsonl(tmp_path / "user/labels.jsonl") == [label("1")]


def test_retry_errors_reopens_question_cache_without_regenerating_answer(tmp_path, monkeypatch):
    job = tmp_path / "user/questions/1"
    write(tmp_path / "user/answers.jsonl", [answer("1")])
    write(job / "answers.jsonl", [answer("1")])
    write(job / "labels.jsonl", [label("1", "JUDGE_ERROR")])

    def retries(_pipeline, items, out_dir, **kwargs):
        assert read_jsonl(out_dir / "answers.jsonl") == [answer("1")]
        assert not read_jsonl(out_dir / "labels.jsonl")
        write(out_dir / "labels.jsonl", [label("1")])

    monkeypatch.setattr(module, "run_judge", retries)
    errors = []
    queue = module.StreamingOfficialJudge(
        tmp_path, Path("fake"), workers=1, max_tokens=1024, on_error=lambda *args: errors.append(args)
    )
    queue.submit("user", {"id": "1"})
    queue.close()
    assert not errors
    assert read_jsonl(tmp_path / "user/labels.jsonl") == [label("1")]


def test_truncation_audit_is_preserved_after_failure_and_not_duplicated(tmp_path, monkeypatch):
    def fails(_pipeline, items, out_dir, **kwargs):
        (out_dir / "answer-context-audit.json").write_text(
            json.dumps({"id": "1", "truncated": True, "removed_context_characters": 30})
        )
        raise RuntimeError("temporary model outage")

    monkeypatch.setattr(module, "run_judge", fails)
    for _ in range(2):
        queue = module.StreamingOfficialJudge(
            tmp_path, Path("fake"), workers=1, max_tokens=1024, on_error=lambda *args: None
        )
        queue.submit("user", {"id": "1"})
        queue.close()
    assert read_jsonl(tmp_path / "answer-context-truncations.jsonl") == [
        {"user": "user", "id": "1", "truncated": True, "removed_context_characters": 30}
    ]


def test_deferred_queue_does_not_call_model_before_start(tmp_path, monkeypatch):
    called = threading.Event()

    def fake(_pipeline, items, out_dir, **kwargs):
        called.set()
        write(out_dir / "answers.jsonl", [answer(items[0]["id"])])
        write(out_dir / "labels.jsonl", [label(items[0]["id"])])

    monkeypatch.setattr(module, "run_judge", fake)
    queue = module.StreamingOfficialJudge(
        tmp_path, Path("fake"), workers=2, max_tokens=1024, on_error=lambda *args: None,
        deferred=True,
    )
    queue.submit("user", {"id": "1"})
    assert not called.is_set()
    assert read_jsonl(tmp_path / "user/questions/1/input.jsonl") == [{"id": "1"}]
    queue.start()
    queue.close()
    assert called.is_set() and queue.completed == 1


def test_endpoint_slot_limit_bounds_model_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFICIAL_LLM_INFLIGHT_PER_SLOT", "1")
    monkeypatch.setattr(module, "ENDPOINTS", (("stub", "key"),))
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake(_pipeline, items, out_dir, **kwargs):
        calls.append(items[0]["id"])
        entered.set()
        assert release.wait(timeout=5)
        write(out_dir / "answers.jsonl", [answer(items[0]["id"])])
        write(out_dir / "labels.jsonl", [label(items[0]["id"])])

    monkeypatch.setattr(module, "run_judge", fake)
    queue = module.StreamingOfficialJudge(
        tmp_path, Path("fake"), workers=4, max_tokens=1024, on_error=lambda *args: None,
    )
    try:
        for qid in ("1", "2", "3", "4"):
            queue.submit("user", {"id": qid})
        assert entered.wait(timeout=5)
        assert len(calls) == 1 and queue.active == 1
    finally:
        release.set()
        queue.close()
    assert queue.completed == 4 and not queue.failures
