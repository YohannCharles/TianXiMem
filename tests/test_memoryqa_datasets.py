"""HaluMem/MuSiQue 的真实输入边界：检查点、变体隔离、标注不泄入与评分接线。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest
from eval.datasets import data_fingerprint, load_halumem, load_musique, shape_note
from eval.datasets.musique import ANSWER_CONTRACT, REFUSAL
from eval.datasets.prepare import entries_for
from eval.experiments import run as runner
from eval.harness import JudgeResult, SearchHit, ServiceClient, build_input_items, pipeline_for
from eval.harness import extra_pipeline as extra
from eval.harness import official_capture_pipeline as capture_pipeline
from eval.harness.judge import run_judge


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _question(answer: str, category: str = "Basic Fact Recall") -> dict:
    return {
        "question": "Where do I live?",
        "answer": answer,
        "question_type": category,
        "difficulty": "medium",
        "evidence": [{"memory_content": "JUDGE-ONLY-EVIDENCE"}],
    }


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    sessions = [
        {
            "start_time": f"Sep {day:02d}, 2025, 18:42:18",
            "dialogue": [
                {"role": "user", "content": content, "timestamp": f"Sep {day:02d}, 2025, 18:42:18"},
                {"role": "assistant", "content": "Understood."},
            ],
            "memory_points": [{"memory_content": "ANNOTATION-ONLY"}],
            "questions": questions,
        }
        for day, content, questions in (
            (4, "I am Alex.", []),
            (5, "I live in Paris.", [_question("Paris")]),
            (6, "FUTURE-CHANGE: I moved to Tokyo.", [_question("Tokyo", "Dynamic Update")]),
        )
    ]
    _write_jsonl(
        tmp_path / "halumem" / "HaluMem-Medium.jsonl",
        [{"uuid": "alex", "persona_info": "PROFILE-ONLY", "sessions": sessions}],
    )
    rows = []
    for answerable in (True, False):
        rows.append(
            {
                "id": "2hop__shared-id",
                "question": "Where do I live?",
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": "Alex",
                        "paragraph_text": "Alex lives in Paris."
                        if answerable
                        else "Alex likes tea.",
                        "is_supporting": answerable,
                    },
                    {
                        "idx": 1,
                        "title": "Distractor",
                        "paragraph_text": "Other facts.",
                        "is_supporting": False,
                    },
                ],
                "answer": "Paris",
                "answer_aliases": ["City of Paris"],
                "answerable": answerable,
                "question_decomposition": [{"answer": "CHAIN-ONLY"}, {"answer": "CHAIN-ONLY"}],
            }
        )
    _write_jsonl(tmp_path / "musique" / "musique_full_v1.0_dev.jsonl", rows)
    return tmp_path


def test_halumem_checkpoints_preserve_history_without_future_or_annotations(bench: Path):
    early, late = load_halumem(bench)
    assert len(early.sessions) == 2 and len(late.sessions) == 3
    assert early.sessions[0] is late.sessions[0]
    assert early.user_id != late.user_id
    assert early.questions[0].qid != late.questions[0].qid
    early_text = "\n".join(m.content for s in early.sessions for m in s.messages)
    late_text = "\n".join(m.content for s in late.sessions for m in s.messages)
    assert "I live in Paris." in early_text
    assert "FUTURE-CHANGE" not in early_text and "FUTURE-CHANGE" in late_text
    for marker in ("ANNOTATION-ONLY", "PROFILE-ONLY", "JUDGE-ONLY-EVIDENCE"):
        assert marker not in late_text
    assert early.sessions[0].messages[0].to_add_payload()["timestamp"] == 1757011338000
    assert early.sessions[0].messages[1].timestamp_ms == early.sessions[0].messages[0].timestamp_ms
    assert late.questions[0].gold["answer"] == "Tokyo"


def test_halumem_spread_retains_users_and_is_deterministic(bench: Path):
    path = bench / "halumem" / "HaluMem-Medium.jsonl"
    alex = json.loads(path.read_text().strip())
    _write_jsonl(path, [alex, {**alex, "uuid": "blair"}])
    first = load_halumem(bench, limit=1, spread=True)
    second = load_halumem(bench, limit=1, spread=True)
    assert len(first) == 2  # 分层抽样每个用户至少一个检查点。
    assert {s.user_id for s in first} == {s.user_id for s in second}
    assert {s.user_id for s in first} == {"halumem-alex-s002", "halumem-blair-s002"}
    assert len(load_halumem(bench, limit=1)) == 1


def test_musique_variants_do_not_collide_or_inject_labels(bench: Path):
    yes, no = load_musique(bench)
    assert yes.user_id != no.user_id
    assert yes.questions[0].qid != no.questions[0].qid
    assert not yes.questions[0].is_abstention and no.questions[0].is_abstention
    assert yes.questions[0].evidence == ("0",) and no.questions[0].evidence == ()
    assert "Paris" in yes.sessions[0].messages[0].content
    assert "Paris" not in no.sessions[0].messages[0].content
    for sample in (yes, no):
        text = "\n".join(m.content for s in sample.sessions for m in s.messages)
        assert "CHAIN-ONLY" not in text and "is_supporting" not in text
        assert all(m.role == "user" for s in sample.sessions for m in s.messages)


def test_musique_local_answer_contract_does_not_change_capture_prompt(bench: Path):
    item = build_input_items(load_musique(bench, limit=1)[0], {})[0]
    assert item["answer_contract"] == ANSWER_CONTRACT
    native_prompt = extra.render_answer_prompt(item)
    assert REFUSAL in native_prompt
    assert "Cannot determine from the memories" not in native_prompt
    capture_item = {k: v for k, v in item.items() if k != "answer_contract"}
    capture_item["gold_answers"] = ["Paris"]
    assert capture_pipeline._answer_prompt(capture_item) == extra.ANSWER_PROMPT.format(
        memories="(no memories)", question=item["question"]
    )


def test_musique_spread_covers_hops_and_answerability(bench: Path):
    path = bench / "musique" / "musique_full_v1.0_dev.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    variants = [
        {**row, "id": f"{hops}hop__shared-id", "question_decomposition": [{}] * hops}
        for hops in (2, 3, 4)
        for row in rows
    ]
    _write_jsonl(path, variants)
    picked = load_musique(bench, limit=1, spread=True)
    assert len(picked) == 6
    assert {s.questions[0].category for s in picked} == {
        f"{hop}-hop/{variant}" for hop in (2, 3, 4) for variant in ("answerable", "unanswerable")
    }


@pytest.mark.parametrize("dataset,loader", [("halumem", load_halumem), ("musique", load_musique)])
def test_duplicate_input_ids_fail_before_sampling(bench: Path, dataset: str, loader):
    path = next((bench / dataset).glob("*.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _write_jsonl(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="重复"):
        loader(bench, limit=1)


@pytest.mark.parametrize(
    "generated,answerable,expected",
    [
        ("Paris", True, True),
        ("City of Paris.", True, True),
        ("Not Paris", True, False),
        ("Paris or Tokyo", True, False),
        ("comparison", True, False),
        (REFUSAL, True, False),
        (REFUSAL, False, True),
        (f" {REFUSAL}\n", False, True),
        (f"{REFUSAL}, maybe Paris", False, False),
        ("Cannot determine from the memories.", False, False),
        ("", False, False),
    ],
)
def test_musique_scoring_handles_aliases_and_strict_refusal(generated, answerable, expected):
    gold = {"answer": "Paris", "answer_aliases": ["City of Paris"], "answerable": answerable}
    assert extra.judge_musique(generated, gold)[0] is expected


@pytest.mark.parametrize("label", ["Correct", "Hallucination", "Omission", "invalid"])
def test_halumem_judge_keeps_three_way_labels_and_gold_out_of_answer(bench, monkeypatch, label):
    sample = load_halumem(bench, limit=1)[0]
    item = build_input_items(sample, {})[0]
    answer_prompt = capture_pipeline._answer_prompt(item)
    assert "JUDGE-ONLY-EVIDENCE" not in answer_prompt
    assert "Paris" not in answer_prompt  # 空检索不回填原始历史或金标。
    sent = []
    monkeypatch.setattr(
        capture_pipeline,
        "_upstream_template",
        lambda *args: (
            "JUDGE {question} REF {reference_answer} POINTS {key_memory_points} OUT {response}"
        ),
    )
    monkeypatch.setattr(extra, "config", lambda stage: ("http://unused", "test", "test"))

    def chat(*args, **kwargs):
        sent.append(args[3])
        return json.dumps({"evaluation_result": label})

    monkeypatch.setattr(extra, "chat", chat)
    result = capture_pipeline._judge(item, "Paris", max_tokens=128)
    assert "JUDGE-ONLY-EVIDENCE" in sent[0] and "REF Paris" in sent[0]
    assert result["is_correct"] is (label == "Correct")
    assert result["label"] == (label if label != "invalid" else "JUDGE_ERROR")


@pytest.mark.parametrize("dataset", ["halumem", "musique"])
def test_http_round_reaches_real_answer_and_judge_functions(bench, tmp_path, monkeypatch, dataset):
    memories: dict[str, list[str]] = {}
    answer_prompts = []
    monkeypatch.setattr(extra, "_config", lambda stage: ("http://unused", "test", "test"))
    monkeypatch.setattr(extra, "config", extra._config)
    monkeypatch.setattr(
        capture_pipeline,
        "_upstream_template",
        lambda *args: (
            "JUDGE {question} REF {reference_answer} POINTS {key_memory_points} OUT {response}"
        ),
    )

    def model_chat(*args, **kwargs):
        prompt = args[3]
        if prompt.startswith("JUDGE "):
            return '{"evaluation_result": "Correct"}'
        answer_prompts.append(prompt)
        if "FUTURE-CHANGE" in prompt:
            return "Tokyo"
        return "Paris" if "Paris" in prompt else REFUSAL

    monkeypatch.setattr(extra, "_chat", model_chat)
    monkeypatch.setattr(extra, "chat", model_chat)

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/add":
            memories.setdefault(body["user_id"], []).extend(m["content"] for m in body["messages"])
            return httpx.Response(200, json={})
        assert request.url.path == "/search"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "memory",
                        "score": 1.0,
                        "created_at": "2025-09-04",
                        "content": "\n".join(memories[body["user_id"]]),
                    }
                ]
            },
        )

    def run_real_functions(pipeline, items, out_dir, *, dataset):
        module = capture_pipeline if dataset == "halumem" else extra
        assert pipeline == pipeline_for(bench, dataset)
        input_path, answers, labels = (
            out_dir / name for name in ("input.jsonl", "answers.jsonl", "labels.jsonl")
        )
        _write_jsonl(input_path, items)
        module.cmd_answer(
            argparse.Namespace(input=str(input_path), output=str(answers), max_tokens=128)
        )
        module.cmd_evaluate(
            argparse.Namespace(
                input=str(input_path), answers=str(answers), output=str(labels), max_tokens=128
            )
        )
        generated = {r["id"]: r["generated_answer"] for r in module._read_jsonl(answers)}
        return [
            JudgeResult(
                qid=r["id"],
                is_correct=r["is_correct"],
                label=r["label"],
                judge_response=r["judge_response"],
                generated_answer=generated[r["id"]],
            )
            for r in module._read_jsonl(labels)
        ]

    monkeypatch.setattr(runner, "run_judge", run_real_functions)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = ServiceClient("http://test", client=http)
        samples, results = runner.run_round(
            dataset=dataset,
            base_url="http://test",
            bench_dir=bench,
            out_dir=tmp_path / "out",
            client=client,
        )
    assert len(samples) == 2 and len(results) == 2
    assert all(r.is_correct for r in results)
    assert all("ANNOTATION-ONLY" not in p and "CHAIN-ONLY" not in p for p in answer_prompts)
    if dataset == "halumem":
        assert "FUTURE-CHANGE" not in answer_prompts[0] and "FUTURE-CHANGE" in answer_prompts[1]
    else:
        assert REFUSAL in answer_prompts[0]
        assert "Cannot determine from the memories" not in answer_prompts[0]
        assert results[1].generated_answer == REFUSAL


def test_musique_evaluate_subprocess_uses_the_actual_pipeline(bench, tmp_path):
    samples = load_musique(bench)
    items = [item for s in samples for item in build_input_items(s, {})]
    input_path, answers, labels = (
        tmp_path / name for name in ("input.jsonl", "answers.jsonl", "labels.jsonl")
    )
    _write_jsonl(input_path, items)
    _write_jsonl(
        answers,
        [
            {"id": items[0]["id"], "generated_answer": "Paris"},
            {"id": items[1]["id"], "generated_answer": REFUSAL},
        ],
    )
    env = dict(os.environ) | {
        "AML_API_KEY": "test",
        "AML_BASE_URL": "http://unused",
        "AML_MODEL": "test",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(pipeline_for(bench, "musique")),
            "evaluate",
            "--input",
            str(input_path),
            "--answers",
            str(answers),
            "--output",
            str(labels),
            "--max-tokens",
            "128",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert all(row["is_correct"] for row in extra._read_jsonl(labels))


@pytest.mark.parametrize("dataset", ["halumem", "musique"])
def test_actual_pipeline_subprocesses_send_retrieval_to_a_local_model(
    bench, tmp_path, monkeypatch, dataset
):
    """两个真实子命令只调用回环地址的模型桩，验证配置注入和真实 I/O。"""
    requests = []

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            prompt = payload["messages"][0]["content"]
            if payload["model"] == "judge-test":
                content = '{"evaluation_result": "Correct"}'
            else:
                content = "Paris" if "RETRIEVED-PARIS" in prompt else REFUSAL
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = bench / ".upstream" / "halumem" / "eval_tools.py"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(
        "EVALUATION_PROMPT_FOR_QUESTION = "
        + repr("JUDGE {question} REF {reference_answer} POINTS {key_memory_points} OUT {response}"),
        encoding="utf-8",
    )
    server = HTTPServer(("127.0.0.1", 0), ModelHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    for name, value in {
        "AML_BASE_URL": base,
        "AML_API_KEY": "test",
        "AML_MODEL": "answer-test",
        "AML_JUDGE_BASE_URL": base,
        "AML_JUDGE_API_KEY": "test",
        "AML_JUDGE_MODEL": "judge-test",
        "TIANXIMEM_BENCHMARK_DIR": str(bench),
        "NO_PROXY": "127.0.0.1,localhost",
    }.items():
        monkeypatch.setenv(name, value)
    samples = runner._load(dataset, bench, 1 if dataset == "halumem" else None)
    items = []
    for sample in samples:
        question = sample.questions[0]
        hits = [] if question.is_abstention else [SearchHit("m", "RETRIEVED-PARIS", "", 1.0)]
        items.extend(build_input_items(sample, {question.qid: hits}))
    try:
        results = run_judge(
            pipeline_for(bench, dataset),
            items,
            tmp_path / "subprocess",
            dataset=dataset,
            max_tokens=128,
            timeout=30,
        )
    finally:
        server.shutdown()
        thread.join(timeout=1)
        server.server_close()
    assert len(results) == len(items) and all(r.is_correct for r in results)
    answers = [r for r in requests if r["model"] == "answer-test"]
    assert len(answers) == len(items)
    assert "RETRIEVED-PARIS" in answers[0]["messages"][0]["content"]
    assert all("JUDGE-ONLY-EVIDENCE" not in r["messages"][0]["content"] for r in answers)
    assert all(r["temperature"] == 0 and r["max_tokens"] == 128 for r in requests)
    if dataset == "halumem":
        judges = [r for r in requests if r["model"] == "judge-test"]
        assert len(judges) == 1
        assert "JUDGE-ONLY-EVIDENCE" in judges[0]["messages"][0]["content"]
    else:
        assert len(requests) == len(items)  # MuSiQue 的本地评分不调用模型。
        assert results[1].generated_answer == REFUSAL


@pytest.mark.parametrize("dataset", ["halumem", "musique"])
def test_fingerprints_and_preparation_track_selected_sources(bench, dataset):
    samples = runner._load(dataset, bench, None)
    record = data_fingerprint(
        bench,
        dataset,
        n_samples=len(samples),
        n_questions=sum(len(s.questions) for s in samples),
        note=shape_note(dataset),
    )
    assert record["files"][0]["name"].endswith(".jsonl")
    assert dataset.lower() in record["note"].lower()
    entries = entries_for(dataset)
    assert any(entry["name"].endswith(record["files"][0]["name"]) for entry in entries)
    if dataset == "halumem":
        assert any(entry["name"] == "halumem/eval_tools.py" for entry in entries)
