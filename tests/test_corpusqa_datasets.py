"""表格/事实核查管线的语料边界、证据严格分、下载失败和真实子进程接线。"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
import tarfile
import threading
import zipfile
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from eval.datasets import corpus_prepare, feverous, hybridqa, manifest, prepare
from eval.datasets.layout import archive_file, local_path
from eval.datasets.registry import _sources, benchmark_dir, data_fingerprint
from eval.experiments import run as runner
from eval.harness import JudgeResult, ServiceClient, build_input_items
from eval.harness import corpusqa_pipeline as pipeline
from eval.harness.run_record import _corpus_dataset_score, build_record


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _replace_entry(monkeypatch: pytest.MonkeyPatch, name: str, **updates) -> dict:
    entries = [entry | updates if entry["name"] == name else entry for entry in manifest.MANIFEST]
    monkeypatch.setattr(manifest, "MANIFEST", entries)
    monkeypatch.setattr(prepare, "MANIFEST", entries)
    return next(entry for entry in entries if entry["name"] == name)


@pytest.fixture
def hybrid_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "hybrid"
    rows = [
        {
            "question_id": "a",
            "table_id": "People_0",
            "question": "Who leads Alpha?",
            "answer-text": "Ada",
        },
        {
            "question_id": "b",
            "table_id": "People_0",
            "question": "Which year?",
            "answer-text": "2000",
        },
    ]
    _json(root / "hybridqa/dev.json", rows)
    _json(
        root / "hybridqa/dev_reference.json",
        {"reference": {"a": "Ada", "b": "2000"}, "table": ["b"], "passage": ["a"]},
    )
    table = {
        "title": "People",
        "header": [["Name", []], ["Year", []]],
        "data": [[["Ada", ["/wiki/Ada"]], ["2000", []]], [["Decoy", []], ["1999", []]]],
        "intro": "Complete roster",
    }
    docs = {
        "tables_tok/People_0.json": table,
        "request_tok/People_0.json": {
            "/wiki/Ada": "Ada leads Alpha.",
            "/wiki/Decoy": "Unrelated passage must survive.",
        },
    }
    archive = root / "hybridqa" / hybridqa.ARCHIVE
    prefix = next(
        e["archive_root"] for e in manifest.MANIFEST if e["name"] == "hybridqa/" + hybridqa.ARCHIVE
    )
    with tarfile.open(archive, "w:gz") as bundle:
        for name, obj in docs.items():
            raw = json.dumps(obj).encode()
            info = tarfile.TarInfo(prefix + "/" + name)
            info.size = len(raw)
            bundle.addfile(info, io.BytesIO(raw))
        # 任意附带路径不在提取白名单中。
        info = tarfile.TarInfo(prefix + "/../../escape.txt")
        info.size = 3
        bundle.addfile(info, io.BytesIO(b"bad"))
    entry = _replace_entry(
        monkeypatch,
        "hybridqa/" + hybridqa.ARCHIVE,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    corpus_prepare.prepare_hybridqa_corpus(root, entry)
    return root


def _wiki_page(name: str, sentence: str) -> dict:
    return {"title": name, "order": ["sentence_0"], "sentence_0": sentence}


@pytest.fixture
def fever_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fever"
    path = root / "feverous" / feverous.DATABASE
    path.parent.mkdir(parents=True)
    page = _wiki_page("Alpha", "Alpha was founded in 2000.")
    page["order"] += ["section_0", "table_0", "list_0"]
    page["section_0"] = {"value": "Statistics", "level": 1}
    page["table_0"] = {
        "caption": "Founders",
        "table": [
            [
                {"id": "header_cell_0_0_0", "value": "Name", "row_span": 1, "column_span": 1},
                {"id": "cell_0_0_1", "value": "Ada", "row_span": 1, "column_span": 2},
            ]
        ],
    }
    page["list_0"] = {"list": [{"id": "item_0_0", "value": "First subsidiary", "level": 0}]}
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE wiki (id TEXT PRIMARY KEY, data TEXT)")
        conn.executemany(
            "INSERT INTO wiki VALUES (?, ?)",
            [
                ("Alpha", json.dumps(page)),
                ("Secret", json.dumps(_wiki_page("Secret", "GOLD_ONLY_TEXT"))),
                ("Música", json.dumps(_wiki_page("Música", "Music magazine."))),
            ],
        )
    rows = [
        {"id": "", "claim": "", "label": "", "evidence": "", "challenge": ""},
        {
            "id": 1,
            "claim": "Alpha was founded in 2000.",
            "label": "SUPPORTS",
            "evidence": [{"content": ["Alpha_sentence_0"], "context": {}}],
            "challenge": "Other",
        },
    ]
    (path.parent / feverous.ANNOTATIONS).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _replace_entry(monkeypatch, "feverous/" + feverous.DATABASE, sha256=digest)
    feverous.prepare_index(root, digest)
    return root


@pytest.fixture
def scoring_sources(monkeypatch: pytest.MonkeyPatch):
    sources = {}
    for name in (hybridqa.SCORER, feverous.SCORER):
        source = archive_file(benchmark_dir({}), name)
        if not source.is_file():
            pytest.skip(
                "Pinned upstream scorers not downloaded; fetch HybridQA/FEVEROUS data first"
            )
        sources[name] = source.read_bytes()

    def install(root: Path) -> None:
        for name, raw in sources.items():
            destination = root / local_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        monkeypatch.setenv("TIANXIMEM_BENCHMARK_DIR", str(root))

    return install


def test_hybridqa_whole_table_all_passages_and_shared_user(hybrid_root: Path) -> None:
    samples = hybridqa.load_hybridqa(hybrid_root)
    assert len(samples) == 1
    assert [q.qid for q in samples[0].questions] == ["hybridqa-a", "hybridqa-b"]
    text = "\n".join(m.content for m in samples[0].sessions[0].messages)
    assert "Decoy" in text and "Unrelated passage must survive." in text
    assert "Who leads Alpha?" not in text and "answer-text" not in text
    assert "2000" in text  # 原始表格含答案值是合法语料。
    assert not (hybrid_root.parent / "escape.txt").exists()
    assert hybridqa.load_hybridqa(hybrid_root, limit=0) == []
    assert len(hybridqa.load_hybridqa(hybrid_root, limit=1, spread=True)[0].questions) == 2


def test_hybridqa_rejects_duplicates_before_limit(hybrid_root: Path) -> None:
    path = hybrid_root / "hybridqa/dev.json"
    rows = json.loads(path.read_text())
    _json(path, rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        hybridqa.load_hybridqa(hybrid_root, limit=1)


def test_hybridqa_corruption_is_preserved_and_check_does_not_repair(hybrid_root: Path) -> None:
    entry = next(e for e in manifest.MANIFEST if e["name"] == "hybridqa/" + hybridqa.ARCHIVE)
    path = hybrid_root / "hybridqa/tables_tok/People_0.json"
    path.write_text("user-edited")
    with pytest.raises(ValueError, match="sha256"):
        corpus_prepare.prepare_hybridqa_corpus(hybrid_root, entry)
    assert path.read_text() == "user-edited"
    path.unlink()
    with pytest.raises(ValueError, match="Missing"):
        corpus_prepare.prepare_hybridqa_corpus(hybrid_root, entry, check_only=True)
    assert not path.exists()
    corpus_prepare.prepare_hybridqa_corpus(hybrid_root, entry)
    assert path.is_file()


def test_feverous_candidates_do_not_use_gold_and_render_all_element_types(fever_root: Path) -> None:
    samples = feverous.load_feverous(fever_root)
    before = samples[0].sessions
    content = "\n".join(m.content for s in before for m in s.messages)
    for marker in (
        "Alpha_sentence_0",
        "Alpha_cell_0_0_1",
        "Alpha_header_cell_0_0_0",
        "Alpha_table_caption_0",
        "Alpha_item_0_0",
    ):
        assert marker in content
    assert "GOLD_ONLY_TEXT" not in content
    path = fever_root / "feverous" / feverous.ANNOTATIONS
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["label"] = "REFUTES"
    rows[1]["evidence"] = [{"content": ["Secret_sentence_0"], "context": {}}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert feverous.load_feverous(fever_root)[0].sessions == before
    with feverous._readonly(fever_root / "feverous" / feverous.INDEX) as index:
        assert feverous.candidate_pages(index, "Música") == ["Música"]


@pytest.mark.parametrize(
    "element_id,expected",
    [
        (
            "Page_with_underscores_header_cell_0_1_2",
            ["Page_with_underscores", "header_cell", "0_1_2"],
        ),
        ("Page_cell_2_table_caption_3", ["Page_cell_2", "table_caption", "3"]),
    ],
)
def test_feverous_ids_parse_from_the_element_suffix(element_id: str, expected: list[str]) -> None:
    assert feverous.evidence_triplet(element_id) == expected


def test_feverous_candidate_rank_keeps_lexical_ties_at_page_limit(fever_root: Path) -> None:
    with sqlite3.connect(fever_root / "feverous" / feverous.INDEX) as index:
        index.execute("DELETE FROM pages")
        index.executemany(
            "INSERT INTO pages VALUES (?, ?, ?)",
            [(f"Page_{i}", "Alpha", "Identical text") for i in range(8, -1, -1)],
        )
        expected = [
            row[0]
            for row in index.execute(
                "SELECT page FROM pages WHERE pages MATCH 'alpha' "
                "ORDER BY bm25(pages, 0, 8, 1), page LIMIT 5"
            )
        ]
        assert (
            feverous.candidate_pages(index, "Alpha") == expected == [f"Page_{i}" for i in range(5)]
        )


def test_feverous_missing_stale_index_and_duplicates_fail_before_requests(fever_root: Path) -> None:
    path = fever_root / "feverous" / feverous.ANNOTATIONS
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(row) for row in rows + rows[1:]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        feverous.load_feverous(fever_root, limit=1)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    (fever_root / "feverous" / feverous.INDEX).unlink()
    with pytest.raises(ValueError, match="index"):
        feverous.load_feverous(fever_root)
    assert feverous.load_feverous(fever_root, limit=0) == []
    with pytest.raises(ValueError, match="index"):
        feverous.prepare_index(fever_root, feverous.source_sha256(), check_only=True)


@pytest.mark.parametrize(
    "answer",
    [
        '{"label":"SUPPORTS","evidence":["Hidden_sentence_0"]}',
        "SUPPORTS",
        '{"label":"SUPPORTS","evidence":"Alpha_sentence_0"}',
        '{"label":"YES","evidence":[]}',
    ],
)
def test_feverous_answer_cannot_cite_invisible_or_invalid_evidence(answer: str) -> None:
    with pytest.raises(ValueError):
        pipeline.parse_feverous_answer(answer, "[Alpha_sentence_0] text")


def test_search_empty_input_never_backfills_source_or_gold(
    hybrid_root: Path, fever_root: Path
) -> None:
    for loader, root in (
        (hybridqa.load_hybridqa, hybrid_root),
        (feverous.load_feverous, fever_root),
    ):
        sample = loader(root, limit=1)[0]
        item = build_input_items(sample, {})[0]
        assert item["retrieved_context"] == ""
        prompt = pipeline.render_answer_prompt(item)
        assert "(no memories)" in prompt
        assert "GOLD_ONLY_TEXT" not in prompt
        assert "[Alpha_sentence_0]" not in prompt
        changed = copy.deepcopy(item)
        changed["gold_answer"] = {"answer": "GOLD_ONLY_TEXT", "label": "REFUTES", "evidence": []}
        assert pipeline.render_answer_prompt(changed) == prompt
        changed.pop("answer_contract")
        with pytest.raises(ValueError, match="contract"):
            pipeline.render_answer_prompt(changed)


def test_zip_extraction_download_hash_crc_and_atomic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"SQLite source bytes"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("original.db", body)
        bundle.writestr("../../escape", "bad")
    zipped = archive.getvalue()
    entry = {
        "name": "feverous/test.db",
        "tier": "official-extra",
        "url": "https://example.invalid/wiki.zip",
        "archive_member": "original.db",
        "bytes": str(len(body)),
        "download_bytes": str(len(zipped)),
        "sha256": hashlib.sha256(body).hexdigest(),
        "download_sha256": hashlib.sha256(zipped).hexdigest(),
    }
    monkeypatch.setattr(prepare, "_download", lambda url, path: path.write_bytes(zipped))
    assert prepare._prepare_entry(entry, tmp_path, offline=False)
    assert (tmp_path / entry["name"]).read_bytes() == body
    assert not (tmp_path.parent / "escape").exists()
    assert not list((tmp_path / ".tmp").iterdir())
    target = tmp_path / entry["name"]
    target.unlink()
    monkeypatch.setattr(prepare, "_download", lambda url, path: path.write_bytes(zipped[:-1]))
    with pytest.raises(prepare.PreparationError, match="压缩包"):
        prepare._prepare_entry(entry, tmp_path, offline=False)
    assert not target.exists()
    assert not list((tmp_path / ".tmp").iterdir())


def test_judge_selection_never_downloads_large_corpora_and_fingerprints_include_inputs(
    hybrid_root: Path, fever_root: Path
) -> None:
    for dataset in ("hybridqa", "feverous"):
        selected = prepare.entries_for(dataset, purpose="judge")
        assert selected and all(local_path(e["name"]).parts[0] == ".upstream" for e in selected)
        assert not any(e["name"].endswith((".db", ".tar.gz")) for e in selected)
    assert hybrid_root / "hybridqa" / hybridqa.CORPUS_RECEIPT in _sources(hybrid_root, "hybridqa")
    assert fever_root / "feverous" / feverous.INDEX in _sources(fever_root, "feverous")


@pytest.mark.parametrize("label", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
def test_upstream_feverous_requires_an_entire_evidence_group_for_every_label(
    fever_root: Path, scoring_sources, label: str
) -> None:
    scoring_sources(fever_root)
    item = build_input_items(feverous.load_feverous(fever_root)[0], {})[0]
    item["retrieved_context"] = (
        "[Alpha_sentence_0] founded\n[Alpha_cell_0_0_1] Ada\n[Alpha_sentence_1] alternate"
    )
    item["gold_answer"] = {
        "label": label,
        "evidence": [
            [["Alpha", "sentence", "0"], ["Alpha", "cell", "0_0_1"]],
            [["Alpha", "sentence", "1"]],
        ],
    }
    missing = pipeline.score_answer(
        item, json.dumps({"label": label, "evidence": ["Alpha_sentence_0"]})
    )
    assert not missing["is_correct"] and missing["metrics"]["label_accuracy"] == 1
    assert missing["metrics"]["evidence_recall"] == 0
    complete = pipeline.score_answer(
        item, json.dumps({"label": label, "evidence": ["Alpha_sentence_1"]})
    )
    assert complete["is_correct"] and complete["metrics"]["strict_score"] == 1


def test_upstream_hybridqa_token_f1_is_separate_from_exact_match(
    hybrid_root: Path, scoring_sources
) -> None:
    scoring_sources(hybrid_root)
    item = build_input_items(hybridqa.load_hybridqa(hybrid_root, limit=1)[0], {})[0]
    item["gold_answer"] = {"answer": "The Ada Lovelace"}
    partial = pipeline.score_answer(item, "Ada")
    assert not partial["is_correct"] and partial["metrics"]["f1"] == pytest.approx(2 / 3)
    exact = pipeline.score_answer(item, "ada lovelace.")
    assert exact["is_correct"] and exact["metrics"]["f1"] == 1


def test_feverous_f1_aggregates_macro_precision_recall_and_handles_zero() -> None:
    rows = [
        JudgeResult(
            str(i),
            False,
            "WRONG",
            "",
            "",
            metrics={
                "strict_score": 0,
                "label_accuracy": 1,
                "evidence_precision": p,
                "evidence_recall": r,
            },
        )
        for i, (p, r) in enumerate(((1.0, 0.0), (0.5, 1.0)))
    ]
    scores = _corpus_dataset_score("feverous", rows)
    assert scores["evidence_f1"] == pytest.approx(0.6)
    assert scores["mean"] == 0 and scores["label_accuracy"] == 1
    zero = JudgeResult(
        "z",
        False,
        "",
        "",
        "",
        metrics={
            "strict_score": 0,
            "label_accuracy": 0,
            "evidence_precision": 0,
            "evidence_recall": 0,
        },
    )
    assert _corpus_dataset_score("feverous", [zero])["evidence_f1"] == 0


@pytest.mark.parametrize("dataset", ["hybridqa", "feverous"])
def test_real_runner_answer_and_evaluate_subprocess_use_only_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hybrid_root: Path,
    fever_root: Path,
    scoring_sources,
    dataset: str,
) -> None:
    root = hybrid_root if dataset == "hybridqa" else fever_root
    scoring_sources(root)
    prompts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompts.append(payload["messages"][0]["content"])
            answer = (
                "Ada"
                if dataset == "hybridqa"
                else json.dumps({"label": "SUPPORTS", "evidence": ["Alpha_sentence_0"]})
            )
            body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AML_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AML_API_KEY", "local-stub")
    monkeypatch.setenv("AML_MODEL", "local-stub")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    adds = []

    def service(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/add":
            adds.append(payload)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    **{key: payload[key] for key in ("request_id", "user_id", "session_id")},
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "hit",
                        "content": (
                            "[Alpha_sentence_0] SEARCH_ONLY Alpha founded in 2000; Ada leads it."
                        ),
                        "created_at": "",
                        "score": 1.0,
                    }
                ],
            },
        )

    try:
        with httpx.Client(transport=httpx.MockTransport(service)) as http:
            client = ServiceClient("http://service.invalid", client=http)
            samples, results = runner.run_round(
                dataset=dataset,
                base_url="http://service.invalid",
                bench_dir=root,
                out_dir=tmp_path / "run",
                limit=1,
                client=client,
            )
        assert adds and len(results) == 1 and results[0].is_correct
        assert len(prompts) == 1 and "SEARCH_ONLY" in prompts[0]
        assert "Unrelated passage must survive." not in prompts[0]
        assert "GOLD_ONLY_TEXT" not in prompts[0]
        record = build_record(
            run_id="test",
            step="test",
            profile="local",
            bench_dir=root,
            samples=samples,
            results=results,
            data_fingerprint=data_fingerprint(
                root, dataset, n_samples=len(samples), n_questions=len(results)
            ),
            models={"embedder": "stub", "llm": "stub", "reranker": "stub"},
        )
        assert record.scores["dataset_score"]["mean"] == 1
        assert record.scores["overall"] == 1
        answers = tmp_path / "run" / samples[0].user_id / "answers.jsonl"
        assert "input_fingerprint" in json.loads(answers.read_text())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_changed_search_input_rejects_stale_answers_before_model_calls(
    hybrid_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = build_input_items(hybridqa.load_hybridqa(hybrid_root, limit=1)[0], {})[0]
    input_path = hybrid_root / "input.jsonl"
    answers = hybrid_root / "answers.jsonl"
    labels = hybrid_root / "labels.jsonl"
    input_path.write_text(json.dumps(item) + "\n")
    calls = []
    monkeypatch.setattr(pipeline, "_config", lambda which: ("local", "local", "local"))

    def chat(*args, **kwargs):
        calls.append(args)
        return "Ada"

    monkeypatch.setattr(pipeline, "_chat", chat)
    args = Namespace(input=str(input_path), output=str(answers), max_tokens=128)
    pipeline.cmd_answer(args)
    pipeline.cmd_answer(args)
    assert len(calls) == 1
    original = answers.read_bytes()
    item["retrieved_context"] = "Different Search result"
    input_path.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="another input"):
        pipeline.cmd_answer(args)
    assert len(calls) == 1 and answers.read_bytes() == original
    labels.write_text("previous labels")
    with pytest.raises(ValueError, match="fingerprint"):
        pipeline.cmd_evaluate(
            Namespace(input=str(input_path), answers=str(answers), output=str(labels))
        )
    assert labels.read_text() == "previous labels"


@pytest.mark.parametrize(
    "invalid",
    [
        '{"label":"SUPPORTS","evidence":["Page_sentence_0"]}',
        "not JSON",
        '{"label":"YES","evidence":[]}',
    ],
)
def test_feverous_one_repair_uses_same_search_without_gold_or_scoring_feedback(
    fever_root: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    item = build_input_items(feverous.load_feverous(fever_root)[0], {})[0]
    item["retrieved_context"] = "[Alpha_sentence_0] SEARCH_ONLY Alpha founded in 2000."
    valid = '{"label":"SUPPORTS","evidence":["Alpha_sentence_0"]}'
    prompts = []
    replies = iter([invalid, valid, invalid, valid])

    def chat(base, key, model, prompt, **kwargs):
        prompts.append(prompt)
        return next(replies)

    def forbidden_scorer(dataset):
        pytest.fail("Answer repair must not load scoring functions")

    monkeypatch.setattr(pipeline, "_chat", chat)
    monkeypatch.setattr(pipeline, "_scorer", forbidden_scorer)
    result = pipeline.generate_answer(item, "local", "local", "local", 128)
    changed = copy.deepcopy(item)
    changed["gold_answer"] = {
        "label": "REFUTES",
        "evidence": [{"content": ["Secret_sentence_0"]}],
        "text": "GOLD_ONLY_TEXT",
    }
    assert pipeline.generate_answer(changed, "local", "local", "local", 128) == result
    assert prompts[:2] == prompts[2:]
    assert result["generated_answer"] == valid
    assert result["answer_attempts"][0]["generated_answer"] == invalid
    assert result["answer_attempts"][0]["validation_error"]
    assert "validation_error" not in result["answer_attempts"][1]
    assert invalid in prompts[1] and "SEARCH_ONLY" in prompts[1]
    assert all("GOLD_ONLY_TEXT" not in p and "Secret_sentence_0" not in p for p in prompts)


def test_feverous_valid_output_does_not_retry(
    fever_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = build_input_items(feverous.load_feverous(fever_root)[0], {})[0]
    calls = []
    valid = '{"label":"NOT ENOUGH INFO","evidence":[]}'

    def chat(*args, **kwargs):
        calls.append(args)
        return valid

    monkeypatch.setattr(pipeline, "_chat", chat)
    result = pipeline.generate_answer(item, "local", "local", "local", 128)
    assert len(calls) == 1 and result["generated_answer"] == valid
    assert result["answer_attempts"] == [{"generated_answer": valid}]
    assert "Page_sentence_0" not in pipeline.render_answer_prompt(item)
    item["answer_contract"] = "feverous-claim-pages-evidence-v1"
    with pytest.raises(ValueError, match="contract"):
        pipeline.render_answer_prompt(item)


def test_feverous_failed_repair_stays_invalid_and_scores_zero(
    fever_root: Path, monkeypatch: pytest.MonkeyPatch, scoring_sources
) -> None:
    scoring_sources(fever_root)
    item = build_input_items(feverous.load_feverous(fever_root)[0], {})[0]
    item["retrieved_context"] = "[Alpha_sentence_0] SEARCH_ONLY"
    invalid = '{"label":"SUPPORTS","evidence":["Hidden_sentence_0"]}'
    calls = []

    def chat(*args, **kwargs):
        calls.append(args)
        return invalid

    monkeypatch.setattr(pipeline, "_chat", chat)
    result = pipeline.generate_answer(item, "local", "local", "local", 128)
    assert len(calls) == len(result["answer_attempts"]) == 2
    assert result["generated_answer"] == invalid
    assert all(attempt["validation_error"] for attempt in result["answer_attempts"])
    score = pipeline.score_answer(item, result["generated_answer"])
    assert score["label"] == "ANSWER_ERROR" and not score["is_correct"]
    assert all(value == 0 for value in score["metrics"].values())


def test_feverous_repair_guide_only_lists_visible_spellings_without_replacing_answer(
    fever_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = build_input_items(feverous.load_feverous(fever_root)[0], {})[0]
    item["retrieved_context"] = (
        "[Alpha_header_cell_0_6_0] Position\n[Alpha_cell_0_6_1] Director\n"
        "[Alpha_cell_0_6_2] 2000\n[Alpha_cell_0_6_3] 2001"
    )
    invalid = '{"label":"SUPPORTS","evidence":["Alpha_cell_0_6_0"]}'
    guide = pipeline.repair_id_guide(invalid, item["retrieved_context"])
    assert "Alpha_header_cell_0_6_0" in guide
    assert set(guide) <= pipeline.visible_evidence(item["retrieved_context"])
    assert "Alpha_cell_0_6_0" not in guide
    assert pipeline.repair_id_guide(invalid, "") == []
    assert pipeline.repair_id_guide("not JSON", item["retrieved_context"]) == []
    prompts = []

    def chat(base, key, model, prompt, **kwargs):
        prompts.append(prompt)
        return invalid

    monkeypatch.setattr(pipeline, "_chat", chat)
    result = pipeline.generate_answer(item, "local", "local", "local", 128)
    assert len(prompts) == 2 and "Visible IDs with similar spellings" in prompts[1]
    assert "Alpha_header_cell_0_6_0" in prompts[1]
    assert result["generated_answer"] == invalid
