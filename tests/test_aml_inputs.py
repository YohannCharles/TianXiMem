"""AML 的输入边界与历史时点：真实 HTTP 客户端、离线语料和中断续跑。"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from eval.datasets import corporatebench, memtrapbench, mquake
from eval.datasets.aml import feverous as aml_feverous
from eval.datasets.aml import load_plans, required_datasets, validate_options
from eval.datasets.aml.common import checkpoint_plans
from eval.datasets.aml.plan import AddEvent, InputPlan, SearchEvent
from eval.datasets.aml.registry import SUPPORTED
from eval.datasets.layout import local_path
from eval.datasets.preprocess import Message, Question, Sample, Session
from eval.datasets.registry import _sources, benchmark_dir
from eval.experiments import run as runner
from eval.experiments.recipes import flags_for, recipe_for
from eval.harness import SearchHit, ServiceClient, build_input_items
from eval.harness import corpusqa_pipeline as pipeline
from eval.harness.plan_driver import (
    PlanStateError,
    compile_plan,
    execute_plan,
    freeze_run,
    send_event,
)
from tests.test_corpusqa_datasets import fever_root as fever_root
from tests.test_corpusqa_datasets import hybrid_root as hybrid_root
from tests.test_corpusqa_datasets import scoring_sources as scoring_sources
from tests.test_memoryqa_datasets import bench as bench


@pytest.fixture
def memory_root(bench: Path) -> Path:
    return bench


def _json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _messages(plan: InputPlan) -> list[dict]:
    return [m for e in plan.events if isinstance(e, AddEvent) for m in e.messages]


def _timeline() -> InputPlan:
    questions = (
        Question("early", "old?", "GOLD_OLD", "fact"),
        Question("late", "new?", "GOLD_NEW", "fact"),
    )
    return InputPlan(
        Sample("source-user", "halumem", (), questions),
        (
            AddEvent("history", ({"role": "user", "content": "user: PAST"},)),
            SearchEvent("early", "old?"),
            AddEvent("history", ({"role": "user", "content": "user: FUTURE"},)),
            SearchEvent("late", "new?"),
        ),
    )


def _client(log: list[dict], *, fail: dict | None = None) -> ServiceClient:
    memory: dict[str, list[str]] = {}
    applied: dict[str, dict] = {}
    fail = fail if fail is not None else {}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        log.append({"path": request.url.path, "body": body})
        if request.url.path == "/add":
            assert set(body) == {"request_id", "user_id", "session_id", "messages"}
            previous = applied.get(body["request_id"])
            if previous is not None:
                assert previous == body
            else:
                applied[body["request_id"]] = body
                memory.setdefault(body["user_id"], []).extend(
                    m["content"] for m in body["messages"]
                )
            return httpx.Response(200, json={"ok": True})
        assert request.url.path == "/search"
        assert set(body) == {"user_id", "query", "top_k"} and body["top_k"] == 100
        if fail.get(body["query"]):
            raise httpx.ReadError("interrupted Search", request=request)
        data = [
            {"id": str(i), "content": text, "created_at": "2024-01-02", "score": 1.0}
            for i, text in enumerate(memory.get(body["user_id"], []))
        ]
        return httpx.Response(200, json={"data": data[:100]})

    return ServiceClient(
        "http://aml.test", client=httpx.Client(transport=httpx.MockTransport(handle))
    )


def test_search_failure_stops_future_add_and_resume_preserves_old_snapshot(tmp_path: Path):
    plan = compile_plan(_timeline(), namespace="resume")
    log: list[dict] = []
    failure = {"old?": True}
    client = _client(log, fail=failure)
    with pytest.raises(httpx.ReadError):
        execute_plan(plan, client=client, out_dir=tmp_path)
    assert [r["path"] for r in log] == ["/add", "/search"]
    assert "FUTURE" not in json.dumps(log)
    failure.clear()
    failure["new?"] = True
    with pytest.raises(httpx.ReadError):
        execute_plan(plan, client=client, out_dir=tmp_path)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert state["next_event"] == 3
    assert [h["content"] for h in state["hits"]["early"]] == ["user: PAST"]
    failure.clear()
    log.clear()
    hits = execute_plan(plan, client=client, out_dir=tmp_path)
    assert [r["body"]["query"] for r in log] == ["new?"]
    assert [h.content for h in hits["early"]] == ["user: PAST"]
    assert [h.content for h in hits["late"]] == ["user: PAST", "user: FUTURE"]
    log.clear()
    assert execute_plan(plan, client=client, out_dir=tmp_path) == hits
    assert log == []


@pytest.mark.parametrize("damage", ["missing_hit", "extra_hit", "bad_cursor", "missing_file"])
def test_damaged_history_fails_before_http(tmp_path: Path, damage: str):
    plan = compile_plan(_timeline(), namespace="damage")
    log: list[dict] = []
    client = _client(log)
    execute_plan(plan, client=client, out_dir=tmp_path)
    path = tmp_path / "checkpoint.json"
    state = json.loads(path.read_text())
    if damage == "missing_hit":
        state["hits"].pop("early")
    elif damage == "extra_hit":
        state["hits"]["another"] = []
    elif damage == "bad_cursor":
        state["next_event"] = 500
    else:
        path.unlink()
        (tmp_path / "answers.jsonl").write_text("{}")
        log.clear()
        with pytest.raises(PlanStateError):
            execute_plan(plan, client=client, out_dir=tmp_path)
        assert log == []
        return
    _json(path, state)
    log.clear()
    with pytest.raises(PlanStateError):
        execute_plan(plan, client=client, out_dir=tmp_path)
    assert log == []


def test_run_manifest_guards_options_gold_and_deleted_user_directory(tmp_path: Path):
    plan = compile_plan(_timeline(), namespace="manifest")
    options = {"top_k": 100, "model": "offline"}
    freeze_run(tmp_path, [plan], execution=options)
    freeze_run(tmp_path, [plan], execution=options)
    with pytest.raises(PlanStateError, match="changed"):
        freeze_run(tmp_path, [plan], execution=options | {"model": "another"})
    changed = replace(
        plan,
        sample=replace(
            plan.sample,
            questions=(replace(plan.sample.questions[0], gold="changed"), plan.sample.questions[1]),
        ),
    )
    with pytest.raises(PlanStateError, match="changed"):
        freeze_run(tmp_path, [changed], execution=options)
    shutil.rmtree(tmp_path / plan.sample.user_id)
    with pytest.raises(PlanStateError, match="missing"):
        freeze_run(tmp_path, [plan], execution=options)


def test_legacy_output_cannot_become_aml_run(tmp_path: Path):
    (tmp_path / "record.json").write_text("{}")
    with pytest.raises(PlanStateError, match="legacy"):
        freeze_run(
            tmp_path, [compile_plan(_timeline(), namespace="legacy")], execution={"top_k": 100}
        )


def test_wire_identity_excludes_gold_and_changes_with_namespace_or_payload():
    original = _timeline()
    first = compile_plan(original, namespace="one")
    gold = replace(
        original,
        sample=replace(
            original.sample,
            questions=(
                replace(original.sample.questions[0], gold="another"),
                original.sample.questions[1],
            ),
        ),
    )
    same = compile_plan(gold, namespace="one")
    assert first.sample.user_id == same.sample.user_id and first.wire() == same.wire()
    assert first.fingerprint()["evaluation_sha256"] != same.fingerprint()["evaluation_sha256"]
    assert compile_plan(original, namespace="two").sample.user_id != first.sample.user_id
    changed = replace(
        original,
        events=(
            replace(original.events[0], messages=({"role": "user", "content": "user: changed"},)),
            *original.events[1:],
        ),
    )
    assert compile_plan(changed, namespace="one").sample.user_id != first.sample.user_id
    assert "GOLD" not in json.dumps(first.wire())


def test_compile_preserves_long_unicode_body_and_session_batch_order():
    body = "长段落\n" * 4000
    messages = ({"role": "user", "content": "Corpus: " + body, "timestamp": 17},) + tuple(
        {"role": "user", "content": f"Corpus: body-{i}"} for i in range(30)
    )
    original = replace(_timeline(), events=(AddEvent("history", messages), *_timeline().events[1:]))
    plan = compile_plan(original, namespace="long")
    chunks = [m for m in _messages(plan) if "长段落" in m["content"]]
    assert "".join(m["content"].removeprefix("Corpus: ") for m in chunks) == body
    assert all(len(m["content"]) <= 8000 and m["timestamp"] == 17 for m in chunks)
    adds = [e for e in plan.events if isinstance(e, AddEvent)]
    assert all(len(e.messages) <= 20 and e.batch_ready for e in adds)
    assert len({e.request_id for e in adds}) == len(adds)
    assert [e.qid for e in plan.events if isinstance(e, SearchEvent)] == ["early", "late"]


@pytest.mark.parametrize(
    "bad", [{"gold": "leak"}, {"role": "system"}, {"timestamp": True}, {"content": ""}]
)
def test_add_payload_rejects_annotations_and_invalid_wire_fields(bad: dict):
    event = AddEvent("s", ({"role": "user", "content": "visible"} | bad,))
    with pytest.raises(ValueError, match="AML plan"):
        replace(_timeline(), events=(event, *_timeline().events[1:])).validate()


def test_question_limit_keeps_add_positions_and_duplicate_searches_fail():
    original = _timeline()
    limited = original.limit_questions(1)
    assert [type(e) for e in limited.events] == [AddEvent, SearchEvent, AddEvent]
    duplicated = replace(original, events=original.events + (original.events[1],))
    with pytest.raises(ValueError, match="uniquely"):
        duplicated.validate()


def test_exact_capture_dispatch_preserves_overlong_message_and_missing_time():
    log: list[dict] = []
    event = AddEvent(
        "captured",
        ({"role": "user", "content": "Corpus: " + "x" * 9000},),
        request_id="opaque-captured-id",
        batch_ready=True,
    )
    send_event(_client(log), "original-capture-user", event)
    assert log[0]["body"] == {
        "request_id": "opaque-captured-id",
        "user_id": "original-capture-user",
        "session_id": "captured",
        "messages": list(event.messages),
    }


def test_halumem_continuous_checkpoints_never_inject_future_or_annotations(memory_root: Path):
    (plan,) = load_plans(memory_root, "halumem")
    assert [type(e) for e in plan.events] == [
        AddEvent,
        AddEvent,
        SearchEvent,
        AddEvent,
        SearchEvent,
    ]
    before = json.dumps([e.messages for e in plan.events[:2]])
    assert "Paris" in before and "FUTURE-CHANGE" not in before
    assert "FUTURE-CHANGE" in json.dumps(plan.wire())
    for marker in ("PROFILE-ONLY", "ANNOTATION-ONLY", "JUDGE-ONLY-EVIDENCE"):
        assert marker not in json.dumps(plan.wire())
    assert [q.gold["answer"] for q in plan.sample.questions] == ["Paris", "Tokyo"]
    (inline,) = load_plans(memory_root, "halumem", time_style="inline", max_questions=1)
    assert all(
        m["content"].startswith("[Time: Sep") and "timestamp" not in m for m in _messages(inline)
    )
    assert len(inline.sample.questions) == 2  # 每个原始检查点分别裁题。


def test_checkpoint_prefix_disagreement_fails():
    first = Sample(
        "a",
        "halumem",
        (Session("s", (Message("user", "original"),)),),
        (Question("q1", "q?", "a", "x"),),
    )
    second = replace(
        first,
        user_id="b",
        sessions=(
            Session("s", (Message("user", "different"),)),
            Session("later", (Message("user", "future"),)),
        ),
        questions=(Question("q2", "q?", "b", "x"),),
    )
    with pytest.raises(ValueError, match="prefixes disagree"):
        checkpoint_plans([first, second])


def test_medmemorybench_merges_only_selected_checkpoints(tmp_path: Path):
    root = tmp_path / "medmemorybench/data/zh"
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "persona_id": 1,
                    "session_id": s,
                    "turn": 1,
                    "role": "user",
                    "content": text,
                    "event_info": json.dumps({"date": "2024-01-01"}),
                }
                for s, text in [(1, "PAST"), (15, "FUTURE"), (30, "UNSELECTED-FUTURE")]
            ]
        ),
        root / "dialogues.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "persona_id": 1,
                    "session_id": s,
                    "query_id": f"q{s}",
                    "question": "Medication?",
                    "query_type": "state_update",
                    "answers": json.dumps([{"content": "GOLD_ONLY", "is_correct": True}]),
                    "metadata": json.dumps({"explanation": "ANNOTATION_ONLY"}),
                }
                for s in (10, 20)
            ]
        ),
        root / "queries.parquet",
    )
    (plan,) = load_plans(tmp_path, "medmemorybench")
    assert [type(e) for e in plan.events] == [AddEvent, SearchEvent, AddEvent, SearchEvent]
    assert "FUTURE" not in json.dumps(plan.events[0].messages)
    wire = json.dumps(plan.wire())
    assert "FUTURE" in wire
    assert not any(
        marker in wire for marker in ("UNSELECTED-FUTURE", "GOLD_ONLY", "ANNOTATION_ONLY")
    )


def test_musique_query_wrapper_and_variants_leave_gold_out(memory_root: Path):
    yes, no = load_plans(memory_root, "musique")
    assert yes.sample.user_id != no.sample.user_id
    assert (
        _messages(yes)[0]["content"] == "Corpus: [Paragraph 0]\nTitle: Alex\n\nAlex lives in Paris."
    )
    search = yes.events[-1]
    assert isinstance(search, SearchEvent) and search.query.startswith(
        yes.sample.questions[0].question + "\n"
    )
    assert "INSUFFICIENT_EVIDENCE" in search.query
    for marker in ("CHAIN-ONLY", "is_supporting", "City of Paris"):
        assert marker not in json.dumps(yes.wire())
    assert no.sample.questions[0].is_abstention


def test_hybridqa_preserves_raw_table_matrix_and_every_passage(hybrid_root: Path):
    (plan,) = load_plans(hybrid_root, "hybridqa")
    messages = _messages(plan)
    table = json.loads(messages[0]["content"].split("\n", 1)[1])
    assert table["data"][0][0] == ["Ada", ["/wiki/Ada"]]
    assert any("Unrelated passage must survive." in m["content"] for m in messages)
    assert "answer-text" not in json.dumps(plan.wire())
    assert [e.query for e in plan.events if isinstance(e, SearchEvent)] == [
        q.question for q in plan.sample.questions
    ]


def test_feverous_shared_json_pool_is_independent_of_gold_and_preserves_unicode(
    fever_root: Path, tmp_path: Path
):
    pool = tmp_path / "pool.json"
    _json(pool, {"titles": ["Alpha", "Música"], "provenance": "declared test pool"})
    (plan,) = load_plans(fever_root, "feverous", pool=pool)
    assert len(_messages(plan)) == 2
    assert json.loads(_messages(plan)[1]["content"].removeprefix("Corpus: "))["title"] == "Música"
    assert "GOLD_ONLY_TEXT" not in json.dumps(plan.wire())
    path = fever_root / "feverous" / aml_feverous.source.ANNOTATIONS
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["label"] = "REFUTES"
    rows[1]["evidence"] = [{"content": ["Secret_sentence_0"], "context": {}}]
    _jsonl(path, rows)
    (changed,) = load_plans(fever_root, "feverous", pool=pool)
    assert changed.wire() == plan.wire()
    assert changed.fingerprint()["evaluation_sha256"] != plan.fingerprint()["evaluation_sha256"]
    assert plan.events[-1].query.endswith("\n\nClaim: " + plan.sample.questions[0].question)
    assert "Do not use outside knowledge." in plan.events[-1].query


@pytest.mark.parametrize(
    "value",
    [
        {"titles": ["Alpha", "Alpha"]},
        {"titles": ["Música", "Música"]},
        {"titles": ["Alpha"], "evidence": ["leak"]},
    ],
)
def test_feverous_pool_rejects_ambiguous_or_annotation_based_lists(tmp_path: Path, value: dict):
    pool = tmp_path / "pool.json"
    _json(pool, value)
    with pytest.raises(ValueError):
        aml_feverous.corpus_titles(pool)


def test_feverous_capture_pool_reads_add_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(aml_feverous, "capture_dir", lambda: tmp_path)
    (tmp_path / "official-searches.jsonl").write_text(
        "not JSON; corpus_titles never reads Search\n"
    )
    _jsonl(tmp_path / "official-attribution.jsonl", [{"dataset": "feverous", "user_id": "shared"}])
    page = json.dumps({"title": "Alpha", "order": ["sentence_0"], "sentence_0": "text"})
    _jsonl(
        tmp_path / "official-adds.jsonl",
        [
            {
                "user_id": "shared",
                "request_id": str(i),
                "messages": [{"content": "Corpus: " + piece}],
            }
            for i, piece in enumerate((page[:25], page[25:]))
        ],
    )
    titles, provenance = aml_feverous.corpus_titles(None)
    assert titles == ["Alpha"] and provenance["adds_sha256"]
    assert not (tmp_path / "official-eval-questions.jsonl").exists()


def _captured_feverous(root: Path, capture: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    capture.mkdir(exist_ok=True)
    monkeypatch.setattr(aml_feverous, "capture_dir", lambda: capture)
    _jsonl(capture / "official-attribution.jsonl", [{"dataset": "feverous", "user_id": "shared"}])
    with aml_feverous.source._readonly(aml_feverous.source.corpus_path(root)) as wiki:
        page = wiki.execute("SELECT data FROM wiki WHERE id='Alpha'").fetchone()[0]
    _jsonl(
        capture / "official-adds.jsonl",
        [{"user_id": "shared", "request_id": "a", "messages": [{"content": "Corpus: " + page}]}],
    )
    query = aml_feverous.SEARCH_INSTRUCTION + "\n\nClaim: Alpha was   founded in 2000."
    _jsonl(
        capture / "official-searches.jsonl",
        [
            {"user_id": "another", "query": "unrelated query without any FEVEROUS wrapper"},
            {"user_id": "shared", "query": query},
            {"user_id": "shared", "query": query},
        ],
    )
    (capture / "official-eval-questions.jsonl").write_text(
        "invalid JSON; no gold/capture kit reads"
    )
    return query


def test_feverous_default_matches_same_capture_claims_without_gold_based_selection(
    fever_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    query = _captured_feverous(fever_root, tmp_path / "capture", monkeypatch)
    path = fever_root / "feverous" / aml_feverous.source.ANNOTATIONS
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(rows[1] | {"id": 2, "claim": "Other public claim excluded by query scope."})
    _jsonl(path, rows)
    (plan,) = load_plans(fever_root, "feverous")
    assert [q.qid for q in plan.sample.questions] == ["feverous-1"]
    assert plan.sample.questions[0].question == "Alpha was founded in 2000."
    assert plan.events[-1].query == query
    assert plan.metadata["query_scope"]["n_captured_requests"] == 2
    assert plan.metadata["query_scope"]["n_unique_claims"] == 1
    rows[1]["evidence"] = [{"content": ["Secret_sentence_0"], "context": {}}]
    _jsonl(path, rows)
    (changed,) = load_plans(fever_root, "feverous")
    assert changed.wire() == plan.wire()  # 即使金标已不在页面池里，也不据此换题或补页。
    assert changed.fingerprint()["evaluation_sha256"] != plan.fingerprint()["evaluation_sha256"]


@pytest.mark.parametrize(
    "damage", ["missing_public", "ambiguous_public", "wrong_wrapper", "no_user_queries"]
)
def test_feverous_capture_claim_mismatch_fails_closed(
    fever_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
):
    capture = tmp_path / "capture"
    query = _captured_feverous(fever_root, capture, monkeypatch)
    annotations = fever_root / "feverous" / aml_feverous.source.ANNOTATIONS
    if damage in {"missing_public", "ambiguous_public"}:
        rows = [json.loads(line) for line in annotations.read_text().splitlines()]
        if damage == "missing_public":
            rows[1]["claim"] = "Another claim."
        else:
            rows.append(rows[1] | {"id": 2})
        _jsonl(annotations, rows)
    else:
        _jsonl(
            capture / "official-searches.jsonl",
            [
                {
                    "user_id": "another" if damage == "no_user_queries" else "shared",
                    "query": "wrong wrapper" if damage == "wrong_wrapper" else query,
                }
            ],
        )
    with pytest.raises(ValueError, match="captured"):
        load_plans(fever_root, "feverous")


def test_json_evidence_is_only_derived_from_complete_retrieved_pages(
    fever_root: Path, tmp_path: Path
):
    pool = tmp_path / "pool.json"
    _json(pool, {"titles": ["Alpha"]})
    (plan,) = load_plans(fever_root, "feverous", pool=pool)
    page = _messages(plan)[0]["content"].removeprefix("Corpus: ")
    cut = page.index("founded") + 3  # 在 JSON 字符串内部切分，不能插入额外换行。
    context = "Q: Corpus: " + page[:cut] + "\nQ: Corpus: " + page[cut:]
    expected = {
        "Alpha_sentence_0",
        "Alpha_header_cell_0_0_0",
        "Alpha_cell_0_0_1",
        "Alpha_table_caption_0",
        "Alpha_item_0_0",
    }
    assert aml_feverous.visible_json_evidence(context) == expected
    assert not aml_feverous.visible_json_evidence("Q: Corpus: " + page[:cut])
    assert not aml_feverous.visible_json_evidence(
        "Q: Corpus: " + page[cut:] + "\nQ: Corpus: " + page[:cut]
    )
    response = json.dumps({"label": "SUPPORTS", "evidence": ["Alpha_sentence_0"]})
    assert pipeline.parse_feverous_answer(response, context, json_pages=True) == (
        "SUPPORTS",
        [["Alpha", "sentence", "0"]],
    )
    with pytest.raises(ValueError, match="outside"):
        pipeline.parse_feverous_answer(response, context)
    with pytest.raises(ValueError, match="outside"):
        pipeline.parse_feverous_answer(response, "", json_pages=True)


def test_json_feverous_scores_strict_upstream_evidence_and_keeps_gold_out_of_prompt(
    fever_root: Path, tmp_path: Path, scoring_sources
):
    scoring_sources(fever_root)
    pool = tmp_path / "pool.json"
    _json(pool, {"titles": ["Alpha"]})
    (plan,) = load_plans(fever_root, "feverous", pool=pool)
    context = _messages(plan)[0]["content"]
    item = build_input_items(plan.sample, {"feverous-1": [SearchHit("h", context, "", 1.0)]})[0]
    item.update(input_contract="aml-v1", answer_contract=aml_feverous.ANSWER_CONTRACT)
    answer = json.dumps({"label": "SUPPORTS", "evidence": ["Alpha_sentence_0"]})
    assert pipeline.score_answer(item, answer)["metrics"]["strict_score"] == 1.0
    missing = json.dumps({"label": "SUPPORTS", "evidence": []})
    metrics = pipeline.score_answer(item, missing)["metrics"]
    assert metrics["label_accuracy"] == 1.0 and metrics["strict_score"] == 0.0
    sentinel = item | {"gold_answer": {"answer": "GOLD_SENTINEL"}}
    assert pipeline.render_answer_prompt(sentinel) == pipeline.render_answer_prompt(item)
    with pytest.raises(ValueError, match="explicit AML"):
        pipeline.render_answer_prompt(item | {"input_contract": "native"})


def test_corporatebench_qa_sets_share_document_sessions_and_original_dates(tmp_path: Path):
    root = tmp_path / corporatebench.DATA_DIR
    path = root / corporatebench.KB_FILE
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("documents/a.txt", "Date: 2024-01-02\nFIRST")
        archive.writestr("documents/b.txt", "SECOND")
        archive.writestr("graph.nq", "ANNOTATION_ONLY")
    for category in corporatebench.QA_TYPES:
        _json(
            root / "data" / category / "zenith_questions.json",
            {
                "questions": [
                    {"id": "1", "question": "q?", "answer": "GOLD_ONLY", "answer_type": "str"}
                ]
            },
        )
    (plan,) = load_plans(tmp_path, "corporatebench")
    adds = [e for e in plan.events if isinstance(e, AddEvent)]
    assert len(adds) == 2 and adds[0].session_id != adds[1].session_id
    assert adds[0].messages == (
        {
            "role": "user",
            "content": "document: Date: 2024-01-02\nFIRST",
            "timestamp": 1704153600000,
        },
    )
    assert "timestamp" not in adds[1].messages[0]
    assert len(plan.sample.questions) == 3
    assert "GOLD_ONLY" not in json.dumps(plan.wire()) and "ANNOTATION_ONLY" not in json.dumps(
        plan.wire()
    )


def test_memtrap_uses_source_dialogue_roles_and_final_trigger(tmp_path: Path):
    for scenario in memtrapbench.SCENARIOS:
        _json(
            tmp_path
            / memtrapbench.DATA_DIR
            / memtrapbench.DATA_SUBDIR
            / scenario
            / f"{scenario}.json",
            [
                {
                    "id": "1",
                    "context_history": [
                        {"role": "user", "content": "remember"},
                        {"role": "assistant", "content": "done"},
                    ],
                    "final_trigger": "trigger?",
                    "gold_standard": "GOLD_ONLY",
                    "test_type": "annotation",
                }
            ],
        )
    plans = load_plans(tmp_path, "memtrapbench")
    assert len(plans) == 4
    for plan in plans:
        assert [m["role"] for m in _messages(plan)] == ["user", "assistant"]
        assert [m["content"] for m in _messages(plan)] == ["user: remember", "assistant: done"]
        assert all("timestamp" in m for m in _messages(plan))
        assert plan.events[-1].query == "trigger?" and "GOLD_ONLY" not in json.dumps(plan.wire())


def test_locomo_mixes_full_lme_distractors_with_scoped_role_variant(tmp_path: Path):
    _jsonl(
        tmp_path / local_path("conversations.jsonl"),
        [
            {
                "sample_id": "conv-1",
                "speaker_a": "Alex",
                "speaker_b": "Blair",
                "sessions": [
                    {
                        "session_index": 1,
                        "date_time": "1:00 pm on 8 May, 2023",
                        "messages": [
                            {"role": "user", "text": "LO_USER"},
                            {"role": "assistant", "text": "LO_ASSISTANT"},
                        ],
                    }
                ],
            }
        ],
    )
    _jsonl(
        tmp_path / local_path("questions.jsonl"),
        [
            {
                "sample_id": "conv-1",
                "qa_id": "lo-q",
                "question": "q?",
                "category": "1",
                "answer": "GOLD_ONLY",
                "evidence": ["D1:1"],
            }
        ],
    )
    _json(
        tmp_path / local_path("lme_s_cleaned.json"),
        [
            {
                "question_id": "lme-1",
                "question": "DISTRACTOR_QUESTION",
                "question_type": "single-session-user",
                "answer": "DISTRACTOR_GOLD",
                "haystack_session_ids": ["s"],
                "haystack_dates": ["2023/05/20 (Sat) 02:21"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "LME_USER"},
                        {"role": "assistant", "content": "LME_ASSISTANT", "has_answer": True},
                    ]
                ],
            }
        ],
    )
    (normal,) = load_plans(tmp_path, "locomo-refined")
    (alluser,) = load_plans(tmp_path, "locomo-refined", add_shape="alluser")
    assert [m["role"] for m in _messages(normal)] == ["user", "assistant", "user", "assistant"]
    assert [m["role"] for m in _messages(alluser)] == ["user", "user", "user", "assistant"]
    assert [m["content"] for m in _messages(normal)] == [
        "Alex: LO_USER",
        "Blair: LO_ASSISTANT",
        "User: LME_USER",
        "Assistant: LME_ASSISTANT",
    ]
    assert normal.metadata["timeline_limit"] and required_datasets("locomo-refined") == (
        "locomo-refined",
        "longmemeval-s",
    )
    for marker in ("GOLD_ONLY", "DISTRACTOR_QUESTION", "DISTRACTOR_GOLD", "has_answer"):
        assert marker not in json.dumps(normal.wire())


def test_mquake_old_new_qa_and_new_branch_background_facts(tmp_path: Path):
    root = tmp_path / mquake.DATA_DIR / mquake.DATA_SUBDIR
    root.mkdir(parents=True)
    row = {
        "case_id": 1,
        "orig_triples_labeled": [
            ["Alex", "religion", "Catholic"],
            ["Catholic", "founder", "Jesus"],
            ["Jesus", "born in", "Galilee"],
        ],
        "orig_triples": [["A", "R", "C"], ["C", "F", "J"], ["J", "B", "G"]],
        "new_triples_labeled": [
            ["Alex", "religion", "Methodism"],
            ["Methodism", "founder", "Romain"],
            ["Romain", "born in", "Paris"],
        ],
        "new_triples": [["A", "R", "M"], ["M", "F", "Romain"], ["Romain", "B", "P"]],
        "edit_triples": [["A", "R", "M"], ["M", "F", "Romain"]],
        "requested_rewrite": [
            {
                "subject": "Alex",
                "prompt": "{} religion",
                "target_true_str": "Catholic",
                "target_true_id": "C",
                "target_new_str": "Methodism",
            },
            {
                "subject": "Methodism",
                "prompt": "{} founder",
                "target_true_str": "Wesley",
                "target_true_id": "W",
                "target_new_str": "Romain",
            },
        ],
        "questions": ["Where was Alex's religion founder born?"],
        "answer": "Galilee",
        "answer_alias": [],
        "new_answer": "Paris",
        "new_answer_alias": [],
    }
    for stem in mquake.FILES:
        pq.write_table(pa.Table.from_pylist([row]), root / f"{stem}-00000-of-00001.parquet")
    (plan,) = load_plans(tmp_path, "mquake-remastered", limit=1)
    assert [type(e) for e in plan.events] == [AddEvent, SearchEvent, AddEvent, SearchEvent]
    originals = [json.loads(m["content"].removeprefix("Corpus: ")) for m in plan.events[0].messages]
    assert {
        "subject": "Romain",
        "relation": "born in",
        "object": "Paris",
        "subject_id": "Romain",
        "relation_id": "B",
        "object_id": "P",
    } in originals
    assert any(f["subject"] == "Methodism" and f["object"] == "Wesley" for f in originals)
    assert not any(f["subject"] == "Methodism" and f["object"] == "Romain" for f in originals)
    assert all(m["content"].startswith("Corpus: UPDATE: replace") for m in plan.events[2].messages)
    assert [q.gold for q in plan.sample.questions] == [["Galilee"], ["Paris"]]
    assert len(load_plans(tmp_path, "mquake-remastered", limit=1, spread=True)) == 4
    assert all("timestamp" not in m for m in _messages(plan))


@pytest.mark.parametrize("dataset", ["tempreason", "longmemeval-s", "beam"])
def test_unsupported_aml_inputs_fail_before_preparation_or_model_checks(
    dataset: str, monkeypatch: pytest.MonkeyPatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported input reached external preconditions")

    monkeypatch.setattr(runner, "ensure_dataset", forbidden)
    monkeypatch.setattr(runner, "judge_preconditions", forbidden)
    assert (
        runner.main(["--dataset", dataset, "--input-contract", "aml-v1", "--embedder", "offline"])
        == runner.EXIT_PRECONDITION_FAILED
    )


def test_cli_default_and_separate_recipes():
    args = runner.build_parser().parse_args(["--dataset", "musique", "--embedder", "offline"])
    assert args.input_contract == "native"
    assert recipe_for("mquake-remastered").n_questions == 768
    assert recipe_for("mquake-remastered", input_contract="aml-v1").n_questions == 240
    assert "--input-contract aml-v1" in flags_for("musique", input_contract="aml-v1")
    with pytest.raises(ValueError, match="inline"):
        validate_options("musique", time_style="inline")
    with pytest.raises(ValueError, match="alluser"):
        validate_options("halumem", add_shape="alluser")


def test_missing_capture_pool_fails_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(aml_feverous, "capture_dir", lambda: tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("missing declared pool reached download or paid preconditions")

    monkeypatch.setattr(runner, "ensure_dataset", forbidden)
    monkeypatch.setattr(runner, "judge_preconditions", forbidden)
    assert runner.main(["--dataset", "feverous", "--input-contract", "aml-v1"]) == 2


@pytest.mark.parametrize("dataset", sorted(SUPPORTED))
def test_frozen_aml_recipe_yields_declared_questions_when_sources_available(dataset: str):
    root = benchmark_dir({})
    for dependency in required_datasets(dataset):
        if any(not path.is_file() for path in _sources(root, dependency)):
            pytest.skip(
                f"{dependency} public sources not prepared; synthetic AML tests run separately"
            )
    if dataset == "feverous":
        try:
            aml_feverous.check_pool_available(None)
        except ValueError:
            pytest.skip("FEVEROUS frozen input needs the locally captured declared page pool")
    recipe = recipe_for(dataset, input_contract="aml-v1")
    plans = load_plans(root, dataset, limit=recipe.limit, spread=recipe.spread)
    assert sum(len(p.sample.questions) for p in plans) == recipe.n_questions
    assert len({p.sample.user_id for p in plans}) == len(plans)


def test_runner_uses_ordered_plan_cached_hits_and_original_questions(
    memory_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen = []

    def judge(path, items, out_dir, *, dataset):
        seen.extend(items)
        return []

    monkeypatch.setattr(runner, "run_judge", judge)
    log: list[dict] = []
    client = _client(log)
    options = dict(
        dataset="halumem",
        base_url=client.base_url,
        bench_dir=memory_root,
        out_dir=tmp_path / "integrated",
        client=client,
        input_contract="aml-v1",
    )
    samples, results = runner.run_round(**options)
    assert len(samples) == 1 and results == [] and len(seen) == 2
    assert "FUTURE-CHANGE" not in seen[0]["retrieved_context"]
    assert "FUTURE-CHANGE" in seen[1]["retrieved_context"]
    assert [item["question"] for item in seen] == ["Where do I live?", "Where do I live?"]
    assert all(item["input_contract"] == "aml-v1" for item in seen)
    log.clear()
    runner.run_round(**options)
    assert log == []
    manifest = json.loads((options["out_dir"] / "input-manifest.json").read_text())
    assert (
        manifest["data_fingerprint"]["n_questions"] == 2
        and manifest["execution"]["pipeline_sha256"]
    )
