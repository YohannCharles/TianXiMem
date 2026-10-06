"""PersonaMem 的 Search → 答案消息闭环与旧答案复用保护；不调用远端模型。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from eval.datasets import Message, Question, Sample, Session
from eval.harness import SearchHit, build_input_items
from eval.harness import personamem_pipeline as pipeline


def _item(context="SEARCH-ONLY-EVIDENCE"):
    return {
        "id": "q1",
        "persona_id": 7,
        "question": "What morning activity should I choose?",
        "correct_answer": "Yoga",
        "incorrect_answers": ["Running"],
        "retrieved_context": context,
        "chat_history": [{"role": "system", "content": "RAW-HISTORY-MUST-NOT-LEAK"}],
    }


@pytest.fixture
def official_stub(monkeypatch):
    seen = []

    def mcq(item):
        seen.append(item)
        return (
            [
                *item["chat_history"],
                {"role": "user", "content": item["question"]},
                {"role": "system", "content": "A. Running\nB. Yoga"},
            ],
            {"A": "Running", "B": "Yoga"},
            "B",
        )

    monkeypatch.setattr(pipeline, "_official", lambda: SimpleNamespace(official_mcq_messages=mcq))
    return seen


def test_each_question_gets_only_its_search_results():
    sample = Sample(
        user_id="pm-7",
        dataset="personamem-v2",
        sessions=(Session("history", (Message(role="user", content="UNRETRIEVED-HISTORY"),)),),
        questions=tuple(
            Question(qid=qid, question=qid, gold={}, category="preference")
            for qid in ("q1", "q2", "q3")
        ),
    )
    hits = {
        "q1": [
            SearchHit(id="a", content="Q1-FIRST", created_at="", score=1),
            SearchHit(id="b", content="Q1-SECOND", created_at="", score=0.5),
        ],
        "q2": [SearchHit(id="c", content="Q2-ONLY", created_at="", score=1)],
    }
    a, b, empty = build_input_items(sample, hits, date_mode="none")
    assert a["retrieved_context"].index("Q1-FIRST") < a["retrieved_context"].index("Q1-SECOND")
    assert "Q2-ONLY" not in str(a)
    assert "Q1-FIRST" not in str(b)
    assert empty["retrieved_context"] == ""
    assert "(no memories)" in str(empty["chat_history"])
    assert "UNRETRIEVED-HISTORY" not in str([a, b, empty])


def test_prompt_replaces_raw_history_and_keeps_options(official_stub):
    item = _item()
    messages, mapping, gold = pipeline.build_mcq_messages(item)
    contents = "\n".join(m["content"] for m in messages)
    assert "SEARCH-ONLY-EVIDENCE" in contents
    assert "RAW-HISTORY-MUST-NOT-LEAK" not in contents
    assert "A. Running\nB. Yoga" in contents
    assert mapping[gold] == "Yoga"
    assert official_stub[0]["correct_answer"] == item["correct_answer"]
    assert item["chat_history"][0]["content"] == "RAW-HISTORY-MUST-NOT-LEAK"
    assert all(m["role"] != "system" for m in messages[1:])


def test_explicit_empty_search_never_falls_back(official_stub):
    item = _item("")
    item["speaker_1_memories"] = "STALE-LEGACY-CONTEXT"
    messages, _, _ = pipeline.build_mcq_messages(item)
    contents = str(messages)
    assert "(no memories)" in contents
    assert "STALE-LEGACY-CONTEXT" not in contents
    assert "RAW-HISTORY-MUST-NOT-LEAK" not in contents


@pytest.mark.parametrize("context", [None, [], {}])
def test_invalid_search_field_is_rejected(context):
    with pytest.raises((ValueError, TypeError), match="检索记忆"):
        pipeline.retrieved_history(_item(context))


def test_missing_search_field_is_rejected():
    item = _item()
    del item["retrieved_context"]
    with pytest.raises(ValueError, match="不能使用原始 chat_history"):
        pipeline.retrieved_history(item)


def test_actual_answer_request_and_resume_guard(tmp_path, monkeypatch, official_stub):
    from eval.harness import api_config

    monkeypatch.setattr(api_config, "ANSWER_API_BASE", "https://model.invalid/v1")
    monkeypatch.setattr(api_config, "ANSWER_API_KEY", "stub")
    monkeypatch.setattr(api_config, "ANSWER_MODEL", "stub-model")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Final Answer: B"}}]})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: client(**kw, transport=httpx.MockTransport(handler))
    )
    source, output = tmp_path / "input.jsonl", tmp_path / "answers.jsonl"
    source.write_text(json.dumps(_item()) + "\n")
    args = SimpleNamespace(input=str(source), output=str(output))
    assert pipeline.cmd_answer(args) == 0
    sent = str(requests[0]["messages"])
    assert "SEARCH-ONLY-EVIDENCE" in sent and "RAW-HISTORY-MUST-NOT-LEAK" not in sent
    answer = json.loads(output.read_text())
    assert answer["option_mapping"][answer["correct_letter"]] == "Yoga"
    assert answer["memory_input_version"] == pipeline.MEMORY_INPUT_VERSION
    assert answer["input_fingerprint"]
    assert pipeline.cmd_answer(args) == 0
    assert len(requests) == 1
    source.write_text(json.dumps(_item("CHANGED-SEARCH")) + "\n")
    with pytest.raises(ValueError, match="新 run-id"):
        pipeline.cmd_answer(args)
    assert len(requests) == 1


def test_old_full_history_answers_are_not_reused(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "answers.jsonl"
    source.write_text(json.dumps(_item()) + "\n")
    output.write_text(json.dumps({"id": "q1", "generated_answer": "B"}) + "\n")
    with pytest.raises(ValueError, match="新 run-id"):
        pipeline.cmd_answer(SimpleNamespace(input=str(source), output=str(output)))
