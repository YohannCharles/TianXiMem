"""Clinical QA must use Search evidence, without leaking gold or stale answers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from eval.harness import extra_pipeline as ep
from eval.harness import official_capture_pipeline as official
from eval.jsonl_io import read_jsonl, write_line

CONFIG = ("https://answer.invalid/v1", "secret-key", "answer-model")


def _item() -> dict:
    return {
        "id": "case-1",
        "dataset": "medmemorybench",
        "category": "multi_hop_clinical_deduction",
        "question": "How do the symptoms across visits relate?",
        "retrieved_context": "[2024-01-01] Measurement was 12.\n[2024-01-03] It was 15.",
        "gold_answer": {
            "query_type": "multi_hop_clinical_deduction",
            "answers": [{"content": "GOLD_ANSWER_SENTINEL", "is_correct": True}],
            "metadata": {"reasoning_chain": ["GOLD_CHAIN_SENTINEL"]},
        },
        "original_history": "UNRETRIEVED_HISTORY_SENTINEL",
    }


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            write_line(stream, row)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    item = _item()
    source, output = tmp_path / "input.jsonl", tmp_path / "answers.jsonl"
    _write(source, [item])
    calls = []

    def fake_chat(base, key, model, prompt, *, max_tokens, timeout):
        calls.append((prompt, max_tokens))
        return "The January 1 and 3 measurements support this explanation."

    monkeypatch.setattr(ep, "_chat", fake_chat)
    monkeypatch.setattr(ep, "_config", lambda kind: CONFIG)
    args = argparse.Namespace(input=str(source), output=str(output), max_tokens=1024)
    return item, source, output, calls, args


def test_clinical_answer_request_uses_only_search_and_keeps_requested_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, source, output, calls, args = _setup(tmp_path, monkeypatch)
    assert ep.cmd_answer(args) == 0
    prompt, budget = calls[0]
    assert item["question"] in prompt and item["retrieved_context"] in prompt
    assert "supported links" in prompt and "patient-specific facts" in prompt
    assert "one short sentence" not in prompt
    assert all(s not in prompt for s in ["GOLD_ANSWER", "GOLD_CHAIN", "UNRETRIEVED_HISTORY"])
    assert budget == 1024
    record = read_jsonl(output)[0]
    assert record["answer_contract"] == ep.MMB_MCD_ANSWER_CONTRACT
    assert "secret-key" not in output.read_text()

    # Same public input resumes after key rotation or judge-only gold changes.
    item["gold_answer"]["answers"] = [{"content": "DIFFERENT_GOLD", "is_correct": True}]
    _write(source, [item])
    monkeypatch.setattr(ep, "_config", lambda kind: (CONFIG[0], "rotated-key", CONFIG[2]))
    assert ep.cmd_answer(args) == 0
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["context", "question", "category", "model", "budget"])
def test_clinical_answer_resume_rejects_a_changed_request_before_calling_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    item, source, output, calls, args = _setup(tmp_path, monkeypatch)
    ep.cmd_answer(args)
    before = output.read_bytes()
    if change == "context":
        item["retrieved_context"] = "A different Search result."
    elif change == "question":
        item["question"] = "A different question?"
    elif change == "category":
        item["category"] = "entity_exact_match"
    elif change == "model":
        monkeypatch.setattr(ep, "_config", lambda kind: (*CONFIG[:2], "other-model"))
    else:
        args.max_tokens = 2000
    _write(source, [item])
    with pytest.raises(ValueError, match="new run-id"):
        ep.cmd_answer(args)
    assert len(calls) == 1
    assert output.read_bytes() == before


@pytest.mark.parametrize("changed_context", [False, True])
def test_clinical_evaluate_rejects_legacy_or_changed_answers_before_judging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_context: bool
) -> None:
    item, source, answers, calls, args = _setup(tmp_path, monkeypatch)
    if changed_context:
        ep.cmd_answer(args)
        item["retrieved_context"] += " Changed evidence."
        _write(source, [item])
    else:
        _write(answers, [{"id": item["id"], "generated_answer": "Legacy short answer."}])

    def forbidden_judge(*args, **kwargs):
        pytest.fail("stale answer reached the judge")

    monkeypatch.setattr(ep, "_judge_one", forbidden_judge)
    labels = tmp_path / "labels.jsonl"
    with pytest.raises(ValueError, match="new run-id"):
        ep.cmd_evaluate(
            argparse.Namespace(
                input=str(source), answers=str(answers), output=str(labels), max_tokens=512
            )
        )
    assert not labels.exists()


def test_clinical_evaluate_uses_recorded_answer_budget_not_the_judge_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, source, answers, calls, args = _setup(tmp_path, monkeypatch)
    args.max_tokens = 2000
    ep.cmd_answer(args)
    seen = []

    def fake_judge(dataset, qid, original, generated, *, judge, max_tokens):
        seen.append((original, generated, max_tokens))
        return {"id": qid, "is_correct": True, "label": "CORRECT", "judge_response": "ok"}

    monkeypatch.setattr(ep, "_judge_one", fake_judge)
    labels = tmp_path / "labels.jsonl"
    assert (
        ep.cmd_evaluate(
            argparse.Namespace(
                input=str(source), answers=str(answers), output=str(labels), max_tokens=512
            )
        )
        == 0
    )
    assert seen == [(item, read_jsonl(answers)[0]["generated_answer"], 512)]
    assert read_jsonl(labels)[0]["is_correct"]


@pytest.mark.parametrize(
    "category",
    [
        "entity_exact_match",
        "multiple_choice",
        "inference_generation",
        "state_update",
        "temporal_localization",
    ],
)
def test_other_medical_qa_formats_remain_unchanged(category: str) -> None:
    item = {**_item(), "category": category}
    expected = ep.ANSWER_PROMPT.format(
        memories=item["retrieved_context"], question=item["question"]
    )
    assert ep.render_answer_prompt(item) == expected


def test_captured_medical_query_does_not_infer_answer_format_from_gold() -> None:
    item = _item()
    item.pop("category")
    item["judge_kind"] = "medmemorybench"
    assert official._answer_prompt(item) == ep.ANSWER_PROMPT.format(
        memories=item["retrieved_context"], question=item["question"]
    )


def test_legacy_nonclinical_answers_can_still_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, source, output, calls, args = _setup(tmp_path, monkeypatch)
    item["category"] = "entity_exact_match"
    _write(source, [item])
    _write(output, [{"id": item["id"], "generated_answer": "12"}])
    assert ep.cmd_answer(args) == 0
    assert not calls


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            '{"node_validations": [{"note": "No matching measurement.”}], '
            '"is_correct": false, "reason": "Missing evidence."}',
            False,
        ),
        (
            '{"node_validations": [{"note": \'Matched "12 mg".\'}], '
            '"is_correct": true, "reason": "All nodes matched."}',
            True,
        ),
        (
            '{"note": \'Quote: {"is_correct": true}\', "is_correct": false}',
            False,
        ),
        (
            '{"note": \'Quote: {"is_correct": true}\', "is_correct": 1}',
            None,
        ),
        (
            '{"is_correct": "unknown", "node_validations": [{"is_correct": true}]}',
            None,
        ),
        (
            '{"node_validations": [{"note": "Incomplete.”}], "is_correct": true',
            None,
        ),
    ],
)
def test_mixed_judge_string_quotes_preserve_the_explicit_verdict(raw, expected) -> None:
    actual, why = ep.judge_mmb_llm_verdict(raw)
    assert actual is expected
    if expected is not None:
        assert "规范化" in why
