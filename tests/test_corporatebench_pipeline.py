"""CorporateBench 的数值/日期误判、空集合、真正 set-F1 和完整 harness 接线。"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.datasets.preprocess import Question, Sample
from eval.harness.corporatebench_pipeline import (
    build_answer_prompt,
    judge_corporatebench,
    score_corporatebench,
)
from eval.harness.run_record import build_record


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("1", 1),
        ("1,000", 1000),
        ("There was one occurrence, on 2024-02-06.", 1),
        ("The meeting took place once, on March 5, 2024.", 1),
        ("It took place once in January 2024, on January 24, 2024.", 1),
        ("The meeting took place once, on February 6, 2024.", 1),
        ("The meeting happened twice in 2024.", 2),
        ("There were 3 occurrences during 2024.", 3),
        ("No meetings took place.", 0),
        ('{"answer": 5}', 5),
    ],
)
def test_counts_use_the_count_rather_than_the_date(generated: str, expected: int) -> None:
    assert judge_corporatebench(generated, {"answer": expected, "answer_type": "int"})[0]


@pytest.mark.parametrize(
    "generated",
    ["2024-02-06", "It happened on March 5, 2024.", "true", "null", "NaN", "1 or 2"],
)
def test_dates_ambiguity_and_other_json_types_are_not_counts(generated: str) -> None:
    assert not judge_corporatebench(generated, {"answer": 1, "answer_type": "int"})[0]


@pytest.mark.parametrize("generated", ["[]", "```json\n[]\n```", '{"answer": []}', "None."])
def test_explicit_empty_set_is_a_valid_answer(generated: str) -> None:
    assert score_corporatebench(generated, {"answer": [], "answer_type": "List[str]"})[0] == 1


@pytest.mark.parametrize(
    "generated",
    ["Cannot determine from the memories.", "", "null", '"unknown"', '[""]'],
)
def test_refusal_is_not_credited_as_an_empty_set(generated: str) -> None:
    assert score_corporatebench(generated, {"answer": [], "answer_type": "List[str]"})[0] == 0


@pytest.mark.parametrize(
    ("generated", "expected_score"),
    [
        ('["Alpha"]', 2 / 3),
        ('["Alpha", "Beta", "Extra"]', 4 / 5),
        ("Alpha, Beta, and Extra.", 4 / 5),
        ('["Beta", "Alpha", "Alpha"]', 1),
        ('["Other"]', 0),
        ("[]", 0),
    ],
)
def test_set_f1_penalizes_both_missing_and_extra_predictions(
    generated: str,
    expected_score: float,
) -> None:
    gold = {"answer": ["Alpha", "Beta"], "answer_type": "List[str]"}
    score, _ = score_corporatebench(generated, gold)
    assert score == pytest.approx(expected_score)
    assert judge_corporatebench(generated, gold)[0] == (expected_score == 1)


@pytest.mark.parametrize(
    ("generated", "wanted"),
    [
        ("Alice Smith and Bob Jones attended the review meeting.", ["Alice Smith", "Bob Jones"]),
        (
            "The employees are Alice Smith, Bob Jones, and Carol White.",
            ["Alice Smith", "Bob Jones", "Carol White"],
        ),
        (
            "Meetings:\n- Quality and Standards Review (2024-02-01)\n"
            "- Quality and Standards Review (2024-03-01)",
            ["Quality and Standards Review"],
        ),
        (
            "1. **Alpha Review** (2024-02-01) - Details.\n2. **Extra Review** (2024-02-02)",
            ["Alpha Review", "Extra Review"],
        ),
        (
            '["Alice & Bob Check-in", "Quality and Standards Review"]',
            ["Alice & Bob Check-in", "Quality and Standards Review"],
        ),
    ],
)
def test_names_titles_and_duplicate_events_are_parsed_as_sets(
    generated: str,
    wanted: list[str],
) -> None:
    assert score_corporatebench(generated, {"answer": wanted, "answer_type": "List[str]"})[0] == 1


@pytest.mark.parametrize(
    ("generated", "gold"),
    [
        ('"Alpha plus Extra"', {"answer": "Alpha", "answer_type": "str"}),
        ("true", {"answer": "Alpha", "answer_type": "str"}),
        ('"not true"', {"answer": True, "answer_type": "bool"}),
        ('"Yes, false"', {"answer": True, "answer_type": "bool"}),
    ],
)
def test_structured_scalar_types_and_contradictions_do_not_pass(
    generated: str,
    gold: dict,
) -> None:
    assert score_corporatebench(generated, gold)[0] == 0


def test_prompt_uses_only_type_metadata_and_never_gold_values() -> None:
    item = {
        "question": "Which employees attended?",
        "retrieved_context": "Alice attended.",
        "gold_answer": {"answer_type": "List[str]", "answer": ["SECRET_GOLD_VALUE"]},
    }
    prompt = build_answer_prompt(item)
    assert "SECRET_GOLD_VALUE" not in prompt
    assert "JSON array" in prompt and "every distinct" in prompt
    assert "Missing a fact" in prompt
    assert "Alice attended." in prompt
    assert prompt == build_answer_prompt(
        item
        | {
            "gold_answer": {
                "answer_type": "List[str]",
                "answer": ["DIFFERENT_SECRET"],
            }
        }
    )


def test_count_rules_do_not_change_the_single_fact_prompt() -> None:
    item = {"question": "What?", "retrieved_context": "A fact."}
    counts = build_answer_prompt(item | {"answer_type": "int"})
    scalar = build_answer_prompt(item | {"answer_type": "str"})
    assert "same event count once" in counts
    assert "JSON integer" in counts
    assert "JSON string" in scalar
    assert "complete coverage" not in scalar


def test_scalar_queries_keep_the_validated_lookup_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from eval.harness import extra_pipeline as ep

    prompts = []

    def fake_chat(base, key, model, prompt, *, max_tokens, timeout):  # noqa: ARG001
        prompts.append(prompt)
        return "Alice"

    item = {
        "id": "scalar",
        "dataset": "corporatebench",
        "question": "Who organized?",
        "retrieved_context": "Alice organized.",
        "gold_answer": {"answer_type": "str", "answer": "Alice"},
    }
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(item) + "\n")
    monkeypatch.setattr(ep, "_config", lambda _: ("unused", "unused", "unused"))
    monkeypatch.setattr(ep, "_chat", fake_chat)
    args = ep.build_parser().parse_args(
        [
            "answer",
            "--input",
            str(path),
            "--output",
            str(tmp_path / "answers.jsonl"),
        ]
    )
    assert ep.cmd_answer(args) == 0
    assert prompts == [
        ep.ANSWER_PROMPT.format(memories=item["retrieved_context"], question=item["question"])
    ]


def test_partial_credit_flows_through_answer_evaluate_harness_and_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eval.harness import extra_pipeline as ep
    from eval.harness import judge as harness

    seen_prompts = []

    def fake_chat(base, key, model, prompt, *, max_tokens, timeout):  # noqa: ARG001
        seen_prompts.append(prompt)
        return '["Alpha", "Beta", "Extra"]'

    monkeypatch.setattr(ep, "_config", lambda _: ("unused", "unused", "unused"))
    monkeypatch.setattr(ep, "_chat", fake_chat)

    def fake_run(pipeline, args, *, timeout):  # noqa: ARG001
        parsed = ep.build_parser().parse_args(args)
        if parsed.command == "answer":
            assert ep.cmd_answer(parsed) == 0
        else:
            assert ep.cmd_evaluate(parsed) == 0

    monkeypatch.setattr(harness, "_run", fake_run)
    gold = {"answer": ["Alpha", "Beta"], "answer_type": "List[str]"}
    items = [
        {
            "id": "q",
            "dataset": "corporatebench",
            "question": "Which?",
            "gold_answer": gold,
            "retrieved_context": "Alpha, Beta, Extra",
        }
    ]
    results = harness.run_judge(
        Path(ep.__file__), items, tmp_path / "run", dataset="corporatebench"
    )
    assert "JSON array" in seen_prompts[0]
    assert results[0].is_correct is False
    assert results[0].partial == pytest.approx(0.8)
    sample = Sample(
        user_id="corp-test",
        dataset="corporatebench",
        sessions=(),
        questions=(Question("q", "Which?", gold, "kb_qa"),),
    )
    record = build_record(
        run_id="corp-test",
        step="step-0",
        profile="local",
        bench_dir=tmp_path,
        samples=[sample],
        results=results,
        data_fingerprint={"dataset": "corporatebench"},
        models={"llm": "test"},
    )
    assert record.scores["overall"] == 0
    assert record.scores["dataset_score"]["mean"] == 0.8
    assert record.scores["dataset_score"]["n"] == 1
    assert record.breakdown["partial_credit"] == {"mean": 0.8, "n": 1}
