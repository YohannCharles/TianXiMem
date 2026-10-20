"""Offline official-run rescoring never touches Add, Search, or Answer."""

from __future__ import annotations

import json
from pathlib import Path

from eval.experiments.official_rescore import rescore_run


def _write(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _question(root: Path, seq: int, *, answerable: bool, answer: str, original_ok: bool) -> Path:
    directory = root / "wire-a" / "questions" / str(seq)
    _write(
        directory / "input.jsonl",
        {
            "id": str(seq),
            "seq": seq,
            "dataset": "musique",
            "question": "Question?",
            "gold_answer": {
                "answerable": answerable,
                "answer": "Robert Roosevelt",
                "answer_aliases": [],
            },
            "gold_answers": ["Robert Roosevelt"] if answerable else ["INSUFFICIENT_EVIDENCE"],
        },
    )
    _write(directory / "answers.jsonl", {"id": str(seq), "generated_answer": answer})
    _write(
        directory / "labels.jsonl",
        {
            "id": str(seq),
            "is_correct": original_ok,
            "label": "CORRECT" if original_ok else "WRONG",
            "judge_response": "old",
        },
    )
    return directory


def test_rescore_run_is_non_destructive_and_applies_musique_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    first = _question(
        run,
        1,
        answerable=True,
        answer="The answer is Robert Roosevelt.",
        original_ok=True,
    )
    second = _question(
        run,
        2,
        answerable=False,
        answer="Cannot determine from the memories.",
        original_ok=True,
    )
    original_first = (first / "labels.jsonl").read_text(encoding="utf-8")
    original_second = (second / "labels.jsonl").read_text(encoding="utf-8")

    summary = rescore_run(run)

    assert summary["network_calls"] is False
    assert summary["original"]["correct"] == 2
    assert summary["corrected"]["correct"] == 0
    assert summary["changed_rows"] == 2
    assert (first / "labels.jsonl").read_text(encoding="utf-8") == original_first
    assert (second / "labels.jsonl").read_text(encoding="utf-8") == original_second

    rows = [
        json.loads(line)
        for line in (run / "offline-rescore" / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["seq"], row["is_correct"]) for row in rows] == [(1, False), (2, False)]
    assert rows[0]["policy"] == "offline-deterministic:musique"
    assert rows[1]["judge_response"].startswith("MuSiQue")
