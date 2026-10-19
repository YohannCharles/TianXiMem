"""`tools/verify_official_gold.py`：金标来源与搬运审计的检测能力。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools import verify_official_gold as audit


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.fixture
def capture(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "capture"
    root.mkdir()
    monkeypatch.setattr(audit, "EXTRA_FILES", ("official-gold-extra.jsonl",))
    _write(
        root / "official-eval-kit.jsonl",
        [
            {
                "seq": 1,
                "dataset": "mquake-remastered",
                "judge_kind": "exact",
                "gold_answers": ["A"],
                "gold_rubric": None,
                "gold_native": None,
                "gold_alternative_answers": [],
                "gold_variant": "original",
                "gold_judging": "",
            },
            {
                "seq": 2,
                "dataset": "locomo-refined",
                "judge_kind": "llm",
                "gold_answers": ["B"],
                "gold_rubric": None,
                "gold_native": None,
                "gold_alternative_answers": [],
                "gold_variant": None,
                "gold_judging": "",
            },
        ],
    )
    _write(
        root / "official-eval-questions.jsonl",
        [
            {"seq": 1, "public_gold": None},
            {"seq": 2, "public_gold": {"gold_locomo_answer": ["B"]}},
        ],
    )
    _write(
        root / "official-gold-extra.jsonl",
        [
            {
                "seq": 1,
                "source_dataset": "mquake-remastered",
                "gold_kind": "exact",
                "gold_answers": ["A"],
                "gold_rubric": None,
                "gold_native": None,
                "gold_alternative_answers": [],
                "gold_variant": "original",
                "gold_judging": "",
                "source_locator": {"file": "q.json", "id": 1},
            }
        ],
    )
    return root


def test_audit_accepts_extra_and_public_provenance(capture: Path) -> None:
    result, hard = audit.audit(capture, sample_per_family=1)
    assert hard == []
    assert result["stats"]["mquake-remastered"]["provenance_extra"] == 1
    assert result["stats"]["locomo-refined"]["provenance_public"] == 1
    assert result["samples"]["mquake-remastered"][0]["source"]["file"] == "q.json"


def test_audit_flags_a_changed_gold(capture: Path) -> None:
    kit = [
        json.loads(line) for line in (capture / "official-eval-kit.jsonl").read_text().splitlines()
    ]
    kit[0]["gold_answers"] = ["X"]
    _write(capture / "official-eval-kit.jsonl", kit)

    _result, hard = audit.audit(capture, sample_per_family=0)
    assert any("seq=1" in item and "gold_answers" in item for item in hard)


def test_audit_flags_missing_provenance(capture: Path) -> None:
    kit = [
        json.loads(line) for line in (capture / "official-eval-kit.jsonl").read_text().splitlines()
    ]
    kit[1]["dataset"] = "unknown"
    _write(capture / "official-eval-kit.jsonl", kit)
    (capture / "official-eval-questions.jsonl").write_text(
        '{"seq": 1, "public_gold": null}\n{"seq": 2, "public_gold": null}\n',
        encoding="utf-8",
    )

    _result, hard = audit.audit(capture, sample_per_family=0)
    assert any("seq=2" in item and "没有 public_gold" in item for item in hard)
