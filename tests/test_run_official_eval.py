"""`tools/run_official_eval.py` 的纯函数：技术错误重判的清理。"""

from __future__ import annotations

import json
from pathlib import Path

from tools.run_official_eval import _clear_technical_errors


def test_run_forwards_streaming_workers_without_changing_answer_budget(tmp_path, monkeypatch):
    from eval.experiments import official_dataset
    from tools import run_official_eval as tool

    monkeypatch.setattr(tool, "preflight", lambda args: ({"questions": [{}]}, tmp_path))
    forwarded = []
    monkeypatch.setattr(official_dataset, "main", lambda argv: forwarded.extend(argv) or 0)
    assert tool.main([
        "run", "--run-id", "streaming", "--service-revision", "v1", "--judge-workers", "4",
        "--ingest-workers", "2", "--serial-prefix", "schedule.json",
        "--judge-scheduling", "after-replay", "--replay-only",
        "--prefix-scheduling", "independent",
    ]) == 0
    assert forwarded[forwarded.index("--judge-workers") + 1] == "4"
    assert forwarded[forwarded.index("--max-tokens") + 1] == "1024"
    assert forwarded[forwarded.index("--ingest-workers") + 1] == "2"
    assert forwarded[forwarded.index("--serial-prefix") + 1] == "schedule.json"
    assert forwarded[forwarded.index("--judge-scheduling") + 1] == "after-replay"
    assert "--replay-only" in forwarded
    assert forwarded[forwarded.index("--prefix-scheduling") + 1] == "independent"


def test_judge_forwards_checkpoint_source_and_workers(tmp_path, monkeypatch):
    from eval.experiments import official_dataset
    from tools import run_official_eval as tool

    forwarded = []
    monkeypatch.setattr(official_dataset, "main", lambda argv: forwarded.extend(argv) or 0)
    assert tool.main([
        "judge",
        "--source-run-dir", str(tmp_path / "official-v1.2"),
        "--run-id", "official-v1.2-answer-v2",
        "--reports-dir", str(tmp_path / "reports"),
        "--configs-dir", str(tmp_path / "configs"),
        "--profile", "compare",
        "--max-tokens", "2048",
        "--judge-workers", "16",
        "--memory-date", "none",
        "--annotate-mark", "paren",
    ]) == 0
    assert forwarded[:3] == ["judge", "--source-run-dir", str(tmp_path / "official-v1.2")]
    assert forwarded[forwarded.index("--run-id") + 1] == "official-v1.2-answer-v2"
    assert forwarded[forwarded.index("--judge-workers") + 1] == "16"
    assert forwarded[forwarded.index("--max-tokens") + 1] == "2048"


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_clear_technical_errors_keeps_real_labels(tmp_path: Path) -> None:
    _write(
        tmp_path / "wire-a/labels.jsonl",
        [
            {"id": "1", "is_correct": True, "label": "CORRECT"},
            {"id": "2", "is_correct": False, "label": "JUDGE_ERROR"},
        ],
    )
    _write(
        tmp_path / "wire-b/labels.jsonl",
        [{"id": "3", "is_correct": False, "label": "WRONG"}],
    )
    # 被强杀留下的半行：不是一条完整判定，清掉即可。
    (tmp_path / "wire-c").mkdir()
    (tmp_path / "wire-c/labels.jsonl").write_text(
        '{"id": "4", "is_correct": true, "label": "CORRECT"}\n{"id": "5", "lab',
        encoding="utf-8",
    )

    assert _clear_technical_errors(tmp_path) == 1
    a = [json.loads(line) for line in (tmp_path / "wire-a/labels.jsonl").read_text().splitlines()]
    assert [(row["id"], row["label"]) for row in a] == [("1", "CORRECT")]
    b = [json.loads(line) for line in (tmp_path / "wire-b/labels.jsonl").read_text().splitlines()]
    assert [(row["id"], row["label"]) for row in b] == [("3", "WRONG")]
    # 半行文件里唯一那条完整记录保留，半行丢掉。
    c = [json.loads(line) for line in (tmp_path / "wire-c/labels.jsonl").read_text().splitlines()]
    assert [(row["id"], row["label"]) for row in c] == [("4", "CORRECT")]

    assert _clear_technical_errors(tmp_path) == 0


def test_compare_requires_existing_runs_and_delegates(tmp_path: Path, monkeypatch) -> None:
    from tools import run_official_eval as tool

    assert (
        tool.main(
            [
                "compare",
                "--left-run-id",
                "v1.0",
                "--right-run-id",
                "latest",
                "--reports-dir",
                str(tmp_path),
            ]
        )
        == 2
    )

    for name in ("v1.0", "latest"):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "run-manifest.json").write_text("{}", encoding="utf-8")

    captured: dict = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    import eval.experiments.official_dataset as official

    monkeypatch.setattr(official, "main", fake_main)
    code = tool.main(
        [
            "compare",
            "--left-run-id",
            "v1.0",
            "--right-run-id",
            "latest",
            "--reports-dir",
            str(tmp_path),
            "--left-name",
            "v1.0",
            "--right-name",
            "main",
        ]
    )
    assert code == 0
    argv = captured["argv"]
    assert argv[:4] == ["compare", "--left", str(tmp_path / "runs/v1.0"), "--right"]
    assert str(tmp_path / "runs/latest") in argv
    assert argv[argv.index("--left-name") + 1] == "v1.0"
    assert argv[argv.index("--right-name") + 1] == "main"
    assert argv[argv.index("--output") + 1].endswith("official-comparison-v1.0-vs-latest")
