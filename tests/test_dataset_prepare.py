"""按需准备与迁移的失败边界：不发外网请求，不把半成品发布为数据。"""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from eval.datasets import prepare
from eval.datasets.layout import archive_file, local_path
from eval.datasets.manifest import MANIFEST


def _entry(name: str, content: bytes, **extra) -> dict:
    return {
        "name": name,
        "tier": "required",
        "url": f"https://example.invalid/{name}",
        "sha256": hashlib.sha256(content).hexdigest(),
        **extra,
    }


def test_selection_keeps_other_datasets_and_large_replay_corpora_out() -> None:
    selected = prepare.entries_for("locomo-refined")
    assert {entry["name"] for entry in selected} == {
        "aml_readme.md",
        "pipeline_locomo-refined.py",
        "locomo_refined.json",
        "questions.jsonl",
        "locomo_refined_readme.md",
    }
    assert not any("lme_s_cleaned" in entry["name"] for entry in selected)
    assert not any(
        "data.z" in entry["name"] for entry in prepare.entries_for("doc-pp", purpose="judge")
    )
    assert not any(
        "HaluMem-Medium" in entry["name"]
        for entry in prepare.entries_for("halumem", purpose="judge")
    )
    assert not any(
        "chat_history_32k" in entry["name"]
        for entry in prepare.entries_for("personamem-v2", purpose="judge")
    )
    assert len({local_path(e["name"]) for e in MANIFEST}) == len(MANIFEST)


def test_missing_file_downloads_to_tmp_and_second_preparation_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b'{"question":"sample"}\n'
    entry = _entry("questions.jsonl", content)
    monkeypatch.setattr(prepare, "MANIFEST", [entry])
    calls = []

    def download(url: str, path: Path) -> None:
        calls.append(url)
        assert path.is_relative_to(tmp_path / ".tmp")
        assert not (tmp_path / local_path(entry["name"])).exists()
        path.write_bytes(content)

    monkeypatch.setattr(prepare, "_download", download)
    prepare.ensure_dataset("locomo-refined", tmp_path)
    prepare.ensure_dataset("locomo-refined", tmp_path, offline=True)
    assert calls == [entry["url"]]
    assert archive_file(tmp_path, entry["name"]).read_bytes() == content
    assert list((tmp_path / ".tmp").iterdir()) == []


@pytest.mark.parametrize("failure", ["truncated", "network"])
def test_failed_download_never_publishes_and_can_be_retried(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"complete-data"
    entry = _entry("locomo_refined.json", content)
    monkeypatch.setattr(prepare, "MANIFEST", [entry])

    def broken(url: str, path: Path) -> None:
        path.write_bytes(b"partial")
        if failure == "network":
            raise urllib.error.URLError("断网")

    monkeypatch.setattr(prepare, "_download", broken)
    with pytest.raises(prepare.PreparationError):
        prepare.ensure_dataset("locomo-refined", tmp_path)
    assert not archive_file(tmp_path, entry["name"]).exists()
    assert list((tmp_path / ".tmp").iterdir()) == []
    monkeypatch.setattr(prepare, "_download", lambda url, path: path.write_bytes(content))
    prepare.ensure_dataset("locomo-refined", tmp_path)
    assert archive_file(tmp_path, entry["name"]).read_bytes() == content


def test_patch_checks_upstream_and_published_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = (
        b"import argparse\nasync with httpx.AsyncClient(timeout=120) as client, "
        b'output.open("a", encoding="utf-8") as handle:\n    pass\n'
    )
    patched = prepare._patch_async_open(original.decode())[0].encode()
    entry = _entry(
        "pipeline_locomo-refined.py",
        patched,
        upstream_sha256=hashlib.sha256(original).hexdigest(),
        local_patch="async-open",
    )
    monkeypatch.setattr(prepare, "MANIFEST", [entry])
    monkeypatch.setattr(prepare, "_download", lambda url, path: path.write_bytes(original))
    prepare.ensure_dataset("locomo-refined", tmp_path)
    assert archive_file(tmp_path, entry["name"]).read_bytes() == patched
    # --patch 只需要评分源文件，不因语料尚未下载而失败，也不会联网。
    archive_file(tmp_path, entry["name"]).write_bytes(original)
    monkeypatch.setattr(prepare, "MANIFEST", [entry, _entry("questions.jsonl", b"not downloaded")])
    monkeypatch.setattr(prepare, "_download", lambda *a: pytest.fail("补丁命令不能下载"))
    assert prepare.main(["--patch", "--dataset", "locomo-refined", "--dir", str(tmp_path)]) == 0
    assert archive_file(tmp_path, entry["name"]).read_bytes() == patched
    assert not archive_file(tmp_path, "questions.jsonl").exists()


def test_offline_missing_and_corrupt_local_files_never_use_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry("questions.jsonl", b"correct")
    monkeypatch.setattr(prepare, "MANIFEST", [entry])

    def never(*args) -> None:
        raise AssertionError("禁止联网")

    monkeypatch.setattr(prepare, "_download", never)
    with pytest.raises(prepare.PreparationError, match="缺"):
        prepare.ensure_dataset("locomo-refined", tmp_path, offline=True)
    assert not (tmp_path / ".tmp").exists()
    target = tmp_path / local_path(entry["name"])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local-edited-data")
    with pytest.raises(prepare.PreparationError, match="sha256"):
        prepare.ensure_dataset("locomo-refined", tmp_path)
    assert target.read_bytes() == b"local-edited-data"


def test_verified_files_survive_later_failure_and_are_not_downloaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _entry("questions.jsonl", b"questions")
    second = _entry("locomo_refined.json", b"conversations")
    monkeypatch.setattr(prepare, "MANIFEST", [first, second])
    calls = []

    def download(url, path):
        calls.append(url)
        if url == second["url"] and calls.count(url) == 1:
            raise urllib.error.URLError("第一次失败")
        path.write_bytes(b"questions" if url == first["url"] else b"conversations")

    monkeypatch.setattr(prepare, "_download", download)
    with pytest.raises(prepare.PreparationError):
        prepare.ensure_dataset("locomo-refined", tmp_path)
    prepare.ensure_dataset("locomo-refined", tmp_path)
    assert calls == [first["url"], second["url"], second["url"]]


def test_migration_preserves_unrecoverable_and_unregistered_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    entries = [_entry("questions.jsonl", b"questions"), _entry("rh3.md", b"notes")]
    monkeypatch.setattr(prepare, "MANIFEST", entries)
    for entry, content in zip(entries, [b"questions", b"notes"], strict=True):
        (source / entry["name"]).write_bytes(content)
    (source / "local-note.txt").write_text("keep me")
    assert prepare.migrate_archive(source, target) == 3
    assert not source.exists()
    assert (target / "locomo-refined/questions.jsonl").read_bytes() == b"questions"
    assert (target / ".legacy/rh3.md").read_bytes() == b"notes"
    assert (target / ".legacy/unregistered/local-note.txt").read_text() == "keep me"


@pytest.mark.parametrize("conflict", ["hash", "destination"])
def test_migration_preflights_all_files_before_moving_any(
    conflict: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    monkeypatch.setattr(prepare, "MANIFEST", [_entry("questions.jsonl", b"questions")])
    (source / "first-note.txt").write_bytes(b"preserve")
    (source / "questions.jsonl").write_bytes(b"bad" if conflict == "hash" else b"questions")
    if conflict == "destination":
        destination = target / local_path("questions.jsonl")
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"different local file")
    with pytest.raises(prepare.PreparationError):
        prepare.migrate_archive(source, target)
    assert (source / "first-note.txt").read_bytes() == b"preserve"
    assert (source / "questions.jsonl").exists()


def test_downloader_streams_into_staging_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"data"))
    dest = tmp_path / "staging/data.json"
    prepare._download("https://example.invalid/data.json", dest)
    assert dest.read_bytes() == b"data"


def test_prepare_and_legacy_command_work_without_site_packages() -> None:
    for args in (
        ["-m", "eval.datasets.prepare", "--list"],
        ["tools/fetch_benchmark_data.py", "--list"],
    ):
        completed = subprocess.run(
            [sys.executable, "-S", *args], capture_output=True, text=True, check=True
        )
        assert "locomo-refined" in completed.stdout


def test_runner_stops_before_paid_requests_when_preparation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eval.experiments import run as runner

    monkeypatch.setattr(runner, "benchmark_dir", lambda: tmp_path)
    monkeypatch.setattr(runner, "judge_preconditions", list)
    observed = []

    def fail(dataset, root, *, offline):
        observed.append((dataset, root, offline))
        raise prepare.PreparationError("缺数据且离线")

    def never(*args, **kwargs):
        raise AssertionError("准备失败后不能发请求")

    monkeypatch.setattr(runner, "ensure_dataset", fail)
    monkeypatch.setattr(runner, "run_round", never)
    assert runner.main(["--dataset", "beam", "--offline"]) == runner.EXIT_PRECONDITION_FAILED
    assert observed == [("beam", tmp_path, True)]
