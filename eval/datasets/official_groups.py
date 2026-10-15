"""采集 Search 的可靠数据集归类、分类导出与固定评分清单。

按 seq 联接原始 Search 和现有判分套件；不以语料形态猜测归属。
未知归属统一为 other。原始文件保持不变，所有 Search 记录均保留。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from tools.build_official_kit import PROMPT_REFS

from eval.datasets.aml.plan import digest
from eval.jsonl_io import read_jsonl

SOURCE_FILES = (
    "official-searches.jsonl",
    "official-eval-kit.jsonl",
    "official-adds.jsonl",
    "official-timeline.jsonl",
)
GROUP_VERSION = "official-groups-v1"
RELIABLE_BASES = {"tag", "official_feedback", "public_gold", "gold_extra"}


def data_identity(manifest: dict) -> str:
    return digest({key: value for key, value in manifest.items() if key != "capture_dir"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def has_answers(row: dict) -> bool:
    return any(value is not None and str(value).strip() for value in row.get("gold_answers") or [])


def group_name(row: dict) -> str:
    dataset = row.get("dataset")
    if dataset in PROMPT_REFS and dataset != "unknown" and row.get("attribution") in RELIABLE_BASES:
        return dataset
    return "other"


def classified_rows(capture: Path) -> list[dict]:
    searches = read_jsonl(capture / SOURCE_FILES[0])
    kit = read_jsonl(capture / SOURCE_FILES[1])
    search_index = {int(row["seq"]): row for row in searches}
    kit_index = {int(row["seq"]): row for row in kit}
    if len(search_index) != len(searches) or len(kit_index) != len(kit):
        raise ValueError("Search 或金标套件包含重复 seq")
    if set(search_index) != set(kit_index):
        raise ValueError("Search 与金标套件的 seq 集合不一致，请先核对/重建套件")
    output = []
    for search in searches:
        gold = kit_index[int(search["seq"])]
        if any(search.get(key) != gold.get(key) for key in ("user_id", "query", "ts")):
            raise ValueError(f"seq={search['seq']} 的题目、用户或时间与套件不匹配")
        dataset = group_name(gold)
        has_gold = has_answers(gold) or bool(gold.get("gold_rubric"))
        scorable = dataset != "other" and gold.get("judge_kind") not in (None, "none") and has_gold
        output.append(
            {
                **gold,
                **search,
                "source_dataset": gold.get("dataset"),
                "dataset": dataset,
                "scorable": scorable,
            }
        )
    return output


def prepare_groups(
    capture: Path,
    output: Path,
    *,
    per_dataset: int | None = None,
    datasets: list[str] | None = None,
) -> dict:
    """一次生成分类文件与固定 seq 清单；已有产物不覆盖，避免替换后续比较的输入。"""
    if per_dataset is not None and per_dataset < 1:
        raise ValueError("--per-dataset 必须为正整数")
    if output.exists():
        raise ValueError(f"分类输出已存在：{output}；复用 manifest 或指定新的 --output")
    rows = classified_rows(capture)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    requested = set(datasets) if datasets is not None else set(groups)
    if requested - set(groups):
        raise ValueError(f"所选数据集在采集中不存在：{sorted(requested - set(groups))}")
    sources = {name: sha256(capture / name) for name in SOURCE_FILES}
    selected = []
    counts = {}
    for dataset, group in sorted(groups.items()):
        eligible = sorted((r for r in group if r["scorable"]), key=lambda r: int(r["seq"]))
        # 按用户轮转取题，避免少量大用户占满固定的小样本；轮转与用户内均为确定性顺序。
        users: dict[str, list[dict]] = defaultdict(list)
        for row in eligible:
            users[row["user_id"]].append(row)
        ordered = [
            user_rows[index]
            for index in range(max((len(v) for v in users.values()), default=0))
            for _user, user_rows in sorted(users.items())
            if index < len(user_rows)
        ]
        picked = ordered[:per_dataset] if per_dataset is not None else ordered
        if dataset not in requested:
            picked = []
        selected.extend({key: row[key] for key in ("seq", "dataset", "user_id")} for row in picked)
        answers = sum(has_answers(row) for row in group)
        counts[dataset] = {
            "searches": len(group),
            "scorable": len(eligible),
            "answers": answers,
            "rubric_only": sum(bool(r.get("gold_rubric")) and not has_answers(r) for r in group),
            "selected": len(picked),
        }
    manifest = {
        "version": GROUP_VERSION,
        "capture_dir": str(capture.resolve()),
        "source_sha256": sources,
        "counting": "search records; unique seq; no question-text deduplication",
        "classification": "question-level reliable attribution; unresolved -> other",
        "selection": {"per_dataset": per_dataset, "datasets": sorted(requested)},
        "counts": counts,
        "questions": sorted(selected, key=lambda r: int(r["seq"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".official-groups-", dir=output.parent))
    try:
        for dataset, group in sorted(groups.items()):
            path = temporary / "searches" / f"{dataset}.jsonl"
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in group:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(temporary / "manifest.json", manifest)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_manifest(path: Path, *, verify_sources: bool = False) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != GROUP_VERSION:
        raise ValueError("不支持的分类清单版本")
    questions = manifest["questions"]
    if len({int(r["seq"]) for r in questions}) != len(questions):
        raise ValueError("固定题目清单包含重复 seq")
    if any(r["dataset"] == "other" for r in questions):
        raise ValueError("other 没有可靠评分来源，不能进入准确率分母")
    if verify_sources:
        capture = Path(manifest["capture_dir"])
        for name in SOURCE_FILES:
            if sha256(capture / name) != manifest["source_sha256"].get(name):
                raise ValueError(f"源文件已变化：{name}；请使用新的分类清单和 run-id")
    return manifest
