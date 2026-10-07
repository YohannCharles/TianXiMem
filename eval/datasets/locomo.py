"""LoCoMo-Refined 加载器——`questions.jsonl`（题与 gold）+ 对话全文。

## 两个文件都得读，**但它们回答的是两个不同的问题**

| 用途 | 取哪个 |
| --- | --- |
| 喂给 `Add` 的对话全文 | [`conversations.jsonl`] **优先**，缺失时退回 [`locomo_refined.json`] |
| 判分用的题与 gold | [`questions.jsonl`] |

**`questions.jsonl` 里没有对话全文**——它的 `evidence_messages` 只是**证据轮**
（一道题平均一两轮），拿它当 Add 的输入会让整个 session 只剩几个孤立 turn。

## ⚠ 归档里**没有** `conversations.jsonl`——这不是缺陷，是出处链

`eval/datasets/manifest.py` 明写它"本清单不收录——它属 `eval/datasets/` 的 clone"。
但归档的 `locomo_refined.json` **含同一份对话全文**，两个来源的差别**只有首尾空白**：

> **已核实（全量 5,882 turn）**：`locomo_refined.json` 的每条 `text`
> 经 `strip()` 后与 `conversations.jsonl` **逐字等价**——209 条仅首尾空白不同、
> **0 条内容不同**、speaker 与 role 的映射零冲突。

所以两个来源在这里**产出完全相同的 `Message` 流**，两条路径都可用：

- 归档路径（`dataset/`）→ `locomo_refined.json` + `strip()`
- 开发路径（`eval/datasets/LoCoMo-Refined/data/public/`）→ `conversations.jsonl`

> ⚠ **D16 说"用 `locomo_refined.json` 会引入 209 条契约违规"——那句的前提是"不 strip"。**
> 契约要求 `content` 首尾无空白，而本模块**两条路径都强制 `strip()`**
> （见 [`preprocess.normalize_content`](./preprocess.py)），所以该违规在这里不可能出现。
> 这条差异值得回填进 `docs/decisions.md` 的 D16——**归档单独就够用，不必依赖 clone**。

## 说话人 → `role` 的映射

数据集里没有 `role`（`locomo_refined.json` 只有 `speaker`），映射是
**`speaker_a` → `user`、`speaker_b` → `assistant`**——这是数据集给两个真人的
**约定标签，不是"用户 vs 助手"**（§6.2 的配对判据在这份数据上实际是
"speaker_a 的一轮 + 对方回应，直到 speaker_a 的下一轮"）。

> 上面那条等价关系由 `tests/test_datasets.py` 守着，**不在运行时断言**：
> 加载器只挑一个来源（`conversations.jsonl` 优先），两边都在场时由用例逐条比对。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..jsonl_io import read_jsonl
from .layout import archive_file
from .preprocess import (
    Message,
    Question,
    Sample,
    Session,
    normalize_content,
    parse_locomo_time,
    to_epoch_ms,
)

__all__ = [
    "load_locomo",
    "conversation_file",
    "CONVERSATIONS_JSONL",
    "PIPELINE",
    "QUESTIONS_JSONL",
    "REFINED_JSON",
]

#: 归档里本数据集的 answer/judge 脚本——harness 用 subprocess 调它，不 import。
PIPELINE = "pipeline_locomo-refined.py"

QUESTIONS_JSONL = "questions.jsonl"
CONVERSATIONS_JSONL = "conversations.jsonl"
REFINED_JSON = "locomo_refined.json"


def conversation_file(bench_dir: Path) -> Path:
    """对话全文的来源文件——`conversations.jsonl` 优先，退回 `locomo_refined.json`。"""
    direct = archive_file(bench_dir, CONVERSATIONS_JSONL)
    if direct.exists():
        return direct
    refined = archive_file(bench_dir, REFINED_JSON)
    if refined.exists():
        return refined
    raise FileNotFoundError(
        f"{bench_dir} 下既没有 {CONVERSATIONS_JSONL} 也没有 {REFINED_JSON}——"
        "对话全文无处可取（见 docs/benchmark-data.md 的出处链）"
    )


def _sessions_from_conversations_jsonl(
    row: dict, *, source: Path
) -> tuple[list[Session], tuple[str, str]]:
    """开发路径：`conversations.jsonl` 一行即一段对话，每条 message **自带 `role`**。"""
    sample_id = row["sample_id"]
    sessions = []
    for s in row["sessions"]:
        session_idx = s["session_index"]
        moment = parse_locomo_time(s["date_time"])
        ms = to_epoch_ms(moment)
        messages = []
        for m in s["messages"]:
            where = f"{source.name}:{sample_id}:D{session_idx}:{m.get('message_index')}"
            messages.append(
                Message(
                    role=m["role"],
                    content=normalize_content(m["text"], where=where),
                    timestamp_ms=ms,
                )
            )
        sessions.append(Session(session_id=f"{sample_id}#s{session_idx}", messages=tuple(messages)))
    return sessions, (row["speaker_a"], row["speaker_b"])


def _sessions_from_refined_json(
    row: dict, *, source: Path
) -> tuple[list[Session], tuple[str, str]]:
    """归档路径：`locomo_refined.json` 的 `conversation`
    是成对出现的 `session_N` / `session_N_date_time`。

    `role` 由 `speaker` 派生（映射见模块 docstring）。
    """
    sample_id = row["sample_id"]
    conv = row["conversation"]
    speaker_a, speaker_b = conv["speaker_a"], conv["speaker_b"]
    if speaker_a == speaker_b:
        raise ValueError(f"{source.name}:{sample_id}：两个说话人同名，`role` 无法派生")

    sessions = []
    index = 1
    while f"session_{index}" in conv:
        moment = parse_locomo_time(conv[f"session_{index}_date_time"])
        ms = to_epoch_ms(moment)
        turns = conv[f"session_{index}"]
        messages = []
        for position, turn in enumerate(turns, start=1):
            speaker = turn["speaker"]
            if speaker == speaker_a:
                role = "user"
            elif speaker == speaker_b:
                role = "assistant"
            else:
                raise ValueError(
                    f"{source.name}:{sample_id}:D{index}:{position}：未知说话人 {speaker!r}"
                )
            messages.append(
                Message(
                    role=role,
                    content=normalize_content(
                        turn["text"], where=f"{source.name}:{sample_id}:D{index}:{position}"
                    ),
                    timestamp_ms=ms,
                )
            )
        sessions.append(Session(session_id=f"{sample_id}#s{index}", messages=tuple(messages)))
        index += 1
    return sessions, (speaker_a, speaker_b)


def _load_conversations(bench_dir: Path) -> dict[str, tuple[list[Session], tuple[str, str]]]:
    source = conversation_file(bench_dir)
    if source.name == CONVERSATIONS_JSONL:
        rows = read_jsonl(source)
        return {r["sample_id"]: _sessions_from_conversations_jsonl(r, source=source) for r in rows}
    rows = json.loads(source.read_text(encoding="utf-8"))
    return {r["sample_id"]: _sessions_from_refined_json(r, source=source) for r in rows}


def _load_questions(path: Path) -> dict[str, list[Question]]:
    """按 `sample_id` 归组题目。**`category` 归一成 `int` 再转 `str`**。

    ⚠ `questions.jsonl` 里是字符串 `"4"`，`locomo_refined.json` 里是整数 `4`——
    不归一化而直接按 `category == 1` 筛选会**静默筛出 0 条**（§12.3）。
    本模块只读前者，但归一化照做：它就是为这条落差存在的。
    """
    grouped: dict[str, list[Question]] = {}
    for ordinal, row in enumerate(read_jsonl(path)):
        gold = row["answer"]
        if gold in (None, [], ""):
            # questions.jsonl 的 1,382 题全是可答题（adversarial 那批只在
            # locomo_refined.json 的 qa 里、且没有 qa_id）。真遇到就抛，别静默当成空 gold。
            raise ValueError(f"{path.name}:{ordinal}：gold 为空——该题不该走判分路径")
        question = Question(
            qid=row["qa_id"],
            question=normalize_content(row["question"], where=f"{path.name}:{row['qa_id']}"),
            gold=gold,
            category=str(int(row["category"])),
            evidence=tuple(row.get("evidence") or ()),
        )
        grouped.setdefault(row["sample_id"], []).append(question)
    return grouped


def load_locomo(bench_dir: str | Path) -> list[Sample]:
    """加载 LoCoMo-Refined。**`user_id` = `sample_id`**（一段对话一个 user，题共享它）。

    排序按 `qa_index`——`questions.jsonl` 的物理顺序，跨 run 稳定即可。
    缺对话的样本会被跳过并在返回值里不可见**但不会静默**：题集与对话集对不上时抛错。
    """
    bench_dir = Path(bench_dir)
    conversations = _load_conversations(bench_dir)
    questions = _load_questions(archive_file(bench_dir, QUESTIONS_JSONL))

    unknown = set(questions) - set(conversations)
    if unknown:
        raise ValueError(f"{QUESTIONS_JSONL} 里的 sample_id 在对话文件里不存在：{sorted(unknown)}")

    samples = []
    for sample_id, qs in questions.items():
        sessions, names = conversations[sample_id]
        samples.append(
            Sample(
                user_id=sample_id,
                dataset="locomo-refined",
                sessions=tuple(sessions),
                questions=tuple(qs),
                speaker_names=names,
            )
        )
    return sorted(samples, key=lambda s: s.user_id)
