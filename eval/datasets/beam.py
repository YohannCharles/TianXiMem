"""BEAM（100K 档）的加载层——**记忆是真对话、裁判用官方那份 pipeline**。

## 这份数据长什么样（2026-09-30 实测，`benchmark_data/beam/`）

```text
data/100K-00000-of-00001.parquet   20 个 conversation，一行一个：
  conversation_seed.category  编程序列（Coding / …）——只用来分层抽样
  chat                        list[list[message]]，**3 个 session**，各 56–72 条消息
                              每条消息：content / role / id / index / question_type / time_anchor
  probing_questions           10 组 × 每组 2 题 = 每个 conversation 20 题
                              每组字段不同，**只有 `rubric` 每组都有**（1–2 条判分要点）
                              temporal / event_ordering 那几组另有 `answer`
```

## 三条要点

1. **时间戳要"往下传播"**：`time_anchor` 只挂在每个 session 的**第一条**消息上
   （形如 `"March-15-2024"`，**不是** ISO 格式）⇒ 同一 session 的其余消息要靠它补
   `timestamp_ms`，否则 `event_time` 全空、`created_at` 只能发空串。
   ⚠ 与 LoCoMo/LME 同一条形：**session 内没有区分度**（§6.1）。
2. **金标是 `rubric`（列表）**，不是 `answer`：官方 `pipeline_beam.py` 的 `rubric_items()`
   认 `rubric_nuggets` / `rubrics` / `rubric`，判分是**逐条三点制**（0 / 0.5 / 1）。
   单独取 `answer` 只覆盖 10 组里的 5 组（[`tools/recover_official_gold.py`][gold] 里那条
   注记是给**采集流量**用的，这里用的是**官方 pipeline**）。

   [gold]: ../../tools/recover_official_gold.py
3. **一格一个 `Sample`**（一个 conversation + 它自己的 20 题）——和 LongMemEval 一样是
   "1 份记忆 : 多道题"，只是那份是 1 题 1 个 haystack、这份是 1 格 20 题。
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = ["PARQUET", "PIPELINE", "TIME_ANCHOR_FORMAT", "USER_PREFIX", "load_beam"]

#: 归档里的相对路径（D16：数据集专属文件名的知识止步于本模块）。
PARQUET = "beam/data/100K-00000-of-00001.parquet"

#: **官方 pipeline**（与 LoCoMo/LME/CL-Bench 一样，在归档里、由 harness subprocess 调）。
PIPELINE = "pipeline_beam.py"

USER_PREFIX = "beam-"

#: `time_anchor` 的形状（实测 `"March-15-2024"`）——**不是 ISO**，别拿 `fromisoformat` 去啃。
TIME_ANCHOR_FORMAT = "%B-%d-%Y"

#: probing 组的顺序（决定题号顺序，别依赖 dict 的插入序）。
GROUPS = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


def _rows(path: Path) -> Iterator[dict]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4):
        yield from batch.to_pylist()


def _epoch_ms(anchor: str) -> int | None:
    try:
        moment = datetime.strptime(anchor.strip(), TIME_ANCHOR_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return int(moment.timestamp() * 1000)


def _sessions(entry: dict, conversation_id: str) -> tuple[Session, ...]:
    """把 `chat` 的 3 个 session 原样搬过来，并**把 `time_anchor` 传播给同 session 的每条消息**。"""
    sessions: list[Session] = []
    for index, messages in enumerate(entry.get("chat") or []):
        anchor = next(
            (str(message["time_anchor"]) for message in messages if message.get("time_anchor")),
            None,
        )
        timestamp = _epoch_ms(anchor) if anchor else None
        sessions.append(
            Session(
                session_id=f"{conversation_id}-s{index + 1}",
                messages=tuple(
                    Message(
                        role=str(message.get("role") or "user"),
                        content=str(message.get("content") or ""),
                        timestamp_ms=timestamp,
                    )
                    for message in messages
                ),
            )
        )
    return tuple(sessions)


def _questions(entry: dict, conversation_id: str) -> tuple[Question, ...]:
    """10 组 × 2 题。**金标取 `rubric`**（每组都有；`answer` 只有 5 组有）。"""
    raw = entry.get("probing_questions")
    probing = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(probing, dict):
        raise ValueError(f"conversation {conversation_id}：probing_questions 解不出 dict")
    questions: list[Question] = []
    for group in GROUPS:
        for offset, item in enumerate(probing.get(group) or []):
            question = item.get("question")
            rubrics = item.get("rubric")
            if not question or not isinstance(rubrics, list) or not rubrics:
                raise ValueError(f"{conversation_id}/{group}#{offset}：缺 question 或 rubric")
            questions.append(
                Question(
                    qid=f"{conversation_id}-{group}-{offset}",
                    question=str(question),
                    # 金标 = rubric 列表（`pipeline_beam.py` 的 `rubric_items()` 直接吃它）
                    gold=[str(r) for r in rubrics],
                    category=group,
                    # 拒答组单独标出来：它的"对"是**说没有**，与其余 9 组的行为相反。
                    is_abstention=group == "abstention",
                )
            )
    return tuple(questions)


_PERSONA_NAME = re.compile(r"Name:\s*([^\n•·]+)")


def _persona_name(entry: dict, conversation_id: str) -> str:
    """从 `user_profile.user_info` 里取 persona 的名字（**官方正文前缀用的就是它**）。

    官方流量里 BEAM 的 user 侧正文写作 `Christina Baker: …`（assistant 侧是
    `Assistant: …`）——见 [`../harness/add_shape.py`](../harness/add_shape.py) 的表。
    ⚠ **解不出就抛**：少一个名字会让检索看不见"谁在说"，而屏幕上看不出任何异常。
    """
    raw = entry.get("user_profile")
    profile = ast.literal_eval(raw) if isinstance(raw, str) else raw
    info = profile.get("user_info") if isinstance(profile, dict) else None
    match = _PERSONA_NAME.search(info or "")
    if not match:
        raise ValueError(f"{conversation_id}：user_profile 里解不出 persona 名（前缀要用它）")
    return match.group(1).strip()


def load_beam(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 BEAM（100K 档）。

    ⚠ 部分跑要加 `spread=True`：parquet 按 `conversation_seed.category` 排，
    `[:limit]` 会只取到一类场景。
    """
    path = Path(bench_dir) / PARQUET
    if not path.exists():
        raise FileNotFoundError(f"缺 {path}（`make fetch-data` 取回 official-extra 那一档）")
    entries = list(_rows(path))
    if limit is not None:
        entries = (
            stratified_sample(
                entries,
                limit,
                key=lambda e: str((e.get("conversation_seed") or {}).get("category") or ""),
            )
            if spread
            else entries[:limit]
        )

    samples: list[Sample] = []
    for entry in entries:
        conversation_id = str(entry.get("conversation_id") or "")
        if not conversation_id:
            raise ValueError(f"{path.name}：有一行没有 conversation_id")
        sessions = _sessions(entry, conversation_id)
        if not sessions:
            raise ValueError(f"{path.name}:{conversation_id}：chat 为空——没有记忆可喂")
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{conversation_id}",
                dataset="beam",
                sessions=sessions,
                questions=_questions(entry, conversation_id),
                # `@speaker` 标签要用它（官方 BEAM 的 user 侧正文就是 `Christina Baker: …`）。
                speaker_names=(_persona_name(entry, conversation_id), "Assistant"),
            )
        )
    return samples
