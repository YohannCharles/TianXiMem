"""CorporateBench（zenith 一家）的加载层——**邮件 + KB 问答，记忆是文档不是对话**。

## 这份数据长什么样（2026-09-30 实测，`benchmark_data/corporatebench/`）

```text
data/kb/zenith.kb             zip：documents/*.txt 354 份（带 Message-ID/From/Date/Subject 的邮件）
                                    + graph.nq（N-Quads 真值图）+ ontologies/
data/kb_qa/zenith_questions.json       250 题   答案类型 int/str/bool/date/List[str]
data/topic_qa/zenith_questions.json    250 题
data/integrated_qa/zenith_questions.json 250 题
```

## 三条**显式的本地约定**

1. **一条文档 = 一条消息**，而**整个语料共用一个 `Session`**。⚠ 曾经是一条文档一个
   `Session`（理由：旧规则下连续同 role 合并会把它们塌成一块）——**D29 之后那个理由没有了**
   ⇒ 353 次 Add 降到 `batches()` 的 41 次，而块数一字不变（353）。
   `--limit` 选的是"哪个 QA 子集"，**语料不跟着裁**（裁了就没法检索）。
2. **三个 QA 文件 = 三个 `Sample`**（各自 250 题、**共用同一份语料**）——这样 `--limit 1`
   能只跑 `kb_qa`，而三家的分数可比（语料相同、问题类型不同）。
3. **`role` 一律 `user`**：邮件是"用户侧的资料"。

## 判分：**可字符串判定**（README 的 "Tasks and evaluation"）

标量按大小写不敏感的精确匹配；`List[str]` 按 **set-F1**（README 原文）。
⇒ `extra_pipeline.py` 里是纯函数，**不用 LLM 裁判**——这是三份里最便宜的一份。
"""

from __future__ import annotations

import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .preprocess import Message, Question, Sample, Session

__all__ = [
    "DATA_DIR",
    "KB_FILE",
    "PIPELINE",
    "QA_TYPES",
    "SHAPE_NOTE",
    "USER_PREFIX",
    "load_corporatebench",
]

DATA_DIR = "corporatebench"

#: KB 是 zip 容器——文档在 `documents/*.txt`（其余是 `graph.nq` 与 ontologies，不喂）。
KB_FILE = "data/kb/zenith.kb"

#: 本数据集**没有官方 AML pipeline**——这条指向我们自己的实现。
PIPELINE = "extra_pipeline.py"

USER_PREFIX = "corp-"

#: 三家 QA（顺序即 `Sample` 顺序）。
QA_TYPES = ("kb_qa", "topic_qa", "integrated_qa")

#: 写进**数据指纹**的 `note`（§13）——形状是我们造的，见模块 docstring 的三条约定。
SHAPE_NOTE = (
    "⚠ **形状是本地约定**：一条文档 = 一条消息，**整份语料共用一个 session**；"
    "三种 QA 各自一个 Sample 且**共用同一份语料**（`limit` 只选 QA 子集，不裁语料）"
)

#: 邮件头的 `Date: 2024-01-23`。取不到就留空（不是漏读）。
_DATE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)


def _timestamp_ms(body: str) -> int | None:
    match = _DATE.search(body)
    if not match:
        return None
    moment = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _documents(bench_dir: Path) -> tuple[Session, ...]:
    path = Path(bench_dir) / DATA_DIR / KB_FILE
    if not path.exists():
        raise FileNotFoundError(f"缺 {path}（`make fetch-data` 取回 official-extra 那一档）")
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith("documents/"))
        if not names:
            raise ValueError(f"{path.name}：zip 里没有 documents/*.txt")
        messages: list[Message] = []
        for name in names:
            body = archive.read(name).decode("utf-8", "replace")
            messages.append(Message(role="user", content=body, timestamp_ms=_timestamp_ms(body)))
    # ⚠ **整个语料一个 session**（曾经一条文档一个）：D29 之后"连续 user 各自独立成块"，
    #   挤在一次 Add 里粒度保得住，而 Add 次数少一个数量级（353 次 → 18 次）。
    return (Session(session_id="corpus", messages=tuple(messages)),)


def load_corporatebench(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 CorporateBench（zenith）。

    `limit` 选的是 **QA 子集**（三个 `Sample`），语料恒为全部 354 份文档。
    `spread` 在这里没有意义（只有 3 组、每组语料相同）——传了也不改变顺序。
    """
    sessions = _documents(bench_dir)
    qa_types = QA_TYPES[:limit] if limit is not None else QA_TYPES

    samples: list[Sample] = []
    for qa_type in qa_types:
        path = Path(bench_dir) / DATA_DIR / "data" / qa_type / "zenith_questions.json"
        if not path.exists():
            raise FileNotFoundError(f"缺 {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("questions") or []
        if not rows:
            raise ValueError(f"{path.name}：questions 为空")
        questions = []
        for row in rows:
            if row.get("answer") is None:
                raise ValueError(f"{path.name}:{row.get('id')}：answer 为空——判不了分")
            questions.append(
                Question(
                    qid=f"{qa_type}-{row.get('id')}",
                    question=str(row["question"]),
                    # 一整个 dict 进 gold：判分既要答案，也要 `answer_type` 才知道怎么比。
                    gold={"answer": row["answer"], "answer_type": row.get("answer_type")},
                    category=qa_type,
                )
            )
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{qa_type}",
                dataset="corporatebench",
                sessions=sessions,
                questions=tuple(questions),
                speaker_names=("user", "assistant"),
            )
        )
    return samples
