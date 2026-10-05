"""PersonaMem-v2 的加载层——⚠ **它的 pipeline 不读检索字段**（这点必须记住）。

## 这份数据长什么样（2026-09-30 实测）

```text
benchmark/text/benchmark.csv     5,000 题 / 200 个 persona（每人 6–42 题）
                                 每题：user_query（repr 串，取里面的 content）、
                                       correct_answer（长文本）、incorrect_answers（干扰项）
data/chat_history_32k/*.json     200 份（**只取了 benchmark 引用到的那些**）
                                 {"metadata": {...}, "chat_history": [message, ...]}，首条是 system
```

## ⚠ 两条与别的数据集不同的地方

1. **它没有"检索"这回事**：官方 `pipeline_v2_personamem.py` 从 `chat_history` 直接拼 prompt，
   **完全不看任何检索字段**（`../CLAUDE.md` 那张契约表记着这一条）。
   ⇒ 接进来**测不出架构改动**，它的价值是**回归对照**：如果某个改动把它也弄坏了，
   那大概率是装配/契约层出了问题，**而不是排序**。
2. **一个 `Sample` = 一个 persona**（记忆 = 它的整段 chat_history，题 = 它的全部题）——
   与 LongMemEval 同形（1 份记忆 : 多道题），只是记忆不走检索。

判分走**官方那份**（MCQ 模式：把 `correct_answer` + `incorrect_answers` 拼成选项、
**按种子洗牌**后让模型选字母，再比字母）——见
[`../harness/personamem_pipeline.py`](../harness/personamem_pipeline.py)。
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

from .preprocess import Message, Question, Sample, Session

__all__ = ["BENCHMARK_CSV", "DATA_DIR", "PIPELINE", "USER_PREFIX", "load_personamem"]

DATA_DIR = "personamem-v2"

BENCHMARK_CSV = "benchmark/text/benchmark.csv"

#: **我们的适配器**（内部直接调官方那份的两个函数）。
PIPELINE = "personamem_pipeline.py"

USER_PREFIX = "pm-"


def _query_text(raw: str) -> str:
    """`user_query` 列是 Python dict 的 repr 串（`{'role': 'user', 'content': …}`）⇒ 取 content。"""
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            payload = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(payload, dict):
            return str(payload.get("content") or payload.get("text") or "")
    return text


def _history(path: Path) -> tuple[Message, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("chat_history")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path.name}：chat_history 缺失或为空")
    return tuple(
        Message(role=str(m.get("role") or "user"), content=str(m.get("content") or ""))
        for m in messages
    )


def load_personamem(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 PersonaMem-v2——**一个 `Sample` = 一个 persona**。

    ⚠ 部分跑要加 `spread=True`（CSV 按 persona 排，`[:limit]` 会只取到前几个 persona）。
    """
    root = Path(bench_dir) / DATA_DIR
    csv_path = root / BENCHMARK_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"缺 {csv_path}（`make fetch-data` 取回 official-extra 那一档）")

    by_persona: dict[str, list[dict]] = {}
    history_path: dict[str, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            persona = str(row["persona_id"])
            by_persona.setdefault(persona, []).append(row)
            link = (row.get("chat_history_32k_link") or "").strip()
            if link:
                history_path.setdefault(persona, link)

    personas = sorted(by_persona, key=lambda p: int(p))
    if limit is not None:
        # `spread`：按题量分层没有意义（每人都是一份记忆），这里按 persona 序号取即可。
        personas = personas[:limit]

    samples: list[Sample] = []
    for persona in personas:
        link = history_path.get(persona)
        if not link:
            raise ValueError(f"persona {persona}：CSV 里没有 chat_history_32k_link")
        path = root / link
        if not path.exists():
            raise FileNotFoundError(
                f"缺 {path}——PersonaMem 的语料**不在归档的公开清单里**，"
                "见 tools/fetch_benchmark_data.py 的 official-extra 说明"
            )
        questions = []
        for index, row in enumerate(by_persona[persona]):
            try:
                incorrect = json.loads(row.get("incorrect_answers") or "[]")
            except json.JSONDecodeError:
                incorrect = []
            questions.append(
                Question(
                    qid=f"pm-{persona}-{index}",
                    question=_query_text(row.get("user_query", "")),
                    # 官方 `official_mcq_options()` 认这两个键名（不是我们自定的桶）。
                    gold={
                        "correct_answer": row.get("correct_answer") or "",
                        "incorrect_answers": [str(a) for a in incorrect],
                        "persona_id": persona,
                    },
                    category=str(row.get("topic_query") or row.get("pref_type") or ""),
                )
            )
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{persona}",
                dataset="personamem-v2",
                sessions=(
                    Session(
                        session_id=f"pm-{persona}",
                        # ⚠ 这份数据**没有时间戳** ⇒ `event_time` 为 NULL、`created_at` 发空串。
                        messages=_history(path),
                    ),
                ),
                questions=tuple(questions),
                speaker_names=("user", "assistant"),
            )
        )
    return samples
