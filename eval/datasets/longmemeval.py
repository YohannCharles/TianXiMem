"""LongMemEval（`lme_s_cleaned.json`）加载器。

> ⚠ **只用 `lme_s_cleaned.json`，不要用 `lme_test.json`。** 后者多出 **1,230 个
> 0-turn session**，而空 session 会污染按"20 条消息"切批的埋点逻辑（§6.5）。
> 那份文件**不在归档里**（它就是等着被误用的坑）；要复核这条结论得先从上游取回它，
> 线索在 `tools/fetch_benchmark_data.py` 的 `DELETED` 段。

## 基数与 LoCoMo **正好相反**：一题一个 haystack

`lme_s_cleaned.json` 是 500 条记录，**每条自带一整套 session**（首条就有 53 个 session）。
所以 `user_id` **按题派生**——同一道题的 haystack 只服务这一道题。
这不是实现选择，是数据集的形状：LME 的问法（"What degree did I graduate with?"）
要求先确定"我是谁"，而"我"由这道题的 haystack 定义。

## 时间在 session 级，**必须为每条消息合成 `timestamp`**

turn 的键只有 `role` + `content`（可选 `has_answer`），时间在**与 `haystack_sessions`
按下标平行**的 `haystack_dates` 里（形如 `"2023/05/20 (Sat) 02:21"`）。
不合成则 `event_time` 全 NULL、`created_at` 只能发 `""`——
**而同一个 session 内所有消息拿到同一个日期，所以 `pair_idx` 是唯一能保证
邻域稳定的东西**（§6.1）。

## 归档里 `question_date` / `has_answer` / `answer_session_ids` **都没有 pipeline 读它们**

已核对：`pipeline_longmemeval-s.py` 与 `pipeline_locomo-refined.py` **除 docstring 外
逐字节相同**，两者都只读 `id` / `question` / gold 四键之一 / `speaker_*_memories`。
`answer_session_ids` 在这里只作为**诊断证据**保留（§13 的 T2 要用），不进 Add 载荷。
"""

from __future__ import annotations

import json
from pathlib import Path

from .preprocess import (
    Message,
    Question,
    Sample,
    Session,
    normalize_content,
    parse_lme_time,
    to_epoch_ms,
)

__all__ = ["load_longmemeval", "LME_JSON", "PIPELINE"]

LME_JSON = "lme_s_cleaned.json"

#: 归档里本数据集的 answer/judge 脚本——harness 用 subprocess 调它，不 import。
PIPELINE = "pipeline_longmemeval-s.py"

#: `user_id` 的前缀——把 LME 的 user 与 LoCoMo 的 `conv-*` 在同一个库里区分开。
USER_PREFIX = "lme-"


def _sessions(entry: dict, *, source: Path) -> tuple[Session, ...]:
    """把 `haystack_sessions` / `haystack_dates` / `haystack_session_ids` 三条平行数组缝起来。

    平行关系是这套数据的**结构性前提**，长度不等就说明取错了文件 ⇒ 抛。
    """
    sessions_raw = entry["haystack_sessions"]
    dates = entry["haystack_dates"]
    ids = entry["haystack_session_ids"]
    if not (len(sessions_raw) == len(dates) == len(ids)):
        raise ValueError(
            f"{source.name}:{entry['question_id']}：haystack 三条平行数组长度不等 "
            f"({len(sessions_raw)}/{len(dates)}/{len(ids)})——这不是 lme_s_cleaned.json 的形状"
        )

    sessions = []
    # `strict=True`：上面已断言长度相等，这里再钉一次——**长度不等时宁可抛，
    # 也不要 zip 静默截断到最短**（那会丢掉 session 且不报错）。
    for session_id, date_str, turns in zip(ids, dates, sessions_raw, strict=True):
        ms = to_epoch_ms(parse_lme_time(date_str))
        messages = tuple(
            Message(
                role=turn["role"],
                content=normalize_content(
                    turn["content"], where=f"{source.name}:{session_id}:{position}"
                ),
                timestamp_ms=ms,
            )
            for position, turn in enumerate(turns, start=1)
        )
        sessions.append(Session(session_id=session_id, messages=messages))
    return tuple(sessions)


def load_longmemeval(bench_dir: str | Path, *, limit: int | None = None) -> list[Sample]:
    """加载 LongMemEval-S。

    `limit` 只给冒烟与调试用（这份文件 277 MB，全量解析要几十秒、几 GB 内存）。
    **正式跑必须不传**——截断的题集会改变分数，而 `registry` 的数据指纹会如实记下
    实际用了几题，所以不留"看起来跑全了"的空间。
    """
    source = Path(bench_dir) / LME_JSON
    entries = json.loads(source.read_text(encoding="utf-8"))
    if limit is not None:
        entries = entries[:limit]

    samples = []
    for entry in entries:
        qid = entry["question_id"]
        gold = entry["answer"]
        if not gold:
            raise ValueError(f"{source.name}:{qid}：gold 为空——该题不该走判分路径")
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{qid}",
                dataset="longmemeval-s",
                sessions=_sessions(entry, source=source),
                questions=(
                    Question(
                        qid=qid,
                        question=normalize_content(entry["question"], where=f"{source.name}:{qid}"),
                        gold=gold,
                        category=entry["question_type"],
                        evidence=tuple(entry.get("answer_session_ids") or ()),
                        is_abstention=qid.endswith("_abs"),
                    ),
                ),
            )
        )
    return samples
