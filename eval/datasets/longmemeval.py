"""LongMemEval（`lme_s_cleaned.json`）加载器。

> ⚠ **只用 `lme_s_cleaned.json`，不要用 `lme_test.json`。** 后者多出 **1,230 个
> 0-turn session**，而空 session 会污染按"20 条消息"切批的埋点逻辑（§6.5）。
> 那份文件**不在归档里**（它就是等着被误用的坑）；要复核这条结论得先从上游取回它，
> 线索在 `eval/datasets/manifest.py` 的 `DELETED` 段。

## 基数与 LoCoMo **正好相反**：一题一个 haystack

`lme_s_cleaned.json` 是 500 条记录，**每条自带一整套 session**（首条就有 53 个 session）。
所以 `user_id` **按题派生**——同一道题的 haystack 只服务这一道题。
这不是实现选择，是数据集的形状：LME 的问法（"What degree did I graduate with?"）
要求先确定"我是谁"，而"我"由这道题的 haystack 定义。

## 时间在 session 级，**必须为每条消息合成 `timestamp`**

turn 的键只有 `role` + `content`（可选 `has_answer`），时间在**与 `haystack_sessions`
按下标平行**的 `haystack_dates` 里（形如 `"2023/05/20 (Sat) 02:21"`）。
不合成则 `event_time` 全 NULL、`created_at` 只能发 `""`——
**而同一个 session 内所有消息拿到同一个日期，所以位置是唯一能保证
邻域稳定的东西**（§6.1；D28 之后位置 = `(request_id, local_index)`，`request_id` 不解析）。

## 归档里 `question_date` / `has_answer` / `answer_session_ids` **都没有 pipeline 读它们**

已核对：`pipeline_longmemeval-s.py` 与 `pipeline_locomo-refined.py` **除 docstring 外
逐字节相同**，两者都只读 `id` / `question` / gold 四键之一 / `speaker_*_memories`。
`answer_session_ids` 在这里只作为**诊断证据**保留（§13 的 T2 要用），不进 Add 载荷。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .preprocess import (
    Message,
    Question,
    Sample,
    Session,
    is_blank_content,
    normalize_content,
    parse_lme_time,
    to_epoch_ms,
)
from .sampling import stratified_sample

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
    # ⚠ **haystack 内的 `session_id` 会重复**（2026-10-02 实测：**9 / 301 个样本**，
    # 例如 `lme-58bf7951` 的 57 个 session 里 `07b7a667_1` 出现两次，指向**两段不同的对话**）。
    #
    # 不消重的后果是**两重的，且都不报错**：
    #
    # 1. **`request_id` 撞车**——它由 `(user_id, session_id, 批序号)` 派生
    #    （[`harness/batching.py`](../../eval/harness/batching.py) 的 `request_id_for`），
    #    同一 id 的第二段会话会算出**同一个 `request_id`、不同的 payload** ⇒
    #    服务按 **D28 回 409**。**实测把整条 LME 基线链打死在 `[3/301]`**
    #    （`lme-58bf7951|07b7a667_1|0`）。那是**响亮**的那一半。
    # 2. **语义漂移**——`session_id` 在本仓是三个机制的作用域：配对、扩窗、
    #    **段合并的分组键**。两段无关对话共用一个 id ⇒ 它们被当成"同一个 session"。
    #    这一半**静默**（正是 `docs/open-questions.md` 的 S7 警告的那类）。
    #
    # ⇒ 首次出现原样保留，之后再出现加 `#n` 后缀。**确定性**（按数组顺序，不依赖哈希/随机），
    # 所以同一份文件每次给出同一批 id。⚠ 这样会改变那 9 个样本的 `id` 与段合并分组
    # ⇒ **与 2026-10-02 之前的 LME 数字严格来说不可比**（影响的题面很小，但它是比对的前提）。
    known = set(ids)
    seen: dict[str, int] = {}
    renamed: list[tuple[str, str]] = []
    out_ids: list[str] = []
    for raw_id in ids:
        seen[raw_id] = seen.get(raw_id, 0) + 1
        unique = raw_id
        if seen[raw_id] > 1:
            suffix = seen[raw_id]
            while unique in known:
                suffix += 1
                unique = f"{raw_id}#{suffix}"
            renamed.append((raw_id, unique))
        known.add(unique)
        out_ids.append(unique)
    if renamed:
        pairs = "、".join(f"{a} → {b}" for a, b in renamed)
        print(
            f"⚠ {source.name}:{entry['question_id']}：haystack 内 `session_id` 重复——"
            f"已就地消重（{pairs}）。不消重会让 `request_id` 撞车（服务回 409）、"
            f"并让两段无关对话落进同一个段合并分组。",
            flush=True,
        )

    # `strict=True`：上面已断言长度相等，这里再钉一次——**长度不等时宁可抛，
    # 也不要 zip 静默截断到最短**（那会丢掉 session 且不报错）。
    for session_id, date_str, turns in zip(out_ids, dates, sessions_raw, strict=True):
        ms = to_epoch_ms(parse_lme_time(date_str))
        messages = _messages(turns, session_id=session_id, timestamp_ms=ms, source=source)
        sessions.append(Session(session_id=session_id, messages=messages))
    return tuple(sessions)


def _messages(
    turns: list[dict], *, session_id: str, timestamp_ms: int, source: Path
) -> tuple[Message, ...]:
    """缝一个 session 的 turn；**空正文的 turn 跳过并告警**（**V14**）。

    ⚠ **为什么是跳过，而不是照旧抛**：归档那份里确实有空正文的消息——2026-09-28
    实测**10 条唯一**（12 次出现，跨 5 个 session；同一 session
    会被多道题的 haystack 复用，所以出现次数 > 唯一数）。而按 §11.3 的口径
    **空正文必须拒绝**（AML 只做 `"\\n".join(...)`，空正文会让相邻两项粘在一起）。
    两条都成立的唯一做法是**在评测层把它剔掉、并把剔掉这件事说出来**：

    | 做法 | 后果 |
    | --- | --- |
    | 照旧抛 | **全量 500 题根本跑不起来**——Step 5 的大跑批直接崩（不是慢） |
    | 静默放行 | 空正文进了 `join`，模型读到的两项粘连，**而没人知道** |
    | **跳过 + 告警** | 全量可跑；剔掉的是**零信息量**的那一条，且每次都被念出来 |

    ⚠ **代价要认**：剔掉之后该 session 后面的位置序号（`local_index`）各前移一位，
    于是这些题的 `id` 与"不剔"的世界不同。但**同一份数据每次跑都剔同样那几条**，
    所以两次 run 之间仍然可比（§13 要的正是这个）。

    **验收**（2026-09-28，全量）：`load_longmemeval("benchmark_data")` ⇒ **500 题 /
    246,738 条消息**（= 246,750 − 12），4.8 秒。
    """
    messages = []
    for position, turn in enumerate(turns, start=1):
        raw = turn["content"]
        if is_blank_content(raw):
            print(
                f"⚠ {source.name}:{session_id}:{position} 的 content 为空——已跳过该条（V14）。"
                "它零信息量，但**位置序号会前移一位**；照旧抛的话全量跑不起来。",
                file=sys.stderr,
            )
            continue
        messages.append(
            Message(
                role=turn["role"],
                content=normalize_content(raw, where=f"{source.name}:{session_id}:{position}"),
                timestamp_ms=timestamp_ms,
            )
        )
    return tuple(messages)


def load_longmemeval(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 LongMemEval-S。

    `limit` 只给冒烟与调试用（这份文件 277 MB，全量解析要几十秒、几 GB 内存）。
    **正式跑必须不传**——截断的题集会改变分数，而 `registry` 的数据指纹会如实记下
    实际用了几题，所以不留"看起来跑全了"的空间。

    ⚠ **`limit` 与 `spread` 是两件事**：`limit=N` 取**前 N 题**（文件按类型分块 ⇒
    很可能只有一类）；`spread=True` 才按比例**跨类**取。**"跑一部分"要用后者**
    ——理由见 [`sampling.py`](./sampling.py)。
    """
    from .layout import archive_file

    source = archive_file(bench_dir, LME_JSON)
    entries = json.loads(source.read_text(encoding="utf-8"))
    if limit is not None:
        entries = (
            stratified_sample(entries, limit, key=lambda e: str(e["question_type"]))
            if spread
            else entries[:limit]
        )

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
