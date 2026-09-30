"""CL-Bench 的加载层（六份数据集里契约最不同的一份）。

## 这份数据长什么样（2026-09-27 实测）

```text
1899 条任务 / 500 个上下文（平均 3.8 题/上下文），每条：
  messages  [system, user, (assistant, user)…]  —— system 是角色指令；
             **user 里装着整份"文档"（规则书 / 手册 / 转录稿，最长 30.9 万字符），
             任务指令在它的最末尾**
  rubrics   判分标准（中位 13 条，最多 114 条）——**严格全有全无**
  metadata  {task_id, context_id, context_category, sub_category}
```

## ⚠ 两条**显式的代理假设**（没有它们就跑不起来，且都不是事实）

### 1. 一道题一个 `Sample`（`user_id = clb-<task_id>`）

同一 `context_id` 下的各条任务**消息并不共享**（实测最长公共前缀 = 0），
差的那些消息就是文档本身 ⇒ 每题自带自己的上下文，像 LongMemEval 那样"1 题 : 1 份记忆"。

### 2. **任务文本 = 末条 user 消息的最后 6,000 字符**

CL-Bench 没有单独的 `question` 字段：**任务指令写在末条 user 的最末尾**
（实测：`…what do Sightings Cards do?`、`Write a concise email to my client…`、
`Component 5: Final Task / Based on …, provide your single response…`）。
末**段**中位只有 434 字符，但有的记录把整本手册塞成一段（最长 13.4 万）
⇒ 取**尾部固定窗口**比"取末段"稳。

⚠ **由此产生的一条口径**：**文档不在问题里，只在记忆里** ⇒ 我们的检索在这份数据上
**是必需的**（检索不到就答不了）。真实平台也许会把整条 user（含文档）当问题传，
那样我们的检索就只是"额外塞进去的东西"——**那条口径没测**，选这条的理由是
`retrieval.selected` 这条通路本来就是为"从记忆里取回证据"设计的。

## ⚠ 与 LoCoMo/LongMemEval 的三处契约差异（不要在别处复用本模块的结论）

| | 那两份 | **CL-Bench** |
| --- | --- | --- |
| 记忆注入字段 | `speaker_1_memories`（平铺字符串） | 嵌套 `retrieval.selected`，
每项 `created_at` + **`text`** |
| 判分 | 二元 CORRECT/WRONG + 严格时间规则 | **rubric 全有全无**（有一条不满足即 0） |
| 失败口径 | 抛错 | **API/JSON 失败一律记 0**（§12.4） |
"""

from __future__ import annotations

import json
from pathlib import Path

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = ["CLBENCH_JSONL", "PIPELINE", "TASK_TAIL_CHARS", "USER_PREFIX", "load_clbench"]

#: 归档里的文件名——**数据集专属文件名的知识止步于本模块**（D16）。
CLBENCH_JSONL = "clbench.jsonl"

#: 归档里的 pipeline 脚本名（harness 侧 `pipeline_for` 用它）。
PIPELINE = "clb_pipeline.py"

#: `user_id` / `qid` 的前缀——一眼能看出这份记录来自哪份数据集。
USER_PREFIX = "clb-"

#: 任务文本取"末条 user 消息的最后多少字符"（见模块 docstring 的假设 2）。
TASK_TAIL_CHARS = 6_000


def _task_text(entry: dict) -> str:
    """末条 user 消息的尾部窗口——**任务指令就在那儿**（假设 2）。"""
    messages = entry.get("messages") or []
    last_user = next(
        (m for m in reversed(messages) if str(m.get("role", "")).lower() == "user"), None
    )
    if last_user is None:
        return ""
    return str(last_user.get("content") or "").strip()[-TASK_TAIL_CHARS:]


def _category(entry: dict) -> str:
    meta = entry.get("metadata") or {}
    top = str(meta.get("context_category") or "").strip()
    sub = str(meta.get("sub_category") or "").strip()
    return f"{top} / {sub}" if top and sub else top or sub


def load_clbench(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 CL-Bench。

    ⚠ **两部分跑要加 `spread=True`**：文件按 `context_category` 排，
    否则 `entries[:limit]` 只取到一类——理由与处置见 [`sampling.py`](./sampling.py)。
    """
    source = Path(bench_dir) / CLBENCH_JSONL
    # ⚠ **按 `"\n"` 切，不用 `splitlines()`**：这份文件里**真的有 `U+2028`**
    #   （实测：`splitlines()` 会在 char 22463 处把一条记录劈成两半 ⇒ `JSONDecodeError`）。
    #   它是 `"\n"` 分隔的 JSONL ⇒ **按写它的方式读**。同一条纪律见
    #   [`../harness/judge.py`](../harness/judge.py) 的 `_jsonl_line`。
    entries = [
        json.loads(line) for line in source.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    if limit is not None:
        entries = (
            # ⚠ 抽样键只用 **`context_category`**（4 类）。用 `_category()`（含 sub_category）
            #   会得到几十个组，而"每组至少 1 条"⇒ `--limit 2` 实际会加载 30 多条。
            stratified_sample(
                entries,
                limit,
                key=lambda e: str((e.get("metadata") or {}).get("context_category") or ""),
            )
            if spread
            else entries[:limit]
        )

    samples: list[Sample] = []
    for entry in entries:
        meta = entry.get("metadata") or {}
        task_id = str(meta.get("task_id") or meta.get("context_id") or "")
        if not task_id:
            raise ValueError(f"{source.name}：一条记录既没有 task_id 也没有 context_id")
        question = _task_text(entry)
        if not question:
            raise ValueError(f"{source.name}:{task_id}：取不到任务文本（末条 user 为空？）")
        rubrics = entry.get("rubrics")
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError(f"{source.name}:{task_id}：rubrics 缺失或为空——没有它判不了分")
        sessions = (
            Session(
                session_id=task_id,
                # ⚠ 这份数据**没有时间戳** ⇒ `timestamp_ms=None` ⇒ `event_time` 为 NULL
                #   ⇒ `created_at` 发空串（§11.3 那条有定义的降级路径）。**不是漏读**。
                messages=tuple(
                    Message(role=str(m.get("role") or "user"), content=str(m.get("content") or ""))
                    for m in entry.get("messages") or []
                ),
            ),
        )
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{task_id}",
                dataset="clbench",
                sessions=sessions,
                questions=(
                    Question(
                        qid=task_id,
                        question=question,
                        # gold 是 **rubric 列表**——`clb_pipeline.py` 的 `official_rubrics()`
                        # 认 `item["rubrics"]`，harness 会原样放进 item。
                        gold=rubrics,
                        category=_category(entry),
                    ),
                ),
                speaker_names=("user", "assistant"),
            )
        )
    return samples
