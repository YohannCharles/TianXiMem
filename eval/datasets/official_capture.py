"""官方采集流量的加载层——**一次 Add 原样重投，不重新切批、不重排**。

> **数据来源**：[`official-dataset-2026-09-29/`](../../official-dataset-2026-09-29/)（不进 git）。
> 它是我们服务在 2026-09-29 那一轮收到的**全部**官方 add/search 请求原文，
> 已被 [`tools/build_official_kit.py`](../../tools/build_official_kit.py) 整理成
> `official-eval-kit.jsonl`（一题一行、带归属与金标）。

## 这个数据集与独立加载器结构不同

| | 其余数据集 | `official-capture` |
| --- | --- | --- |
| Add 的 payload | **我们造**（`add_shape` 加前缀、切批） | **AML 造的原文**——重投即是复现 |
| 顺序 | 先灌完整个 sample，再逐题检索 | **按 timeline 交错**（见下「未来泄漏」） |
| 题与语料的对应 | 数据集自带 | 采集的 `prior_adds_for_user` 记着每题之前的 add 数 |

⇒ 所以这里**故意不接** `add_shape` / `batching`：重投原文再加一遍标签就是加了两遍；
重切批会改变批界（**D24 下批界直接决定记忆块的形状**）。两处都是静默出错。

⚠ **代价照实说**：别的数据集上"本地发的像不像线上"是个**近似问题**
（`add_shape` 那 93.9% 就是这么量出来的）；这一份是**逐字节相同**，
所以它上面的分数**比任何别的数据集都更接近线上**——但也因此**与它们不可比**。

## 未来泄漏：为什么必须按 timeline 走

全量 43,272 条 add 里，**88.3% 的 user 是"先 add 完再 search"**，但剩下 12% 是交错的。
本仓有实测量：6,146 道可评分的题里，**2,507 道（41%）的提问时刻早于该 user 的最后一条 add**
⇒ 若先把整个 user 的语料灌完再检索，这 2,507 道会**读到未来**（README §5.4 的同一条）。

⇒ [`replay_plan()`][eval.datasets.official_capture.replay_plan] 按采集里**真实的时间序**
把 add 与 search 排成一条流——这正是 README §7「顺序只约束 user 内」的落地。

## 只加载**可评分**的题

`official-eval-kit.jsonl` 的 10,144 行里只有 6,146 行有金标或官方判分
（`judge_kind != "none"`）。没标签的题**跑了也没有分数**，而解答+裁判要花真金白银的
网关时间 ⇒ 缺省只加载可评分的那批。**要全量复放**（例如量检索副作用）传
`--all-questions`（见 [`replay_official.py`](../../eval/experiments/replay_official.py)）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .preprocess import Message, Question, Sample, Session
from .registry import capture_dir

__all__ = [
    "ADDS_FILE",
    "KIT_FILE",
    "TIMELINE_FILE",
    "ReplayEvent",
    "UserPlan",
    "load_official_capture",
    "plan_for",
    "replay_plan",
]

KIT_FILE = "official-eval-kit.jsonl"
ADDS_FILE = "official-adds.jsonl"
TIMELINE_FILE = "official-timeline.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    """时间序上的一步：一次 `Add` 或一次 `Search`。"""

    seq: int
    ts: str
    kind: str
    #: `add` 专用——**采集原文的那四个字段**，重投时逐字发出去。
    request_id: str | None = None
    session_id: str | None = None
    messages: tuple[dict, ...] = ()
    #: `search` 专用——`official-eval-kit.jsonl` 的那一行（含金标与归属）。
    question: dict | None = None


@dataclass(frozen=True, slots=True)
class UserPlan:
    """一个 user 的完整重放计划（**user 间无依赖，可任意穿插**——README §7）。"""

    user_id: str
    events: tuple[ReplayEvent, ...]
    dataset: str = "official-capture"

    @property
    def questions(self) -> tuple[dict, ...]:
        return tuple(e.question for e in self.events if e.question is not None)

    @property
    def add_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "add")

    def to_sample(self) -> Sample:
        """给 `run_record` / `summarize` 用的轻量 `Sample`——**只为记录与分类**。

        `sessions` 留空：语料已经以 [`ReplayEvent`][eval.datasets.official_capture.ReplayEvent]
        的原文形态躺在计划里，这里再装一遍只会在两处之间制造漂移。
        """
        questions = tuple(
            Question(
                qid=str(row["seq"]),
                question=str(row["query"]),
                gold=row,
                category=str(row["dataset"]),
                evidence=(),
                is_abstention=False,
            )
            for row in self.questions
        )
        return Sample(
            user_id=self.user_id,
            dataset=self.dataset,
            sessions=(),
            questions=questions,
        )


def _messages(record: dict) -> tuple[dict, ...]:
    """采集原文的 `messages`——**逐字保留**（含首尾空白与缺 `timestamp` 的那种）。

    ⚠ 与其余加载器的口径**故意不同**：那边 `normalize_content` 会 `strip()`，
    因为 payload 由我们造；这里 payload 是 AML 造的，`strip()` 就不是原样重投了。
    全量实测：454,937 条消息里 0 条空正文、3,669 条有首尾空白、21,434 条没有 `timestamp`。
    """
    return tuple(dict(message) for message in record.get("messages") or [])


def _adds_by_request(adds_path: Path, wanted_users: set[str]) -> dict[str, dict]:
    """`request_id` → 采集原文那条 add（只留 `wanted_users` 的——原文 300+ MB）。"""
    index: dict[str, dict] = {}
    with adds_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("user_id") in wanted_users:
                index[record["request_id"]] = record
    return index


def replay_plan(
    dataset_dir: Path | None = None,
    *,
    only_users: list[str] | None = None,
    scorable_only: bool = True,
    max_questions: int | None = None,
) -> list[UserPlan]:
    """按 timeline 排出一批 user 的重放计划。

    **排序 = 可评分题数降序**（并列按 `user_id`）——`--users N` 因此取到的是
    "分数产出最高的 N 个 user"，而不是"前 N 个 user"。实测这条很陡：
    **261 个 user 就覆盖 90% 的可评分题**（14,143 条 add），全量是 1,580 个 user / 43,272 条。
    """
    root = Path(dataset_dir) if dataset_dir is not None else capture_dir()
    kit = _read_jsonl(root / KIT_FILE)
    qa_by_seq = {
        int(row["seq"]): row for row in kit if not scorable_only or row["judge_kind"] != "none"
    }

    timeline = _read_jsonl(root / TIMELINE_FILE)
    wanted = set(only_users) if only_users else {row["user_id"] for row in qa_by_seq.values()}
    wanted &= {event["user_id"] for event in timeline}
    adds = _adds_by_request(root / ADDS_FILE, wanted)

    per_user: dict[str, list[ReplayEvent]] = {user: [] for user in wanted}
    dropped_searches = 0
    for event in timeline:
        user = event["user_id"]
        if user not in per_user:
            continue
        if event["kind"] == "add":
            record = adds.get(event["request_id"])
            if record is None:  # 只存在于旧导出里的那些（README §2 的边界）
                continue
            per_user[user].append(
                ReplayEvent(
                    seq=int(event["seq"]),
                    ts=event["ts"],
                    kind="add",
                    request_id=record["request_id"],
                    session_id=record["session_id"],
                    messages=_messages(record),
                )
            )
        else:
            row = qa_by_seq.get(int(event["seq"]))
            if row is None:
                dropped_searches += 1
                continue
            per_user[user].append(
                ReplayEvent(seq=int(event["seq"]), ts=event["ts"], kind="search", question=row)
            )

    plans: list[UserPlan] = []
    for user, events in per_user.items():
        events.sort(key=lambda e: (e.ts, e.kind != "add"))
        if max_questions is not None:
            keep = max_questions
            trimmed: list[ReplayEvent] = []
            for event in events:
                if event.kind == "search":
                    if keep <= 0:
                        continue
                    keep -= 1
                trimmed.append(event)
            events = trimmed
        if any(e.kind == "search" for e in events):
            plans.append(UserPlan(user_id=user, events=tuple(events)))
    plans.sort(key=lambda plan: (-len(plan.questions), plan.user_id))
    if dropped_searches:
        print(f"  （跳过 {dropped_searches} 条 search：不在可评分集合里）", flush=True)
    return plans


def plan_for(user_id: str, dataset_dir: Path | None = None, **kwargs) -> UserPlan | None:
    """只要一个 user 的计划（冒烟用）。找不到返回 `None`。"""
    for plan in replay_plan(dataset_dir, only_users=[user_id], **kwargs):
        return plan
    return None


def load_official_capture(
    dataset_dir: Path | None = None,
    *,
    limit: int | None = None,
    scorable_only: bool = True,
) -> list[Sample]:
    """把重放计划转成 `Sample` 列表——**给记录层与测试用**，重放本身走 `replay_plan`。

    `sessions` 是采集原文的 add，一条 add 一个 `Session`（`message_count` 因此是真实的）。
    """
    plans = replay_plan(dataset_dir, scorable_only=scorable_only)
    if limit is not None:
        plans = plans[:limit]
    samples: list[Sample] = []
    for plan in plans:
        adds = [event for event in plan.events if event.kind == "add"]
        sessions = tuple(
            Session(
                session_id=event.session_id or f"seq-{event.seq}",
                messages=tuple(
                    Message(
                        role=str(message.get("role") or "user"),
                        content=str(message.get("content") or ""),
                        timestamp_ms=message.get("timestamp"),
                    )
                    for message in event.messages
                ),
            )
            for event in adds
        )
        sample = plan.to_sample()
        samples.append(
            Sample(
                user_id=sample.user_id,
                dataset=sample.dataset,
                sessions=sessions,
                questions=sample.questions,
            )
        )
    return samples
