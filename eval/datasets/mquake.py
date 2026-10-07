"""MQuAKE-Remastered 的加载层（官方套件里最"不像对话"的一份）。

## 这份数据长什么样（2026-09-30 实测，`benchmark_data/mquake-remastered/`）

```text
4 份 parquet（CF3k / CF6334 / CF9k / T），每行一个 case：
  orig_triples[_labeled]     改写**前**的事实链（Q-id 三元组 / 自然语言三元组）
  edit_triples               改写本身（Q-id 三元组）
  requested_rewrite[]        改写的可读形式（subject / relation / target_new_str）
  new_triples[_labeled]      改写**之后**的链
  questions[]                同一个 case 的 3 种问法（答案都相同）
  answer / answer_alias      改写前的答案       ← ⚠ 我们**不用**这一侧
  new_answer / new_answer_alias  改写后的答案   ← ✅ 金标
```

## 两条**显式的本地约定**（数据集没有这些概念，是我们造的）

1. **一条事实 = 一条消息**（不是一页、更不是一整段），而**整个样本共用一个 `Session`**。
   ⚠ 能这么挤是因为 **D29**：没有非 user 跟随的连续 user **每条各自独立成块** ⇒ 粒度一样，
   而 Add 次数是 `batches()` 的 **12**（一条一个 `Session` 要 **221** 次）。
2. **一个 `Sample` = `cases_per_user` 个连续 case**（默认 64）。MQuAKE 本身没有"用户"这个概念，
   但评测需要"记忆远多于证据"才有检索可言（一个 case 只有 2–4 条事实）。
   ⇒ 这些 case 互为干扰项，题目取自它们自己的 `questions`。

## 记忆文本：**我们拼的**，不是数据集给的

* 事实：`"<subject> — <relation> — <object>"`（取自 `*_labeled`）
* 改写：**一句自然语言**，由 `requested_rewrite[].prompt` 填出来——
  `"Ellie Kemper is a citizen of Croatia (this replaces the earlier value)"`。
  ⚠ 写法是实测选的：三元组形状（`… — P27 — Croatia`）**模型会答"记忆里没有"**，
  自然句才对。测法与结论见 `eval/reports/ledger.md`。

⚠ 记忆里**同时**有原始事实与 UPDATE ⇒ **金标一律取 `new_answer`**，不存在
"该取旧的还是新的"那种歧义（[`tools/recover_official_gold.py`](../../tools/recover_official_gold.py)
里那条逐条判定是给**采集到的官方流量**用的——那边的记忆不由我们构造）。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = [
    "CASES_PER_USER",
    "DATA_DIR",
    "PIPELINE",
    "SHAPE_NOTE",
    "USER_PREFIX",
    "load_mquake",
]

#: 归档里的子目录（D16：数据集专属文件名的知识止步于本模块）。
DATA_DIR = "mquake-remastered"
DATA_SUBDIR = "data"

#: 本数据集**没有官方 pipeline**——这条指向我们自己的实现（见 `eval/harness/extra_pipeline.py`）。
PIPELINE = "extra_pipeline.py"

USER_PREFIX = "mqk-"

#: 一个 `Sample` 装多少个 case（约定，不是数据集事实）。
CASES_PER_USER = 64

#: 写进**数据指纹**的 `note`（§13）——上面两条约定是**我们造的**，改它会改分数，
#: 而两次 run 的 record 否则会长得一模一样。与切批口径同一条理由
#: （[`../harness/batching.py`](../harness/batching.py)："常量不是旋钮"）。
SHAPE_NOTE = (
    "⚠ **形状是本地约定**：一条事实/改写 = 一条消息，**整个 Sample 共用一个 session**；"
    f"一个 Sample = {CASES_PER_USER} 个 case（MQuAKE 没有「用户」这个概念，"
    "捆起来是为了让记忆远多于证据——一个 case 只有 2–4 条事实，检索无从谈起）"
)

#: ⚠ 四份文件的 `case_id` **各自从 1 开始**，互不重叠地拼 `sample_id` 时必须带上文件名。
FILES = ("CF3k", "CF6334", "CF9k", "T")


def _triple_text(subject: str, relation: str, object_: str) -> str:
    return f"{subject} — {relation} — {object_}"


def _update_text(rewrite: dict) -> str:
    """把改写拼成**一句自然语言**——⚠ 这行是实测调出来的，别改回三元组形状。

    `requested_rewrite[].prompt` 形如 `"{} is a citizen of"`，填上 subject 就是人话：

        Ellie Kemper is a citizen of Croatia (this replaces the earlier value).

    **为什么不用 `"UPDATE: … — P27 — Croatia"`**：2026-09-30 用真模型量过四种写法，
    只有**自然句**那一种答得对（见 `eval/reports/ledger.md`）——`relation_id`（`P27`）
    对语言模型是噪声，而"替换掉旧值"这层语义又必须说出来（否则与原始事实直接矛盾）。
    """
    subject = str(rewrite.get("subject", ""))
    target = str(rewrite.get("target_new_str", ""))
    template = str(rewrite.get("prompt") or "")
    head = template.format(subject) if "{}" in template else subject
    return f"{head} {target} (this replaces the earlier value)".strip()


def _rows(path: Path) -> Iterator[dict]:
    """流式读——四份加起来 27272 行，一次一行不占内存。"""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        yield from batch.to_pylist()


def _memory_messages(row: dict) -> list[Message]:
    """一个 case 的记忆：原始事实 + 改写——**一条消息一条事实**（顺序即事实链的顺序）。

    ⚠ **整个 case 挤在一个 `Session` 里是对的**：**D29** 之后一段没有非 user 跟随的连续
    user，**每条各自独立成块** ⇒ 粒度照样保得住，而 Add 次数少一个数量级。
    """
    messages: list[Message] = []
    for triple in row.get("orig_triples_labeled") or []:
        subject, relation, object_ = (str(x) for x in triple)
        messages.append(Message(role="user", content=_triple_text(subject, relation, object_)))
    for rewrite in row.get("requested_rewrite") or []:
        messages.append(Message(role="user", content=_update_text(rewrite)))
    return messages


def _questions(row: dict, stem: str) -> tuple[Question, ...]:
    """一个 case 的 3 种问法——**同一个答案**，所以三条共享同一份金标。"""
    answer = row.get("new_answer")
    if not answer:
        raise ValueError(f"{stem}:{row['case_id']}：没有 new_answer——金标取不到")
    gold = [str(answer)] + [str(a) for a in (row.get("new_answer_alias") or []) if a]
    questions = []
    for index, text in enumerate(row.get("questions") or []):
        questions.append(
            Question(
                qid=f"{stem}-{row['case_id']}-{index}",
                question=str(text),
                # 别名表也进 gold：判分时**任一命中即对**（大小写/空白归一化见 extra_pipeline）。
                gold=sorted(set(gold)),
                category=stem,
            )
        )
    return tuple(questions)


def load_mquake(
    bench_dir: str | Path,
    *,
    limit: int | None = None,
    spread: bool = False,
    cases_per_user: int = CASES_PER_USER,
) -> list[Sample]:
    """加载 MQuAKE-Remastered。

    ⚠ **部分跑要加 `spread=True`**：四份文件按 `CF3k → CF6334 → CF9k → T` 排，
    `entries[:limit]` 只会取到 `CF3k`——而四份的问法/事实并不相同。
    """
    picked = case_chunks(bench_dir, limit=limit, spread=spread, cases_per_user=cases_per_user)
    samples: list[Sample] = []
    for stem, chunk in picked:
        messages: list[Message] = []
        questions: list[Question] = []
        for row in chunk:
            messages.extend(_memory_messages(row))
            questions.extend(_questions(row, stem))
        first_case = chunk[0]["case_id"]
        user_id = f"{USER_PREFIX}{stem}-{first_case:05d}"
        samples.append(
            Sample(
                user_id=user_id,
                dataset="mquake-remastered",
                sessions=(Session(session_id=user_id, messages=tuple(messages)),),
                questions=tuple(questions),
                speaker_names=("user", "assistant"),
            )
        )
    return samples


def case_chunks(
    bench_dir: str | Path,
    *,
    limit: int | None = None,
    spread: bool = False,
    cases_per_user: int = CASES_PER_USER,
) -> list[tuple[str, list[dict]]]:
    """共享原始读取和抽样；原生与 AML 适配分别决定序列化及提问阶段。"""
    if cases_per_user < 1:
        raise ValueError(f"cases_per_user 必须 >= 1，收到 {cases_per_user}")
    root = Path(bench_dir) / DATA_DIR / DATA_SUBDIR
    chunks: list[tuple[str, list[dict]]] = []  # (文件名, 一个 Sample 的 case 列表)
    for stem in FILES:
        path = root / f"{stem}-00000-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺 {path}（`make fetch-data` 取回 official-extra 那一档）")
        rows = list(_rows(path))
        for start in range(0, len(rows), cases_per_user):
            chunks.append((stem, rows[start : start + cases_per_user]))

    if limit is not None:
        picked = (
            stratified_sample(chunks, limit, key=lambda pair: pair[0]) if spread else chunks[:limit]
        )
    else:
        picked = chunks

    return picked
