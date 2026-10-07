"""TempReason 的加载层——**只有 L2/L3 能当记忆任务**（L1 的 context 全是空的）。

## 这份数据长什么样（2026-09-30 实测）

```text
test_l1.json / test_l1_future.json   ⛔ **context 是空串**、也没有 fact_context——
                                     它本来就是"不给上下文"的纯日期算术
                                     （"6 year 4 month after Nov, 1185"）⇒ **不是记忆任务**，
                                     所以本加载器只认 L2/L3。
test_l2.json   5,397 题 / 999 个唯一 context
test_l3.json   4,426 题 / 981 个唯一 context
  每行：question · date（锚点日期）· text_answers.text（可接受的答案串，字符串列表）
        context（维基式长文，中位 ~2.8–3.0k 字符，最长 114k）
        fact_context（时序三元组转成的句子："X works for Y from Jan, 1949 to Jan, 1953"）

## ⚠ 为什么 `fact_context` 是**记忆**

它**与正文页同等共享**，不给它这份数据集的分数就不可读：

* **同一页的各题共用同一份事实集**——实测（2026-10-03）60 个多题页**全部**成立
  ⇒ 与正文页一样是**共享的**，不是"几乎每题一份"。
* 而**正文页里往往没有时间区间**（维基传只写"曾任某校教授"，不写起止），
  问题问的是**某一天的雇主是谁** ⇒ 事实句是**唯一能推出答案的材料**。
* 它**不泄题**：那份事实集里**同时含 gold 和 `neg_answers`**（几个不同时间段的雇主），
  只有**挑对区间**才答得出——这正是这份数据集要考的。

**不给它**（`base-tr`，332 题）：gold 串只在 `fact_context` 里、不在正文页的 **239/332**；
错题里 **259/310 是逐字拒答**（`Cannot determine from the memories.`）
⇒ **overall 0.066 量的是"我们没给那份事实"**，与时间推理能力无关。
```

## 形状（**照"记忆远多于证据"这条评测量纲来的**）

* **一个 `Sample` = 一个唯一 context**（同一页被问 3–4 道题）。
* **页按句切成记忆**：一句 = 一条消息，而**整页共用一个 `Session`**。
  ⚠ 能这么挤是因为 **D29**：一段没有非 user 跟随的连续 user **每条各自独立成块** ⇒
  粒度一样、块数一字不变（31），而 Add 次数是 **2**（一句一个 `Session` 要 **31** 次）。
  ⚠ 一句话不是自然段：维基页里**答案就藏在一句话里**，按段切会让粒度太粗、检索变成"整页要不要"。
  ⚠ **切句不能切缩写**（`F.C.` / `Jr.` / `St.` 的点号不是句末）——见 `_ABBREVIATION_TAIL`。
  切错的后果是**事实句被拦腰截断、区间那半自成一块**（2026-10-04 修，实测 16.0% → 0.14%）。

## 判分：**我们自己写的**（字符串命中）

上游没有随数据发布评测脚本（我们只下了 4 个 json）⇒ 口径是我们定的：
**归一化后任一 `text_answers.text` 出现在回答里即算对**——见
[`../harness/extra_pipeline.py`](../harness/extra_pipeline.py) 的 `judge_tempreason`。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .preprocess import Message, Question, Sample, Session
from .sampling import stratified_sample

__all__ = ["DATA_DIR", "PIPELINE", "USER_PREFIX", "load_tempreason"]

DATA_DIR = "tempreason"

#: 本数据集没有官方 AML pipeline ⇒ 走我们自写的实现。
PIPELINE = "extra_pipeline.py"

USER_PREFIX = "tr-"

#: 只收这两档（L1 没有上下文，见模块 docstring）。
FILES = ("test_l2.json", "test_l3.json")

#: 切句：**换行也算**（维基页的标题/段落之间只有换行，不切会把标题粘到正文上，
#: + 句末标点后的空白。**不切小数点**（"8.5" 要留住）。
#: ⚠ 上游正文里确有粘在一起的（如 `Jaroslav PelikanJaroslav Jan Pelikan Jr.`，是它抓取时
#:   把标题并进正文留下的），那不是本函数的锅。
_SENTENCE_BREAK = re.compile(r"\n+\s*|(?<=[.!?])\s+")

#: **点号是缩写的一部分，不是句末**——`F.C.` / `A.F.C.` / `U.S.A.` / `J.`（首字母缩写）。
#:
#: ## 为什么必须有这一条（2026-10-04 实测）
#:
#: 事实句的宾语常常是**足球俱乐部**，而它们的名字以缩写结尾：
#:
#: ```text
#: Raúl Servín plays for Atlas F.C. from Jan, 1991 to Jan, 1992.
#: ```
#:
#: 只在 `.` 后切 ⇒ 切成 `…plays for Atlas F.C.` 与 **`from Jan, 1991 to Jan, 1992.`**，
#: 而**后半句自成一条记忆块**（块按句排、各占一个名额）。
#:
#: 后果是**这份数据集真正要考的那一步没了**：问 "May, 1991 他在哪支队"，检索带回来的
#: 是那半句没有区间的队名，模型只能拒答（`base-tr` 的 332 题里 259 道逐字拒答，
#: 其中一道就是 `Atlas F.C.`——答案本来躺在被切走的那半句里）。
#: 量化：**L2 的事实行被切碎 7,061/44,168（16.0%）、L3 5,255/38,432（13.7%）**；
#: 到题目层是 **539/5,397（10%）的 L2 与 360/4,426（8%）的 L3**，
#: gold 恰好落在被切碎的那一行上。
_ABBREVIATION_TAIL = re.compile(r"(?:\b[^\W\d_]\.){1,}$")

#: 另一种缩写：**有元音、不像首字母缩写**，只能逐个列。剩下的那 0.6% 全是这些（实测列举）。
_ABBREVIATION_WORDS = re.compile(
    r"\b(?:jr|sr|st|inc|ltd|co|corp|smt|shri|dr|mr|mrs|ms|prof|gov|sen|rep|"
    r"capt|gen|col|lt|sgt|vs|etc|no|fig|mt|ft|ave|blvd|univ|dept|est)\.$",
    re.IGNORECASE,
)

#: ⚠ **下面这条规则试过、被实测否掉了**：`1. FC Köln`（德语俱乐部的序数）也想保住，
#: 于是想在"数字 + 点 + 全大写短词"时不切——但实测页面正文里 `数字. 大写词` 有
#: **58,834 处**，其中 1,684 处下一个词是全大写且 ≤4 字符，**而 `1909. A year later`
#: 正是其中一员**（`A` 满足条件）⇒ 会把真句界合并掉。**代价远大于收益，不做。**


def _is_abbreviation(text: str) -> bool:
    """这个点号是缩写的一部分吗（⇒ **不是句末**）。"""
    return bool(_ABBREVIATION_TAIL.search(text) or _ABBREVIATION_WORDS.search(text))


def _sentences(text: str) -> list[str]:
    """切句。**缩写里的点号不算句末**（见 `_ABBREVIATION_TAIL`）。"""
    source = text or ""
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(source):
        # `(?<=[.!?])\s+` 的匹配从空白起（lookbehind 不占位）⇒ 标点就在它前面。
        if "\n" not in match.group() and _is_abbreviation(source[: match.start()]):
            continue
        parts.append(source[start : match.start()])
        start = match.end()
    parts.append(source[start:])
    return [part.strip() for part in parts if part.strip()]


def _facts_first(facts: list[str]) -> list[str]:
    """把各组 `fact_context` 切成句子并**去重保序**。

    ⚠ **去重是必须的**：同一页的几道题**共用同一份事实集**（实测：60 个多题页全部成立），
    不去重会让同一句在记忆里出现 3–4 次 ⇒ 每个重复句各占一个块、白占名额。

    ⚠ **它排在正文页之前**（`_facts_first` 这个名字就是口径）：事实句是这份数据集真正要考的东西
    （"X works for Y from A to B"），而正文页是背景。位置只影响名次，不影响配对
    ——每句都是 `role="user"`，D32 之后**每条各自成块**。
    """
    seen: set[str] = set()
    out: list[str] = []
    for fact in facts:
        for sentence in _sentences(fact):
            if sentence not in seen:
                seen.add(sentence)
                out.append(sentence)
    return out


def _answers(row: dict) -> list[str]:
    payload = row.get("text_answers")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload.replace("'", '"'))
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        values = payload.get("text")
        if isinstance(values, str):
            return [values]
        if isinstance(values, list):
            return [str(v) for v in values]
    return []


def load_tempreason(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 TempReason（L2/L3）——**一个 `Sample` = 一个唯一 context 页**。

    ⚠ 部分跑要加 `spread=True`：文件按主体排，`[:limit]` 会只取到开头那几个页。
    """
    root = Path(bench_dir) / DATA_DIR
    groups: dict[tuple[str, str], dict] = {}
    for name in FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"缺 {path}（`make fetch-data` 取回 official-extra 那一档）")
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                context = row.get("context") or ""
                if not context:
                    # L1 那两档会走到这里——**不静默**，说明它为什么进不来。
                    raise ValueError(
                        f"{name}:{index}：没有 context——TempReason 只有 L2/L3 是记忆任务"
                    )
                key = (name, context)
                entry = groups.setdefault(key, {"rows": [], "context": context, "facts": []})
                entry["rows"].append((index, row))
                facts = row.get("fact_context")
                if isinstance(facts, str) and facts.strip():
                    entry["facts"].append(facts)
    keys = sorted(groups)  # 稳定序：文件序 + 页在原文件里的出现序
    keys.sort(key=lambda k: (FILES.index(k[0]), min(i for i, _ in groups[k]["rows"])))
    if limit is not None:
        picked = stratified_sample(keys, limit, key=lambda k: k[0]) if spread else keys[:limit]
    else:
        picked = keys

    samples: list[Sample] = []
    for key in picked:
        name, _context = key
        entry = groups[key]
        sentences = _sentences(entry["context"])
        if not sentences:
            raise ValueError(f"{name}：context 切不出句子")
        # ⚠ **`fact_context` 必须进记忆**（2026-10-03 修，实测依据见模块 docstring 的
        # "为什么 fact_context 是记忆"一节）。本来的做法是**只灌正文页**，后果是
        # **239/332 题的 gold 从来不在记忆里**、**259/310 的错题是逐字拒答**
        # ⇒ overall 0.066 里绝大多数量的是"我们没有那份事实"。
        sentences = _facts_first(entry["facts"]) + sentences
        if not sentences:
            raise ValueError(f"{name}：context 与 fact_context 都切不出句子")
        questions = []
        for index, row in entry["rows"]:
            answers = _answers(row)
            if not answers:
                raise ValueError(f"{name}:{index}：text_answers 里没有答案")
            questions.append(
                Question(
                    qid=f"{name[:-5]}-{index}",
                    question=str(row["question"]),
                    gold=answers,
                    category=name[:-5],
                )
            )
        samples.append(
            Sample(
                user_id=f"{USER_PREFIX}{name[:-5]}-{min(i for i, _ in entry['rows']):05d}",
                dataset="tempreason",
                # ⚠ **一页一个 session（一句一条消息）**：D29 之后没有非 user 跟随的
                #   连续 user 每条各自独立成块 ⇒ 粒度保得住，而 Add 次数少一个数量级。
                sessions=(
                    Session(
                        session_id=f"{name[:-5]}-{min(i for i, _ in entry['rows']):05d}",
                        messages=tuple(
                            Message(role="user", content=sentence) for sentence in sentences
                        ),
                    ),
                ),
                questions=tuple(questions),
                speaker_names=("user", "assistant"),
            )
        )
    return samples
