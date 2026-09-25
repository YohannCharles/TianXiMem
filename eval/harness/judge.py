"""包住 AML pipeline 的裁判——**subprocess 调用，不 import 它们**。

## 为什么只能 subprocess

`pipeline_locomo-refined.py` 的**文件名里带连字符**，无法 `import`；它和
`pipeline_longmemeval-s.py` 都是带 argparse 子命令的独立脚本。
harness 要么 subprocess 调它、要么**复制它的逻辑**——后者会让
"契约以 pipeline 代码为准"这条失去意义（复制的当天就对不上了）。

## 注入格式：**键名是 `speaker_1_memories`**

§17.3 的 **V3** 记着"用什么键名注入没有明文"。**读 pipeline 源码即可回答**：

```python
fallback_memories = item.get("retrieved_context", item.get("memories", ""))
values = {
    "speaker_1_name": item.get("speaker_1_name", "speaker 1"),
    "speaker_1_memories": item.get("speaker_1_memories", fallback_memories),
    "speaker_2_name": item.get("speaker_2_name", "speaker 2"),
    "speaker_2_memories": item.get("speaker_2_memories", ""),
    ...
}
```

⇒ **主字段是 `speaker_1_memories`**（回退链 `speaker_1_memories → retrieved_context →
memories`）；**`speaker_2_memories` 没有回退，缺即空串**——这条不对称是
[`../CLAUDE.md`](../CLAUDE.md) 里"退化不对称"的出处。

> ⚠ **仍然未知的是"怎么分"**，不是"用哪个键"：真实 AML 侧按什么把我们的返回列表
> 切成两个 speaker 块，**归档里看不到**（§17.1 的 S1 是同一类问题的记忆版）。
> ⇒ 本模块把**全部命中的 `content` 拼成一个字符串放进 `speaker_1_memories`**，
> `speaker_2_memories` 留空。**这是一个显式的代理假设**，不是事实——
> `memory_field` 参数就是给它留的出口。**别把它当成已验证的契约。**

## 记住这条不对称（会改变结果）

| 数据集 | `evaluate` 步遇到 ID 集合不一致时 |
| --- | --- |
| `pipeline_locomo-refined.py` / `pipeline_longmemeval-s.py` | **`raise SystemExit`——硬失败** |
| `clb_pipeline.py` | `answers.get(ident, {})`——**缺答案一律判 0** |

⇒ **同一个 harness 缺陷（漏跑几题）在两份数据集上表现完全不同**：
一份崩掉，一份静默掉分。**别把后者当成"模型变差了"**（`../CLAUDE.md`）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from eval.datasets import Sample
from eval.datasets.locomo import PIPELINE as LOCOMO_PIPELINE
from eval.datasets.longmemeval import PIPELINE as LME_PIPELINE

from .annotate import MARKS, annotate
from .driver import SearchHit

__all__ = [
    "DATE_HEADER",
    "DATE_MODES",
    "DATE_PREFIX",
    "MARKS",
    "JudgeResult",
    "MEMORY_FIELD",
    "build_input_items",
    "render_memories",
    "run_judge",
    "pipeline_for",
]

#: 命中列表的注入位置（见模块 docstring 的 V3 讨论）。
MEMORY_FIELD = "speaker_1_memories"

#: 数据集 → 归档 pipeline 文件名。**数据集专属文件名的知识止步于 `eval/datasets/`**（D16）
#: ——这里的映射就是那条边界在 harness 侧的接口。
_PIPELINES = {
    "locomo-refined": LOCOMO_PIPELINE,
    "longmemeval-s": LME_PIPELINE,
}


def pipeline_for(bench_dir: Path, dataset: str) -> Path:
    """取归档里的 pipeline 脚本路径。"""
    try:
        name = _PIPELINES[dataset]
    except KeyError:
        raise ValueError(f"未知数据集 {dataset!r}——没有对应的归档 pipeline") from None
    return Path(bench_dir) / name


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """一题的判分结果——`judge_response` 原样留着，**判分失败时它是唯一的线索**。"""

    qid: str
    is_correct: bool
    label: str
    judge_response: str
    generated_answer: str


#: 日期前缀的形状——**与 `src/tianxi_am/common/render.py` 的 `DATE_PREFIX` 必须逐字相同**
#: （那边是 `content` 里加日期时用的同一段）。
#: ⚠ 这里是**故意重复的一份字面量**：harness 不许 import `src/`（本目录最硬的一条边界，
#: [`../../tests/test_harness.py`](../../tests/test_harness.py) 用 AST 钉着），
#: 而"两处格式必须一致"由 [`../../tests/test_experiments.py`](../../tests/test_experiments.py)
#: 的等价断言保证——**重复的是常量，不是逻辑**。
#:
#: ⚠ 同样的形状也是 AML 自己的约定：[`../../docs/contract.md`](../../docs/contract.md) §3
#: 记着 CL-Bench 那条路径用 `- [timestamp] text` 渲染我们返回的 `created_at`。
DATE_PREFIX: str = "[{date}] "

#: 注入里怎么带日期——**三档**（`--memory-date` 的取值域）。
#:
#: * `none`：完全不带（**基线**，也是"AML 侧只取 content"那条 S1 假设）
#: * `per_item`：每条记忆（= 一个段）前面加 `[YYYY-MM-DD] `（2026-09-25 第一版）
#: * `per_pair`：**每一对**前面都加 `[YYYY-MM-DD] `——同一个日期，但离该句更近。
#:   动机（2026-09-25）：temporal 的失败里有这么一类——模型**不缺信息**（`[2023-07-15]`
#:   就在段首、"Last Friday" 就在下面），但不把锚点接到自己那句上（见 `eval/reports/ledger.md`）
#: * `per_pair_wd`：`per_pair` + **星期**（`[2023-10-22 Sun] `）。⛔ **实测更差，别再跑**：
#:   动机是"temporal 错的 70 题里 29 道的 gold 是某星期几之前/之后"，但那天**一道都没救回来**，
#:   整体还从 0.633 掉到 0.601、三段全降
#:   （见 [`../../docs/decisions.md`](../../docs/decisions.md) **D21**）。
#:   留在模式表里只为"试过、有据可查"。
#: * `header`：在上面的基础上，**记忆块顶部再加一行说明**，明说每块前面的日期是它的会话日期
#: * `annotate`：**不动结构，只把句子里的相对表达就地注解成绝对日期**
#:   （`last Tues (July 18, 2023)`）——动机见 [`annotate.py`](./annotate.py) 的模块 docstring
#:   （2026-09-25：`t1-dated` 的 temporal 79 道里 **27 道卡在"要换算"这一步**）。
#:   ⚠ **它不加前缀**：索引侧 `packaging.inject_abs_time=true` 已经给每一对加了日期，
#:   这里再前缀一次会变成两个日期。
#:   ⚠ **锚点用的是段级 `created_at`，生产侧用的是"每一对自己的 `event_time`"**——
#:   两者只在**会话跨天**时才不同。LoCoMo 的 3 段 70 个 session **一个跨天的都没有**
#:   （2026-09-26 核过）⇒ 这份数据上两者等价。**换数据集/换语料时要重核这一条**，
#:   否则实验量的是 A、上线跑的是 B，而两边都不报错。
#:
#: 为什么要有 `header`：`per_item` 实测**没能改变模型行为**（27/35 仍答相对，与基线 26/35 几乎相同）
#: ——"看得见日期"≠"用得上日期"。所以第二版把语义**写明**，看是提示不够清楚还是模型做不到。
DATE_MODES: tuple[str, ...] = ("none", "per_item", "per_pair", "per_pair_wd", "header", "annotate")

#: `header` 模式的那行说明。`{dates}` 是**去重后的会话日期**，按出现顺序。
DATE_HEADER: str = (
    "[Note: the memories below come from conversation sessions dated {dates}. "
    "Each memory block is prefixed with the date of its own session.]"
)


def _weekday(date: str) -> str:
    """`2023-10-22` → `Sun`。解析不了就返回空串（**不抛**：注入形状不该让整轮跑挂掉）。"""
    from datetime import date as _date

    try:
        return _date.fromisoformat(date).strftime("%a")
    except ValueError:  # pragma: no cover —— created_at 由我们自己生成，形状是固定的
        return ""


def _date_header(hits: list[SearchHit]) -> str:
    """`header` 模式的首行：把这些记忆覆盖到的会话日期**列出来**（去重、按出现顺序）。"""
    dates: list[str] = []
    for hit in hits:
        if hit.created_at and hit.created_at not in dates:
            dates.append(hit.created_at)
    return DATE_HEADER.format(dates=", ".join(dates))


def _annotated(hit: SearchHit, mark: str) -> str:
    """把一条命中的正文按**它自己的会话日期**注解。**解析不了就原样返回**（不抛）。

    注入形状不该让整轮跑挂掉——与 `_weekday()` 同一条纪律。
    """
    if not hit.created_at:
        return hit.content
    try:
        anchor = date.fromisoformat(hit.created_at)
    except ValueError:  # pragma: no cover —— created_at 由我们自己生成，形状是固定的
        return hit.content
    return annotate(hit.content, anchor, mark=mark)


def render_memories(
    hits: list[SearchHit], *, date_mode: str = "none", annotate_mark: str = "paren"
) -> str:
    """把命中拼成注入文本。

    **只取 `content`**——这是 S1 那条代理假设的落点（"AML 从返回项里取哪个字段"）。
    用 `"\\n"` 拼接而不是别的分隔符：归档的 `memory_text()` 对列表正是 `"\\n".join(...)`，
    所以"我们自己拼"与"把列表交给它拼"**逐字相同**。

    `include_date=True` ⇒ 每条前面加 `[created_at] `；**默认 `False`**（理由见下）。

    ## 为什么会有这个开关（2026-09-25）

    LoCoMo 的时间题 gold 几乎全是**绝对日期**（`2 July 2023`），而对话正文里说的是
    **相对**（"yesterday"）；裁判的 TIME 块要求**单位与 gold 一致、且不许相对↔绝对互转**。
    答案 prompt 第 7 条允许换算，前提是"**记忆的时间戳让它清晰**"——
    ⇒ 时间戳不可见，那条规则就无法执行，模型只能照抄 "Yesterday"，**必判错**
    （实测 `eval/reports/ledger.md`：gold 含绝对日期的 35 道里 **26 道答成相对、全部判错**）。

    ⚠ **`created_at` 服务本来就返回了**（`data[]` 四字段之一），是**本函数原先把它丢了**。
    但"AML 真实侧往 `speaker_1_memories` 里填什么"仍是 **S1 未知**：
    开这个开关是**换一条代理假设**，不是修一个 bug——所以默认关，
    开启时要在 run record 里记明（runner 的 `--memory-date` + `--switches`）。
    """
    if date_mode == "none":
        return "\n".join(hit.content for hit in hits)

    if date_mode == "annotate":
        # **不加前缀**——索引侧（`packaging.inject_abs_time`）已经给每一对加了日期。
        # 这一档只做"在相对表达后面跟一个括号注"，其余一字不动。
        return "\n".join(_annotated(hit, annotate_mark) for hit in hits)

    if date_mode in ("per_pair", "per_pair_wd"):
        with_weekday = date_mode == "per_pair_wd"
        # 段里的正文形如 `Q: …\nA: …\nQ: …`（`common/render.py` 的模板）⇒ 按行首的 `Q: `
        # 切成"对"，再给**每一对**前面加同一个日期。日期本来就只有段级粒度
        # （同 session 共享一个日期），所以信息量与 `per_item` 相同，变的是**距离**。
        numbered: list[str] = []
        for hit in hits:
            if not hit.created_at:
                numbered.append(hit.content)
                continue
            stamp = hit.created_at
            if with_weekday:
                stamp = f"{stamp} {_weekday(hit.created_at)}"
            prefix = DATE_PREFIX.format(date=stamp)
            pairs = hit.content.split("\nQ: ")
            numbered.append(
                "\n".join(
                    f"{prefix}{pair}" if index == 0 else f"{prefix}Q: {pair}"
                    for index, pair in enumerate(pairs)
                )
            )
        return "\n".join(numbered)

    lines = [
        f"{DATE_PREFIX.format(date=hit.created_at)}{hit.content}" if hit.created_at else hit.content
        for hit in hits
    ]
    if date_mode == "header":
        lines = [_date_header(hits), *lines]
    return "\n".join(lines)


def build_input_items(
    sample: Sample,
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    memory_field: str = MEMORY_FIELD,
    date_mode: str = "none",
    annotate_mark: str = "paren",
) -> list[dict]:
    """构造 pipeline `answer` 步读的 JSONL 项。

    字段名**以 pipeline 代码为准、不照 readme**（`generated_answer` 才是 stage 间的规范
    字段；readme 写的 `predicted_answer` / `hypothesis` **没有 pipeline 读**）。

    `gold_answer` 直接放 `Question.gold` 的原始形态（LoCoMo 是 `list[str]`）——
    归档的 `gold_answer()` 走 `memory_text()`，**列表由它自己用 `\\n` 拼**，
    在这里预先拼会改变那条链路的行为。
    """
    items = []
    for question in sample.questions:
        hits = hits_by_qid.get(question.qid, [])
        item: dict = {
            "id": question.qid,
            "question": question.question,
            "gold_answer": question.gold,
            "speaker_1_name": sample.speaker_names[0],
            "speaker_2_name": sample.speaker_names[1],
            # `speaker_2_memories` **没有回退**，缺即空串——这里就照实留空，
            # 让不对称原样暴露在 prompt 里，而不是我们自己造一个"看起来更合理"的形状。
            "speaker_2_memories": "",
        }
        item[memory_field] = render_memories(hits, date_mode=date_mode, annotate_mark=annotate_mark)
        items.append(item)
    return items


def _subprocess_env() -> dict[str, str]:
    """注入 `PYTHONPATH` 让 `from api_config import (...)` 找到本目录那份。

    `sys.path.insert(0, <不存在路径>)` 不会中断 import——它只是塞进一个没有该模块的
    条目，import 继续往后找到这里。归档因此**保持只读**。
    """
    env = dict(os.environ)
    harness_dir = str(Path(__file__).resolve().parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = harness_dir + (os.pathsep + existing if existing else "")
    return env


def _run(pipeline: Path, argv: list[str], *, timeout: float | None) -> None:
    completed = subprocess.run(
        [sys.executable, str(pipeline), *argv],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{pipeline.name} {' '.join(argv[:1])} 失败（exit {completed.returncode}）\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )


def run_judge(
    pipeline: Path,
    items: list[dict],
    out_dir: Path,
    *,
    max_tokens: int = 256,
    timeout: float | None = None,
) -> list[JudgeResult]:
    """跑 `answer` → `evaluate` 两步，返回逐题结果。

    ⚠ **`out_dir` 必须由调用方建**——`pipeline_locomo-refined.py` **不会
    `mkdir(parents=True)`**（`clb_pipeline.py` 会）。文件不存在时它以 `"a"` 模式打开，
    所以目录缺了就是 `FileNotFoundError`。

    ⚠ **超时默认 `None`**：Full run 要连续跑 0.5–2 天，一个"看着合理"的超时会把
    **慢**伪装成**失败**（§15）。要设就设得比 `Full` 的时长还宽。

    **两步之间的不对称**：`answer` 以**追加**模式打开并按 `id` 跳过已完成的题
    （所以中断后重跑是安全的）；`evaluate` 以 `"w"` **覆盖**打开。
    而两个 pipeline 的 `evaluate` 都在 ID 集合不一致时 **`raise SystemExit`**——
    **漏跑几题是硬失败，不是静默掉分**。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"
    answers_path = out_dir / "answers.jsonl"
    labels_path = out_dir / "labels.jsonl"

    with input_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    # `--model` / `--base-url` / `--api-key-env` 都**是死的**：这两个子命令在协程开头
    # 就用 `api_config` 的常量重写 `args.*`。所以这里一个都不传（传了只会让人以为生效了）。
    _run(
        pipeline,
        [
            "answer",
            "--input",
            str(input_path),
            "--output",
            str(answers_path),
            "--max-tokens",
            str(max_tokens),
        ],
        timeout=timeout,
    )
    _run(
        pipeline,
        [
            "evaluate",
            "--input",
            str(input_path),
            "--answers",
            str(answers_path),
            "--output",
            str(labels_path),
            "--max-tokens",
            str(max_tokens),
        ],
        timeout=timeout,
    )

    generated = {
        row["id"]: row["generated_answer"]
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    results = []
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(
                JudgeResult(
                    qid=row["id"],
                    is_correct=row["is_correct"],
                    label=row["label"],
                    judge_response=row["judge_response"],
                    generated_answer=generated.get(row["id"], ""),
                )
            )
    return results
