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
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from eval.datasets import Sample
from eval.datasets.clbench import PIPELINE as CLB_PIPELINE
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
    "clbench": CLB_PIPELINE,
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


#: 日期前缀的形状——**与 `src/tianximem/common/render.py` 的 `DATE_PREFIX` 必须逐字相同**
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

    `date_mode` 决定日期怎么带——**逐档的口径都写在 `DATE_MODES` 上**（含每档的实测结论）；
    默认 `"none"`（理由见下）。

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
    if sample.dataset == "clbench":
        return _build_clbench_items(sample, hits_by_qid)

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
        text = render_memories(hits, date_mode=date_mode, annotate_mark=annotate_mark)
        # ★ 平台的答案阶段是"按返回顺序取 117,760 token 前缀"——**harness 要自己模拟**，
        #   因为基线未必守预算（实测 ReFind 有一题返回 89.8 万字符）。详见常量的注释。
        text, cut = truncate_to_platform_prefix(text)
        if cut:
            _warn_truncated(question.qid)
        item[memory_field] = text
        items.append(item)
    return items


def _warn_truncated(qid: str) -> None:
    """注入被平台前缀截断——**两条注入路径共用这一行字**。"""
    print(f"  ⚠ {qid}：注入被截到平台前缀（{PLATFORM_TOKEN_PREFIX:,} token）", flush=True)


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


def _run(
    pipeline: Path, argv: list[str], *, timeout: float | None, attempts: int = 3
) -> None:
    """跑一个 pipeline 子命令，**瞬时故障自动重试**。

    ## 为什么必须重试（2026-09-26，一次真实事故）

    LongMemEval 那轮跑到第 47 题时，**判分侧一次 TLS 握手失败**
    （`httpcore ... start_tls` → `httpx.ConnectError`）就把**两小时的跑批整个打死**，
    而**没有任何东西会重试**。那一轮实际被同类故障打断过 11 次。

    ⚠ **重试是安全的**，因为两个子命令都幂等：`answer` 以**追加**模式打开、按 `id`
    跳过已完成的题；`evaluate` 以 `"w"` **覆盖**打开。⇒ 重跑一次最坏是白跑一遍。
    ⇒ **不重试的代价（整轮作废）远大于重试的代价（几十秒）**。

    ⚠ **重试次数要有界**：真 bug（形状不符、数据坏了）重试 3 次还是失败 ⇒ 照样抛，
    错误信息原样带出去——**不把"一直失败"伪装成"在重试"**。
    """
    last: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            [sys.executable, str(pipeline), *argv],
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode == 0:
            if attempt > 1:
                print(
                    f"  （{pipeline.name} {' '.join(argv[:1])}：第 {attempt} 次尝试成功）",
                    flush=True,
                )
            return
        last = RuntimeError(
            f"{pipeline.name} {' '.join(argv[:1])} 失败（exit {completed.returncode}）\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )
        if attempt < attempts:
            wait = 5 * attempt
            print(
                f"  ⚠ {pipeline.name} {' '.join(argv[:1])} "
                f"第 {attempt}/{attempts} 次失败（exit {completed.returncode}），"
                f"{wait}s 后重试（该步骤幂等）",
                flush=True,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc] —— attempts ≥ 1，循环至少设过一次


def _sanitize_jsonl(path: Path) -> int:
    """把 `answers.jsonl` 修成**归档读得下去**的样子；返回改动了几行。

    **两件事一起做**（都只针对这个文件，且都保持语义不变）：

    | 症状 | 处置 |
    | --- | --- |
    | 末尾有**半行 JSON**（上次被强杀留下的） | **丢掉**——对应的题会被重新生成 |
    | 某行里含 `U+2028` 之类的字符（**归档自己写的**，我们只读不写） | **就地转义**成 `\u2028` |

    ⚠ 第二条是**我们控制不了写入侧**时的唯一办法：`answers.jsonl` 由归档的 `answer()`
    以 `json.dumps(..., ensure_ascii=False)` 写、又由它自己的 `rows()`（`splitlines()`）
    读 —— 万一**模型生成的答案**里带 `U+2028`，**归档会在下一次续跑时崩在自己写的文件上**。
    我们改不了归档，所以只能在它读之前把文件修好。转义是安全的：`json.loads` 会解回
    同一个字符，**语义一字不变**。

    ## 为什么需要它（2026-09-26，一次真实事故，比网络抖动更隐蔽）

    那次事故的真实链条是这样的：

    1. 判分侧**一次 TLS 抖动**把 runner 打死；
    2. 死的时候 pipeline 正以**追加**模式写 `answers.jsonl`（逐行 `write` + `flush`，
       见 `pipeline_longmemeval-s.py` 的 `answer()`）⇒ 留下**半行 JSON**；
    3. 而同一个 `answer()` 开头会 `rows(output)` **读它来跳过已完成的题**
       （`done = {item["id"] for item in rows(output)}`）⇒ `JSONDecodeError`；
    4. ⇒ **之后每次续跑都在同一个样本上确定性失败**——表现为"卡在同一题"，
       而**看起来像网络问题**（第一次确实是），于是排查方向全错。

    ⚠ **只在末尾删**：中途的坏行说明别的问题（磁盘、并发写），不该被静默吞掉。
    ⚠ 删掉的只有**半行**——它对应的那道题会被重新生成（`answer` 是追加 + 跳过已完成）。
    """
    if not path.exists():
        return 0
    # ⚠ **按 `"\n"` 切，不用 `splitlines()`**：这个文件是归档按 `+ "\n"` 写出来的，
    #   而 `splitlines()` 会把正文里的 `U+2028` 也当换行——那会把一条**完整**的记录
    #   看成两条坏的，于是被"修"掉（该记录会被重新生成，白花一次调用）。
    raw = path.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()  # 文件以 `"\n"` 结尾 ⇒ `split` 会多出一个空元素，重写时要还回去

    # ① 末尾的半行：从后往前丢（只丢**解析不了**的）
    keep = len(raw)
    while keep > 0 and raw[keep - 1].strip():
        try:
            json.loads(raw[keep - 1])
            break
        except json.JSONDecodeError:
            keep -= 1
    dropped = len(raw) - keep

    # ② 会被 `splitlines()` 劈开的字符：**就地转义**（语义不变）
    escaped = 0
    fixed: list[str] = []
    for line in raw[:keep]:
        if line.strip() and any(char in line for char in _LINE_BREAKS):
            try:
                line = _jsonl_line(json.loads(line)).rstrip("\n")
                escaped += 1
            except json.JSONDecodeError:  # pragma: no cover —— 上面刚验过能解析
                pass
        fixed.append(line)

    if dropped or escaped:
        path.write_text("".join(f"{line}\n" for line in fixed), encoding="utf-8")
        print(
            f"  ⚠ 修复 `{path.name}`：丢掉末尾 {dropped} 行不完整的 JSON、"
            f"转义 {escaped} 行里的行分隔符字符",
            flush=True,
        )
    return dropped + escaped


#: `str.splitlines()` **会断行**、而 `json.dumps(ensure_ascii=False)` **不转义**的字符。
#:
#: ⚠ `\n` / `\r` / `\x0b` / `\x0c` / `\x1c-\x1e` 都由 `json.dumps` 自己转义掉了；
#: 只有 **`\x85` / `\u2028` / `\u2029`** 是它不碰、而 `splitlines()` 照断的——
#: 这三个才是真正会咬人的（列全是为了"下次有人加字符时不假思索"）。
_LINE_BREAKS: Final[tuple[str, ...]] = ("\x85", "\u2028", "\u2029")


def _jsonl_line(item: dict) -> str:
    """把一项写成**一行** JSONL——**转义掉 `splitlines()` 会断行的字符**。

    ## 为什么（2026-09-26，LongMemEval 那轮的真实阻塞）

    归档 pipeline 的 `rows()` 是这么读的：

    ```python
    [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ```

    **`splitlines()` 断的比 `\n` 多得多**（`\x85`、`\u2028`、`\u2029`、`\v`、`\f`、`\x1c-\x1e`），
    而 `json.dumps(..., ensure_ascii=False)` **不转义**其中三个 ⇒ 只要**正文里出现
    `U+2028`**，那一行就会被劈成两半，前半段的字符串没有闭合
    ⇒ `JSONDecodeError: Unterminated string`。

    ⇒ 现象是"**卡在同一道题**"（实测 47 个席位里只有 1 个含 `U+2028` ⇒
    **数据相关**，却看起来像网络问题，排查方向整个跑偏）。**写入侧转义是唯一的修法**
    ——读的那侧是归档代码，不许改。
    """
    line = json.dumps(item, ensure_ascii=False)
    for char in _LINE_BREAKS:
        if char in line:
            # 换成 JSON 转义序列：`json.loads` 会解回同一个字符，语义一字不变。
            line = line.replace(char, f"\\u{ord(char):04x}")
    return line + "\n"


#: 平台在**答案阶段**对返回列表做的截断：**按返回顺序取 117,760 token 前缀**（§2.2 / §6.4）。
#:
#: ⚠ **harness 必须自己模拟这一步**，不能假设"服务返回的证据自己会守预算"：
#: 我们自己的服务确实守（`packaging` 的双预算），但**基线不一定**——
#: 2026-09-26 实测：ReFind 在 LongMemEval 上有一题返回了 **897,838 字符**（≈22–30 万 token），
#: 判分提示直接爆 128k ⇒ 网关 400 ⇒ **整轮被打死**。而**平台会替它截断**，
#: 所以正确的模拟是"截前缀"，不是"让整轮炸掉"。
#:
#: ⚠ 分词器用 **`o200k_base`**（= 平台答案模型 `gpt-4o-mini` 的分词器），与
#: `budget.tokenizer` 同一口径——**不要用字符数近似**（§6.4）。
PLATFORM_TOKEN_PREFIX: Final[int] = 117_760


def _encoder():
    import tiktoken  # 只在真要用时才 import（harness 的纯逻辑用例不必装它）

    return tiktoken.get_encoding("o200k_base")


def truncate_to_platform_prefix(text: str) -> tuple[str, bool]:
    """按平台规则截断；返回 `(文本, 是否截过)`。"""
    encoder = _encoder()
    ids = encoder.encode(text)
    if len(ids) <= PLATFORM_TOKEN_PREFIX:
        return text, False
    return encoder.decode(ids[:PLATFORM_TOKEN_PREFIX]), True


def _build_clbench_items(sample: Sample, hits_by_qid: dict[str, list[SearchHit]]) -> list[dict]:
    """CL-Bench 的项——**形状与那两份完全不同**
    （见 [`../datasets/clbench.py`](../datasets/clbench.py)）。

    | 字段 | 哪来的 / 为什么 |
    | --- | --- |
    | `idx` | `clb_pipeline.py` 的 `row_id()` **先认 `idx`**（`answer` 靠它跳过已完成） |
    | `system_prompt` | 记录里第一条 `system` 消息——**raw 文件里没有这个顶层键**，不补就塌成空串 |
    | `question` | 加载器切出来的任务文本（末条 user 的尾部窗口） |
    | `rubrics` | 判分标准；`official_rubrics()` 认 `item["rubrics"]` |
    | `retrieval.selected` | **每项 `created_at` + `text`**——⚠ 读的是 **`text`**，
而且**缺 `text` 的项会被直接跳过**（`docs/contract.md` §5） |

    ⚠ 记忆块**按平台的 117,760 token 前缀截断**，且**按项截**（不切半个段）——
    与 `packaging` 的"段是原子单位"同一条理由。
    """
    system_prompt = next(
        (
            message.content
            for session in sample.sessions
            for message in session.messages
            if message.role.lower() == "system"
        ),
        "",
    )
    items: list[dict] = []
    for question in sample.questions:
        hits, cut = _prefix_within_budget(hits_by_qid.get(question.qid, []))
        if cut:
            _warn_truncated(question.qid)
        items.append(
            {
                "idx": question.qid,
                "question": question.question,
                "system_prompt": system_prompt,
                "rubrics": question.gold,
                "retrieval": {
                    "selected": [
                        {"created_at": hit.created_at, "text": hit.content} for hit in hits
                    ]
                },
            }
        )
    return items


def _prefix_within_budget(
    hits: list[SearchHit], budget: int = PLATFORM_TOKEN_PREFIX
) -> tuple[list[SearchHit], bool]:
    """只保留**放得进预算的前缀项**；返回 `(保留的项, 是否截过)`。

    ⚠ **整项保留**（不切半个）——与 `packaging` 的"段是原子单位"同一条理由：
    切一半的段会让模型读到缺一环的上下文，而 `content` 里看不出来。
    """
    encoder = _encoder()
    kept: list[SearchHit] = []
    used = 0
    for hit in hits:
        cost = len(encoder.encode(hit.content))
        if used + cost > budget:
            return kept, True
        kept.append(hit)
        used += cost
    return kept, False


def _read_clbench_labels(answers_path: Path, labels_path: Path) -> list[JudgeResult]:
    """读 CL-Bench 的两份产物——**字段名与那两份完全不同**。

    | 文件 | 谁写的 / 读什么 |
    | --- | --- |
    | `answers.jsonl` | `answer` 步写 **`model_output`**（不是 `generated_answer`），
行键是 **`idx`** |
    | `labels.jsonl` | `evaluate` 步写 `rubric_clbench_score` / `_rationale` /
`_requirement_status` / `_requirement_ratio` |

    ⚠ **它是严格全有全无**：`score` 只有 0 或 1 ⇒ 映射成 `is_correct = score >= 1.0`，
    并把 `requirement_ratio`（满足了几成要求）一并留进 `judge_response`——**分档信息在
    二值化之后会丢**，而"差一点就满分"与"完全跑偏"是两回事。
    """
    generated = {
        str(row.get("idx", row.get("id", ""))): str(row.get("model_output") or "")
        for row in (
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
    }
    results: list[JudgeResult] = []
    for line in labels_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        ident = str(row.get("idx", row.get("id", "")))
        score = float(row.get("rubric_clbench_score") or 0.0)
        ratio = float(row.get("rubric_clbench_requirement_ratio") or 0.0)
        results.append(
            JudgeResult(
                qid=ident,
                is_correct=score >= 1.0,
                label="CORRECT" if score >= 1.0 else "WRONG",
                judge_response=(
                    f"[rubric score={score:.0f} ratio={ratio:.2f}] "
                    f"{str(row.get('rubric_clbench_rationale') or '')[:1500]}"
                ),
                generated_answer=generated.get(ident, ""),
            )
        )
    return results


def run_judge(
    pipeline: Path,
    items: list[dict],
    out_dir: Path,
    *,
    dataset: str = "locomo-refined",
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
            handle.write(_jsonl_line(item))

    # ★ 续跑前先修掉上次被强杀留下的半行——否则 `answer` 步读它就 `JSONDecodeError`，
    #   而表现是"**卡在同一道题**"（详见 `_sanitize_jsonl` 的事故记录）。
    _sanitize_jsonl(answers_path)

    # `--model` / `--base-url` / `--api-key-env` 都**是死的**：这两个子命令在协程开头
    # 就用 `api_config` 的常量重写 `args.*`。所以这里一个都不传（传了只会让人以为生效了）。
    # ⚠ **`clb_pipeline.py` 的两个子命令都不收 `--max-tokens`**（它的 `argparse` 里没有
    #   这个选项，传了会直接 `unrecognized arguments` 退出）——**按 pipeline 代码为准**。
    extra = [] if dataset == "clbench" else ["--max-tokens", str(max_tokens)]
    _run(
        pipeline,
        ["answer", "--input", str(input_path), "--output", str(answers_path), *extra],
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
            *extra,
        ],
        timeout=timeout,
    )

    if dataset == "clbench":
        return _read_clbench_labels(answers_path, labels_path)

    generated = {
        row["id"]: row["generated_answer"]
        for row in (
            # ⚠ 同样按 `"\n"` 切：答案文本里出现 `U+2028` 时，`splitlines()`
            #   会把这一条劈开 ⇒ 答案对不上号（而**不会报错**）。
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").split("\n")
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
