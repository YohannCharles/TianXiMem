"""把一次检索结果渲染成**喂给裁判的输入项**（裁判输入侧的唯一样子）。

本模块是 `judge.py` 拆出来的第二半（2026-10-15）：那里只剩"跑 pipeline 子进程 + 编排"，
"一次检索结果长什么样、按什么顺序注入、超了平台前缀怎么截"这一层在这里。

## 它同时拥有"数据集 → pipeline"那张表

`_PIPELINES` / `_LOCAL_PIPELINES` / `_ADAPTER_PIPELINES` 与 `pipeline_for` 住在这里，
因为**谁给这个数据集出题、按哪种形状出题**是同一件事的两面：`build_input_items`
要按数据集分派到不同家族的构造器（`EXTRA_DATASETS` 就在它下面）。`judge.py` 照旧
把它们**重新导出**，`from eval.harness.judge import pipeline_for` 全线不变。

⚠ **"打 HTTP、不 import `src/`"这条边界在本目录的 [`CLAUDE.md`](./CLAUDE.md)，本文件不重复。**
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

from eval.datasets import Sample
from eval.datasets.beam import PIPELINE as BEAM_PIPELINE
from eval.datasets.clbench import PIPELINE as CLB_PIPELINE
from eval.datasets.feverous import ANSWER_CONTRACT as FEVEROUS_ANSWER_CONTRACT
from eval.datasets.hybridqa import ANSWER_CONTRACT as HYBRIDQA_ANSWER_CONTRACT
from eval.datasets.locomo import PIPELINE as LOCOMO_PIPELINE
from eval.datasets.longmemeval import PIPELINE as LME_PIPELINE
from eval.datasets.musique import ANSWER_CONTRACT as MUSIQUE_ANSWER_CONTRACT

from .annotate import annotate
from .driver import SearchHit

#: 命中列表的注入位置（见模块 docstring 的 V3 讨论）。
MEMORY_FIELD = "speaker_1_memories"


#: 数据集 → 归档 pipeline 文件名。**数据集专属文件名的知识止步于 `eval/datasets/`**（D16）
#: ——这里的映射就是那条边界在 harness 侧的接口。
_PIPELINES = {
    "locomo-refined": LOCOMO_PIPELINE,
    "longmemeval-s": LME_PIPELINE,
    "clbench": CLB_PIPELINE,
    "beam": BEAM_PIPELINE,
}


#: **我们自己写的** pipeline——官方没有发布这三份的（见
#: [`extra_pipeline.py`](./extra_pipeline.py) 的 docstring：它不是契约，分数只在仓内比）。
_LOCAL_PIPELINES = {
    "mquake-remastered": "extra_pipeline.py",
    "memtrapbench": "extra_pipeline.py",
    "corporatebench": "extra_pipeline.py",
    "medmemorybench": "extra_pipeline.py",
    "tempreason": "extra_pipeline.py",
    "musique": "extra_pipeline.py",
    "hybridqa": "corpusqa_pipeline.py",
    "feverous": "corpusqa_pipeline.py",
}


#: **适配器**型 pipeline：脚本住在仓库里，但**判分逻辑来自归档那份官方实现**
#: （它的 `evaluate` 形状与通用两子命令不同，见
#: [`personamem_pipeline.py`](./personamem_pipeline.py)）。
#: ⚠ 它们不在 `EXTRA_DATASETS` 里；PersonaMem 的 `retrieved_context` 经适配器
#: 转为官方认的 `chat_history`，官方采集则逐题分派。
_ADAPTER_PIPELINES = {
    "personamem-v2": "personamem_pipeline.py",
    "halumem": "official_capture_pipeline.py",
    #: 官方采集的重放（`eval/experiments/replay_official.py`）——它一个 pipeline 里
    #: **按题分派**到 8 个数据集的 answer/judge，所以既不是归档那份、也不是
    #: `extra_pipeline.py` 的五选一（见 `official_capture_pipeline.py` 的 docstring）。
    "official-capture": "official_capture_pipeline.py",
}


#: 走 `_LOCAL_PIPELINES` 的数据集——`build_input_items` 也按它分派。
EXTRA_DATASETS = frozenset(_LOCAL_PIPELINES)


def pipeline_for(bench_dir: Path, dataset: str) -> Path:
    """取 pipeline 脚本路径：**官方的在归档里、我们写的（含适配器）在本目录**。"""
    if dataset in _LOCAL_PIPELINES:
        return Path(__file__).resolve().parent / _LOCAL_PIPELINES[dataset]
    if dataset in _ADAPTER_PIPELINES:
        return Path(__file__).resolve().parent / _ADAPTER_PIPELINES[dataset]
    try:
        name = _PIPELINES[dataset]
    except KeyError:
        raise ValueError(f"未知数据集 {dataset!r}——没有对应的 pipeline") from None
    from eval.datasets.layout import archive_file

    return archive_file(bench_dir, name)


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


#: 注入里怎么带日期（`--memory-date` 的完整取值域）。
#:
#: * `none`：完全不带（**基线**，也是"AML 侧只取 content"那条 S1 假设）
#: * `per_item`：每条记忆（= 一个段）前面加 `[YYYY-MM-DD] `
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
#: ——"看得见日期"≠"用得上日期"。所以 `header` 把语义**写明**，看是提示不够清楚还是模型做不到。
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

    ⚠ **`created_at` 服务本来就返回了**（`data[]` 四字段之一），是**本函数必须显式带上它**。
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
    execution_record: dict[str, str] | None = None,
) -> list[dict]:
    """构造 pipeline `answer` 步读的 JSONL 项。

    字段名**以 pipeline 代码为准、不照 readme**（`generated_answer` 才是 stage 间的规范
    字段；readme 写的 `predicted_answer` / `hypothesis` **没有 pipeline 读**）。

    `gold_answer` 直接放 `Question.gold` 的原始形态（LoCoMo 是 `list[str]`）——
    归档的 `gold_answer()` 走 `memory_text()`，**列表由它自己用 `\\n` 拼**，
    在这里预先拼会改变那条链路的行为。

    ## `execution_record`（**默认 `None` = 与本参数存在之前逐字相同**）

    `qid → 一段机器生成的执行记录`。给了就在**记忆原文之前**前置它，再一起过平台前缀截断
    （顺序固定：**前部执行记录、后部真实记忆原文**）。

    ⚠ **它是 Plan 模式专用的答案输入口径**（`agentic.mode=plan` 的 `plan_history`）：
    那一路会生成**中间跳的答案**，而答案阶段需要看到它们。⛔ React 模式**不传它**，
    于是响应与答案输入都逐字不变——这也是"不改 React 输出、不破公共 Search 契约"的落点：
    执行记录**不坐 Search 的响应**，只由本地评测从诊断记录里取出来前置。

    ⚠ **它必须自带"这是模型生成的执行记录"的标记**（渲染在 `tools/coq_eval.py`）：
    ⛔ 不伪装成记忆、不编造 memory id、不进向量索引。
    """
    if sample.dataset == "clbench":
        return _build_clbench_items(sample, hits_by_qid)
    if sample.dataset == "beam":
        return _build_beam_items(
            sample, hits_by_qid, date_mode=date_mode, annotate_mark=annotate_mark
        )
    if sample.dataset == "personamem-v2":
        return _build_personamem_items(
            sample, hits_by_qid, date_mode=date_mode, annotate_mark=annotate_mark
        )
    if sample.dataset in EXTRA_DATASETS or sample.dataset == "halumem":
        return _build_extra_items(
            sample,
            hits_by_qid,
            date_mode=date_mode,
            annotate_mark=annotate_mark,
            execution_record=execution_record,
        )
    if execution_record:
        # ⛔ **响亮失败，不静默忽略**：这几个数据集的项形状是逐份定制的（CL-Bench 是
        #   嵌套的 `retrieval.selected`、BEAM / PersonaMem 各有自己的键），
        #   "前置执行记录"在那几种形状上**没有定义**。静默丢掉它会让 Plan 那一臂
        #   拿着**少一段上下文**的输入去判分，而分数看起来完全正常。
        raise ValueError(
            f"`execution_record` 还没有为 {sample.dataset!r} 的输入形状定义——"
            "请在对应的 `_build_*_items` 里接上，或者不要给它传执行记录。"
        )

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
        text = _prepend_execution_record(text, (execution_record or {}).get(question.qid))
        # ★ 平台的答案阶段是"按返回顺序取 117,760 token 前缀"——**harness 要自己模拟**，
        #   因为基线未必守预算（实测 ReFind 有一题返回 89.8 万字符）。详见常量的注释。
        #   ⚠ 顺序：**先前置执行记录、再截断**——反过来的话执行记录会把已经过预算
        #   的记忆挤出窗口，而"前部是执行记录"这条口径就没意义了。
        text, cut = truncate_to_platform_prefix(text)
        if cut:
            _warn_truncated(question.qid)
        item[memory_field] = text
        items.append(item)
    return items


def _prepend_execution_record(text: str, record: str | None) -> str:
    """把执行记录放到记忆原文**之前**。**两条注入路径共用这一行字。**

    ⚠ 两种"没有"要分开：`record is None`（这一题没有记录，例如 React 那一臂）
    与 `record == ""`（有记录但渲染出来是空的）——后者仍然前置（那是空块，无害）。
    """
    if not record:
        return text
    return f"{record}\n\n{text}"


def _build_extra_items(
    sample: Sample,
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    date_mode: str,
    annotate_mark: str,
    execution_record: dict[str, str] | None = None,
) -> list[dict]:
    """本地评测的平铺记忆输入项（含复用上游 QA 裁判的 HaluMem）。

    **形状是我们定的**（没有对应的官方 AML 输入适配）：
    记忆走一个平铺的 `retrieved_context`，题目与金标原样带上，
    `dataset` 让 `extra_pipeline.py` 知道该用哪套判分。⇒ **分数只在仓内前后对比**。

    `execution_record` 的含义见 `build_input_items`（**默认 `None` = 逐字不变**）。
    """
    items: list[dict] = []
    for question in sample.questions:
        hits = hits_by_qid.get(question.qid, [])
        text = render_memories(hits, date_mode=date_mode, annotate_mark=annotate_mark)
        text = _prepend_execution_record(text, (execution_record or {}).get(question.qid))
        text, cut = truncate_to_platform_prefix(text)
        if cut:
            _warn_truncated(question.qid)
        items.append(
            {
                "id": question.qid,
                "dataset": sample.dataset,
                "question": question.question,
                "gold_answer": question.gold,
                "category": question.category,
                "retrieved_context": text,
            }
        )
        if sample.dataset == "halumem":
            # 复用采集分派层的上游三分类裁判；证据只在裁判侧，不进答案 prompt。
            gold = question.gold
            if not isinstance(gold, dict) or not isinstance(gold.get("answer"), str):
                raise ValueError(f"HaluMem {question.qid}：缺参考回答")
            items[-1]["gold_answers"] = [gold["answer"]]
        if sample.dataset == "musique":
            from eval.datasets.musique import answer_contract_for

            # ⭐ **只挑"链式多跳"那一档**：它没有拒答题 ⇒ 不给拒答出口
            #    （那句子不是保险，是出口：实测这一档 10/25 是模型自己弃答，
            #    而它要测的是"检索有没有把证据送到它面前"）。
            # ⛔ MuSiQue 原生的可答/不可答变体**照旧**——整档大数据评测确实要考拒答。
            items[-1]["answer_contract"] = answer_contract_for(question.category)
        if sample.dataset in {"hybridqa", "feverous"}:
            from eval.datasets.feverous import ANSWER_CONTRACT as FEVEROUS_CONTRACT
            from eval.datasets.hybridqa import ANSWER_CONTRACT as HYBRIDQA_CONTRACT

            items[-1]["answer_contract"] = (
                HYBRIDQA_CONTRACT if sample.dataset == "hybridqa" else FEVEROUS_CONTRACT
            )
    return items


def _warn_truncated(qid: str) -> None:
    """注入被平台前缀截断——**两条注入路径共用这一行字**。"""
    print(f"  ⚠ {qid}：注入被截到平台前缀（{PLATFORM_TOKEN_PREFIX:,} token）", flush=True)


#: 官方采集行 → 该数据集自己的答案契约（缺省 = 走通用 prompt）。
#: 判分侧按 `item["dataset"]` 分派；答案侧靠这个字段选中**同一份实现**。
#: ⚠ MuSiQue 采集行没有 `category` ⇒ 一律按原生 MuSiQue-Full（`ANSWER_CONTRACT`，
#: 含拒答出口）处理，不推断成链式档。
_OFFICIAL_ANSWER_CONTRACTS = {
    "musique": MUSIQUE_ANSWER_CONTRACT,
    "hybridqa": HYBRIDQA_ANSWER_CONTRACT,
    "feverous": FEVEROUS_ANSWER_CONTRACT,
}


def build_official_items(
    questions: list[dict],
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    date_mode: str = "none",
    annotate_mark: str = "paren",
) -> list[dict]:
    """官方采集（`official-capture`）的输入项——**题是套件的一行，不是 `Question`**。

    ⚠ 与其余建造函数的区别只有一处：**这一族题的"数据集"是逐题不同的**
    （同一个 run 里 locomo / scriptmem / mquake … 混着走），所以 `dataset` 取自
    **题自己的归属**（`official-eval-kit.jsonl` 的 `dataset` 字段），而不是 `Sample.dataset`。
    分派 pipeline（[`official_capture_pipeline.py`](./official_capture_pipeline.py)）
    正是按它选 answer prompt 与裁判。

    `gold_answer` 给**该数据集裁判要的原生形状**：有 `gold_native` 就原样给
    （CorporateBench 的 `answer_type`、MedMemoryBench 的 `query_type`…），否则给
    `gold_answers` 那个列表（MQuAKE / TempReason 的别名表就是它）。
    """
    items: list[dict] = []
    for row in questions:
        qid = str(row["seq"])
        hits = hits_by_qid.get(qid, [])
        text = render_memories(hits, date_mode=date_mode, annotate_mark=annotate_mark)
        text, cut = truncate_to_platform_prefix(text)
        if cut:
            _warn_truncated(qid)
        gold = row.get("gold_native")
        item = {
            "id": qid,
            "seq": row["seq"],
            "dataset": row["dataset"],
            "judge_kind": row["judge_kind"],
            "task": row.get("task"),
            # **原样**是采集里那一串（含选项与尾指令）——线上检索用的就是它，
            # 改动会让检索侧的输入与线上不一致。
            "question": row.get("query"),
            "question_text": row.get("question"),
            "options": row.get("options") or [],
            "instruction": row.get("instruction"),
            "gold_answer": gold if gold is not None else (row.get("gold_answers") or []),
            "gold_answers": row.get("gold_answers") or [],
            "gold_rubric": row.get("gold_rubric"),
            "gold_judging": row.get("gold_judging") or "",
            "retrieved_context": text,
        }
        contract = _OFFICIAL_ANSWER_CONTRACTS.get(row["dataset"])
        if contract:
            item["answer_contract"] = contract
        if row["dataset"] == "medmemorybench" and isinstance(gold, dict) and gold.get("query_type"):
            # 该数据集自己的题类字段；临床多跳的专用答案 prompt 靠它分派。
            item["category"] = str(gold["query_type"])
        items.append(item)
    return items


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


#: 与 `budget.tokenizer`（`src` 的 `DEFAULT_TOKENIZER`）**同一口径**。
#: 提成具名常量是为了让 `tests/test_experiments.py::test_platform_token_budget_matches_src`
#: 能断言两处相等——**两侧各写各的字面量，分叉时不会报错**。
PLATFORM_TOKENIZER: Final[str] = "o200k_base"


def _encoder():
    import tiktoken  # 只在真要用时才 import（harness 的纯逻辑用例不必装它）

    return tiktoken.get_encoding(PLATFORM_TOKENIZER)


def _encode(encoder, text: str) -> list[int]:
    """**唯一的编码入口**——`disallowed_special=()` 只在这里给。

    ⛔ **不给这个参数会让整轮跑批崩掉，而且崩在数据上、不是崩在代码上**：
    `o200k_base` 默认把 `<|endoftext|>` 这类**字面字符串**当成特殊 token 并**抛
    `ValueError`**。而记忆正文里出现它完全正常——**2026-10-02 实测两条链各自死在
    这里**：LongMemEval 的 haystack（`run-lme.out`）与 CorporateBench 的邮件正文。
    两者的共同点是"**重跑一次还死在同一题上**"（数据没变），看起来像"这一题特别慢"，
    **极易被误判成网络/网关问题**。

    ⚠ `src/tianximem/common/tokens.py` 的 `O200kCounter.count` **一直是传的**——
    但 harness **不许 import `src/`**（[`../CLAUDE.md`](../CLAUDE.md)），
    所以这两个字面量**故意重复了一份**（同 `PLATFORM_TOKEN_PREFIX` 那条），
    只能靠本注释与 `tests/test_harness.py` 的用例守。
    """
    return encoder.encode(text, disallowed_special=())


def truncate_to_platform_prefix(text: str) -> tuple[str, bool]:
    """按平台规则截断；返回 `(文本, 是否截过)`。"""
    encoder = _encoder()
    ids = _encode(encoder, text)
    if len(ids) <= PLATFORM_TOKEN_PREFIX:
        return text, False
    return encoder.decode(ids[:PLATFORM_TOKEN_PREFIX]), True


def _build_clbench_items(sample: Sample, hits_by_qid: dict[str, list[SearchHit]]) -> list[dict]:
    """CL-Bench 的项——**形状与那两份完全不同**
    （见 [`../datasets/clbench.py`](../datasets/clbench.py)）。

    | 字段 | 哪来的 / 为什么 |
    | --- | --- |
    | `idx` | `clb_pipeline.py` 的 `row_id()` **先认 `idx`**（`answer` 靠它跳过已完成） |
    | `system_prompt` | 记录里第一条 `system` 消息——**raw 文件里没有这个顶层键**，
    不补就塌成空串 |
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


def _build_personamem_items(
    sample: Sample,
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    date_mode: str,
    annotate_mark: str,
) -> list[dict]:
    """保留官方选项字段；每题的历史只来自该题的 Search 结果。"""
    from .personamem_pipeline import retrieved_history

    items: list[dict] = []
    for question in sample.questions:
        gold = question.gold if isinstance(question.gold, dict) else {}
        item = {
            "id": question.qid,
            "question": question.question,
            "persona_id": gold.get("persona_id"),
            "correct_answer": gold.get("correct_answer", ""),
            "incorrect_answers": gold.get("incorrect_answers", []),
            "retrieved_context": render_memories(
                hits_by_qid.get(question.qid, []),
                date_mode=date_mode,
                annotate_mark=annotate_mark,
            ),
        }
        item["chat_history"] = retrieved_history(item)
        items.append(item)
    return items


def _build_beam_items(
    sample: Sample,
    hits_by_qid: dict[str, list[SearchHit]],
    *,
    date_mode: str,
    annotate_mark: str,
) -> list[dict]:
    """BEAM 的项——**金标是 `rubric` 列表**，记忆走 `context`。

    | 字段 | 哪来的 / 为什么 |
    | --- | --- |
    | `id` | `answer` 步靠它跳过已完成 |
    | `context` | `context_text()` 的**首选**键（回退链见它的实现：`context` → … → speaker 块） |
    | `rubric` | `rubric_items()` 认的三个键之一——**判决逐条三点制**，缺了它这条直接崩 |
    | `question_type` | 判分结果里带上（`event_ordering` 那一组另走对齐分支） |

    ⚠ 记忆块按平台的 117,760 token 前缀截断，**按项截**（不切半个段）——与
    `packaging` 的"段是原子单位"同一条理由。
    """
    items: list[dict] = []
    for question in sample.questions:
        hits, cut = _prefix_within_budget(hits_by_qid.get(question.qid, []))
        if cut:
            _warn_truncated(question.qid)
        items.append(
            {
                "id": question.qid,
                "question": question.question,
                "context": render_memories(hits, date_mode=date_mode, annotate_mark=annotate_mark),
                "rubric": list(question.gold or []),
                "question_type": question.category,
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
        cost = len(_encode(encoder, hit.content))
        if used + cost > budget:
            return kept, True
        kept.append(hit)
        used += cost
    return kept, False
