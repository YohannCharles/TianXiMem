"""**冻结的抽样口径**——"每个数据集跑多少题"的唯一声明处。

## 为什么必须有它

本仓的分数**只在同一口径内可比**，而 10 个数据集的**全量题数差着量级**
（2026-10-02 实测）：

```text
mquake 69,618 · tempreason 9,823 · personamem 5,000 · medmemorybench 1,939
clbench 1,899 · memtrapbench 1,050 · corporatebench 750 · lme 500 · beam 400 · locomo 1,382
```

⇒ **全量跑不成立**（mquake 全量按本机速率要跑几十天），每个数据集都必须有一个
"取多少"的口径。在此之前它只活在**各次 run 的命令行里**（`--limit 3` 之类），
**下一次想复现得自己猜**——而"猜错一个数"的后果是**分数看起来完全正常、却不可比**。

## ⚠ `limit` 在每个数据集上数的**不是同一个东西**

这是本表最容易被读错的一列，逐数据集实测过：

* **对话数**（locomo / beam）——`limit 3` 是**三段对话**，不是三题
* **题数**（longmemeval）——每条记录 = 1 题，且**各自带一整套 haystack**
* **任务数**（clbench）——1 条任务 = 1 题 = 1 份自己的文档语料
* **persona 数**（personamem）——`limit 11` 是 11 个人，共 319 题
* **样本数**（medmemorybench / tempreason / memtrapbench）——按 (persona, 检查点) / 页 / 记录
* **QA 子集**（corporatebench）——`limit 1` 选 `kb_qa` 那一份（250 题），**语料不跟着裁**
* **chunk 数**（mquake）——每个 chunk 自带一整套语料

⇒ **对 beam 断言"--limit 15 = 15 题"是错的**（那是 15 个对话 = 300 题）。

## ⚠ `spread` 不是可选项，是**必选项或禁用项**（逐数据集）

多数数据集**按类别分块排**（LME 按 `question_type`、CL-Bench 按 `context_category`、
beam 按 `conversation_seed.category` …），不加 `--spread` 的 `[:limit]` 会**塌成单一类别**
——**分数看起来正常**，只是那一类被当成了全体。少数几份是 no-op（locomo / personamem）。

`--spread` 用的是 [`sampling.py`](../datasets/sampling.py) 的 `stratified_sample`：
组内**等间隔**取（无随机、无 seed）⇒ **可复现**。⚠ 但它有个地板：
**每组至少取 1 条 ⇒ 实际条数可能多于 `limit`，上界是"组数"**。

## 这个表怎么用

```bash
# 一键（口径自动带上）
uv run python eval/experiments/run.py --dataset clbench --frozen --base-url ...

# 或者把 flags 拷出来自己拼
uv run python -c "from eval.experiments.recipes import flags_for; print(flags_for('clbench'))"
```

**`n_questions` 那一列是被钉住的**：`tests/test_experiments.py` 会逐个加载、断言
实际题数等于它——**改了 k 而忘了改这里，测试立刻红**（否则"口径漂了"只会在
两次 run 的分数不可比时**静默**表现出来）。

> ⚠ **加载器改了、题数就会变**（例如 V14 那次跳过空正文）。这张表因此**依赖归档数据**：
> 没有 `dataset/` 时那条断言会 skip，**但不会假装通过**。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

__all__ = ["AML_FROZEN_RECIPES", "FROZEN_RECIPES", "Recipe", "flags_for", "recipe_for"]


@dataclass(frozen=True, slots=True)
class Recipe:
    """一个数据集的冻结口径。**`n_questions` 是复核实测值，不是估算。**"""

    #: 传给 `run.py --limit` 的值。`None` = **不抽样**（该数据集全量就在可接受范围内）。
    limit: int | None
    #: 传不传 `--spread`。**no-op 的数据集写 `False`**（写了也只是噪音）。
    spread: bool
    #: **实测**题数——`tests/test_experiments.py` 拿它当断言值。
    n_questions: int
    #: `limit` 数的**是什么**（本表最容易读错的一列，见模块 docstring）。
    axis: str
    #: 这个数字是怎么来的 / 会怎么骗人。**别删警告。**
    note: str = ""

    def flags(self) -> str:
        """可直接粘到 `run.py` 上的参数串。"""
        parts: list[str] = []
        if self.limit is not None:
            parts += ["--limit", str(self.limit)]
        if self.spread:
            parts.append("--spread")
        return " ".join(parts) or "(none)"


#: **冻结口径表**。依据：2026-10-02 逐个数据集实测 + **独立复核**（复核者重新加载、
#: 并试图证伪"轴 / spread 必要性 / 确定性 / 覆盖度"四条）。
FROZEN_RECIPES: Final[dict[str, Recipe]] = {
    "hybridqa": Recipe(
        limit=120,
        spread=True,
        n_questions=120,
        axis="**题数**（抽题后按 table_id 共用完整表格/关联段落）",
        note="按上游 table/passage/other 分层；文本检索输入为本地适配，EM/token-F1 复用上游函数。",
    ),
    "feverous": Recipe(
        limit=120,
        spread=True,
        n_questions=123,
        axis="**claim 数**（每题通过固定 claim-only 索引检索候选整页）",
        note=(
            "按 label/challenge 分层；候选池不读金标，页面数与索引版本由加载器固定。"
            "记录标签及严格证据组指标；属于本地候选检索评测，不与完整官方系统分数对齐。"
        ),
    ),
    "halumem": Recipe(
        limit=60,
        spread=True,
        n_questions=203,
        axis="**提问检查点数**（一个用户的一个提问会话）",
        note=(
            "按用户分层，覆盖各类 QA；检查点只投喂当时可见的对话，使用独立 user_id。"
            "后续检查点重复投喂历史；本地只测 QA，不测上游记忆抽取/更新指标。"
        ),
    ),
    "musique": Recipe(
        limit=120,
        spread=True,
        n_questions=120,
        axis="**题数**（每题自带候选段落，可答/不可答变体独立）",
        note=(
            "Full dev 按跳数及可答性分层；可答/不可答两种变体均保留。"
            "只测本地答案别名归一化后精确匹配与拒答，不是上游 F1/sufficiency 指标。"
        ),
    ),
    "locomo-refined": Recipe(
        limit=3,
        spread=False,
        n_questions=346,
        axis="**对话数**（10 段里取前 3 段；切片在加载之后做）",
        note=(
            "这条轴上只有 k=3 落在 250–350：k=2→210、k=4→525（大跳，conv-42 自带 179 题）。"
            "⚠ 类别不均是固有的：open-domain 只有 12 题 ⇒ 那一维子分没有分辨率。"
            "与现有基线（0.7775）同口径。"
        ),
    ),
    "longmemeval-s": Recipe(
        limit=100,
        spread=True,
        n_questions=101,
        axis="**题数**（每条记录 = 1 题，各带一整套 haystack ⇒ 每题一个 user）",
        note=(
            "全量 500。**必须 spread**：不加会整类丢掉 knowledge-update 与 "
            "single-session-assistant。"
            " ⚠ **2026-10-02 由 300 降到 100**：`--limit 300` 实测 **4.2 分钟/题**"
            "（每题要灌 ~250 块、且 500 份 haystack 互不重复）⇒ 整轮 ~20 小时，"
            "而本地网关在 Cloudflare 后面、撑不住这种长占。降档的代价是"
            "**噪声底从 ~1pt 升到 ~1.8pt**（按 1/√n 折算）⇒ 它只挡得住 2pt 以上的改动。"
            " ⚠ 换 k **不是取子集**（组内 step 随 k 变，k=100 与 k=300 的题集互不包含）"
            "⇒ **不要拿两次不同 k 的分数比**。"
        ),
    ),
    "clbench": Recipe(
        limit=300,
        spread=True,
        n_questions=299,
        axis="**任务数**（1 条任务 = 1 题，自带自己的文档语料）",
        note=(
            "全量 1,899。**必须 spread**：不加时 Procedural Task Execution 只占 4%"
            "（全量 24.8%）。⚠ 取整抖动实测：k=300 拿到 **299** ⇒ 冻结的是 k，不是题数。"
            "✅ **本机能跑**（2026-10-09 复测）：全量墙钟 **≈3.5 小时**"
            "（`--judge-workers 4`；Add 那一段仍是串行）。"
            " ⚠ **它的官方分是全有全无的 LLM rubric 判分**（`clb_pipeline.py` 的原文："
            '"strict, all-or-nothing … The final score is binary"）⇒ 一道题从 0 翻到 1 要'
            "**每一条** rubric 都满足，**中间进展在 `overall` 里看不见**。"
            " 但归档裁判同时写 `requirement_ratio`（满足条数/总条数），本仓已把它接成"
            " **`partial_credit`**（见 [`../../reports/CLAUDE.md`](../../reports/CLAUDE.md)）"
            "——**判断改动要看那一列**，`overall` 只用来对齐榜分。"
        ),
    ),
    "beam": Recipe(
        limit=None,
        spread=False,
        n_questions=400,
        axis="**对话数**（20 个对话 × 每个 20 题）",
        note=(
            "**已是全量**（本卡 100K 只有 20 行 parquet）。"
            "✅ **本机能跑**（2026-10-09 复测）：单题上下文 **≈110k token**"
            "（2 题 879 KB），全量墙钟 **≈1 小时**（`--judge-workers 4`）。"
            "⚠ 一旦要抽样就**必须 spread**（8 个 category 分块排）。"
            " 细节见 eval/reports/ledger.md 的「跑一轮要多久」。"
        ),
    ),
    "corporatebench": Recipe(
        limit=1,
        spread=False,
        n_questions=250,
        axis="**QA 子集**（`kb_qa` / `topic_qa` / `integrated_qa` 三份，各 250 题）",
        note=(
            "**语料不随 limit 裁**（裁了就没法检索）⇒ 三个子集共用同一份 353 篇文档。"
            "`spread` 在这里没有分组键可言（loader 根本不调用它）。"
        ),
    ),
    "mquake-remastered": Recipe(
        limit=1,
        spread=True,
        n_questions=768,
        axis="**chunk 数**（每个 chunk 自带一整套事实语料）",
        note=(
            "**必须 spread**：四份 parquet（CF3k/CF6334/CF9k/T）按文件顺序拼，"
            "不加只会取到 CF3k。k=1 是四组全在场的**最小**档 ⇒ 768 题超出常规区间，"
            "但**代表性优先于题量**（单组样本的分数不能代表这份数据集）。"
        ),
    ),
    "medmemorybench": Recipe(
        limit=40,
        spread=True,
        n_questions=388,
        axis="**样本数**（一个样本 = 一个 persona 的一个检查点）",
        note="20 个 persona 组、每组恰好 10 条。下一档 k=60→564，中间没有档位。",
    ),
    "memtrapbench": Recipe(
        limit=250,
        spread=True,
        n_questions=250,
        axis="**记录数**（1 条记录 = 1 题）",
        note=(
            "**必须 spread**：不加只覆盖 2 个场景（Inertia / number_game）。"
            "加了之后 6 个场景各 36–48 题。"
        ),
    ),
    "personamem-v2": Recipe(
        limit=11,
        spread=False,
        n_questions=319,
        axis="**persona 数**（`limit 11` = 11 个人，共 319 题）",
        note=(
            "⚠ `spread` 在这份上是 **no-op**（加载器收下参数但从不使用），所以写 False。"
            "本地适配器用 Search 片段替代历史；旧完整历史成绩不可比，重跑使用新 run-id。"
        ),
    ),
    "tempreason": Recipe(
        limit=64,
        spread=True,
        n_questions=332,
        axis="**页数**（`key = (文件, 主体)`）",
        note="2 组（`test_l2` / `test_l3`）。下一档 k=84→404。",
    ),
}


AML_FROZEN_RECIPES: Final[dict[str, Recipe]] = {
    name: replace(
        FROZEN_RECIPES[name],
        note="AML-compatible input; see the input manifest for corpus/timeline scope.",
    )
    for name in (
        "corporatebench",
        "memtrapbench",
        "locomo-refined",
        "medmemorybench",
        "halumem",
        "musique",
        "hybridqa",
        "feverous",
    )
}
AML_FROZEN_RECIPES["mquake-remastered"] = Recipe(
    limit=40,
    spread=True,
    n_questions=240,
    axis="public case count; original and updated QA",
    note="One public case per user; all four sources sampled; original→QA→UPDATE→QA.",
)
for _name, _axis in {
    "corporatebench": "QA 子集数；合并为一个公司用户，每篇文档独立 session",
    "locomo-refined": "对话数；每段对话配一个确定性公开 LME haystack",
    "halumem": "源提问检查点数；同一 persona 持续 Add/Search",
    "medmemorybench": "源提问检查点数；同一 persona 持续 Add/Search",
    "feverous": "claim 数；全部问题共用声明的整页 JSON 池",
}.items():
    AML_FROZEN_RECIPES[_name] = replace(AML_FROZEN_RECIPES[_name], axis=_axis)
AML_FROZEN_RECIPES["feverous"] = replace(
    AML_FROZEN_RECIPES["feverous"],
    n_questions=116,
    note=(
        "Default captured Add pool and same-user captured Search claims; "
        "explicit pools define a different source scope."
    ),
)


def recipe_for(dataset: str, *, input_contract: str = "native") -> Recipe:
    """取一个数据集的冻结口径。**未知数据集响亮失败**（不静默退回"不抽样"）。"""
    if input_contract not in {"native", "aml-v1"}:
        raise ValueError(f"Unknown input contract {input_contract!r}")
    table = FROZEN_RECIPES if input_contract == "native" else AML_FROZEN_RECIPES
    try:
        return table[dataset]
    except KeyError:
        known = " / ".join(sorted(table))
        raise KeyError(f"数据集 {dataset!r} 没有冻结口径。已登记的是：{known}") from None


def flags_for(dataset: str, *, input_contract: str = "native") -> str:
    """该数据集的参数字符串（可直接粘到 `run.py`）。"""
    flags = recipe_for(dataset, input_contract=input_contract).flags()
    return flags if input_contract == "native" else f"--input-contract {input_contract} {flags}"
