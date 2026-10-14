"""MuSiQue-Full dev：每道题的候选段落组成一个独立用户的记忆语料。

Full 的可答/不可答变体共用 id，user_id/qid 必须带变体标记。只把 title 与
paragraph_text 送进 Add；answer、is_supporting 与 question_decomposition 均不注入。
spread 按跳数和可答性分层。判分有**两套口径**（可答题：默认规则判分、`llm` 为模型裁判；
拒答恒走精确匹配），**都不复现**上游支持段落 F1 或成组 answer-sufficiency 指标。
原生文本包装与采集不同；候选段落顺序保留源数据顺序。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from .preprocess import Message, Question, Sample, Session, normalize_content
from .sampling import stratified_sample

DATA_DIR = "musique"
JSONL = "musique_full_v1.0_dev.jsonl"
REFUSAL = "INSUFFICIENT_EVIDENCE"
ANSWER_CONTRACT = "musique-local-qa-v1"
#: **宽松口径**：金标作为"一个词"出现在回答里即算对（见 `extra_pipeline.judge_musique`
#: 的对照表）。⚠ 它会把"答得比金标更具体"也算对，**同时制造假阳性** ⇒
#: **两套口径并排报，不许替换**。2026-10-11 实测它只翻 1 道。
LOOSE_ANSWER_CONTRACT = "musique-local-qa-v2-loose"
#: **链式多跳那一档**的答案口径（2026-10-12）：**不给拒答出口**。
#:
#: ⛔ 它**不是**"改写严的那条"——**判分与 v1 逐字相同**（都走归一化精确匹配）。
#: 它只改**答案 prompt**：这一档**必须给出一个答案**，不许回 `INSUFFICIENT_EVIDENCE`。
#:
#: **为什么单列一档**：那句"证据不足就回 INSUFFICIENT_EVIDENCE"是个**出口**，
#: 而本仓已经量过两次它的后果——普通 QA 250 题上 **160/250 逐字拒答、通过 0**，
#: 只删掉那句话 ⇒ 总分 0.3997 → **0.4857**；TempReason 逐字拒答的 76 道
#: **证据全部就在眼前**，删掉 ⇒ **44/76 新答对**（0.7711）。
#: MuSiQue 链式那 25 题上同一个病：**10/25 是模型自己拒答**。
#: 而多跳这一档要测的是"检索有没有把证据送到它面前"，**不是"它会不会认输"**。
#:
#: ⛔ **只改这一档**：MuSiQue 原生的可答/不可答变体、以及**别的数据集**的评测**一律不动**
#: （那些历史数字要可比）。⇒ 选口径的是 `answer_contract_for(question.category)`，
#: 判据是"是不是链式那一档"，**不是**题目可不可答。
DIRECT_ANSWER_CONTRACT = "musique-local-qa-v3-direct"

#: 链式 haystack 那一档的 `category` 后缀（`load_musique_haystack` 产的题都带它）。
CHAIN_CATEGORY = "chain"


def answer_contract_for(category: str) -> str:
    """这一档该用哪条答案口径。

    ⭐ **只回答"链式多跳"这一档**：它是 `…-hop/chain`，用 `DIRECT_ANSWER_CONTRACT`（无拒答出口）；
    其余一切（MuSiQue 原生的 `…-hop/answerable` / `…-hop/unanswerable`，以及别的数据集）
    **照旧走 `ANSWER_CONTRACT`**——它们的评测口径一个字都不改。
    """
    return (
        DIRECT_ANSWER_CONTRACT if str(category).endswith(f"/{CHAIN_CATEGORY}") else ANSWER_CONTRACT
    )


USER_PREFIX = "musique-"
SHAPE_NOTE = (
    f"MuSiQue-Full dev（{ANSWER_CONTRACT}）：每题一个独立 user，"
    "语料只含候选段落 title/paragraph_text。"
    "可答/不可答变体用不同 user_id/qid；spread 按跳数和可答性分层。"
    "本地评分为**可答两套口径**：默认 `extra_pipeline --musique-judge rule`（答案别名归一化后"
    "精确匹配，快速复现历史数字），显式 `llm` 才判语义等价；拒答标记恒走精确匹配。"
    "**都不是**上游答案/支持段落 F1 或成组 sufficiency 指标；"
    "候选段落保留源数据顺序，原生包装与采集不同。"
)


def _variant(row: dict) -> str:
    if not isinstance(row.get("answerable"), bool):
        raise ValueError(f"MuSiQue {row.get('id')}：answerable 必须为 bool")
    return "answerable" if row["answerable"] else "unanswerable"


def read_rows(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[dict]:
    """加载 Full dev；limit 数题目，spread 覆盖不同跳数及可答/不可答变体。"""
    path = Path(bench_dir) / DATA_DIR / JSONL
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(str(row["id"]), _variant(row)) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("MuSiQue：同一 id/可答性变体重复，qid 会冲突")
    if limit is not None:
        if limit < 0:
            raise ValueError("MuSiQue：limit 必须非负")
        if limit == 0:
            return []
        rows = (
            stratified_sample(
                rows,
                limit,
                key=lambda row: f"{len(row['question_decomposition'])}-hop/{_variant(row)}",
            )
            if spread
            else rows[:limit]
        )

    return rows


def load_musique(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    return samples_from_rows(read_rows(bench_dir, limit=limit, spread=spread))


def samples_from_rows(rows: list[dict]) -> list[Sample]:
    """共享题目/金标验证；AML 适配从同一原始 row 读取段落结构。"""
    samples: list[Sample] = []
    for row in rows:
        variant = _variant(row)
        user_id = f"{USER_PREFIX}{row['id']}-{variant}"
        messages = tuple(
            Message(
                role="user",
                content=normalize_content(
                    f"Title: {paragraph['title']}\n"
                    + normalize_content(
                        paragraph["paragraph_text"], where=f"MuSiQue {user_id} paragraph body"
                    ),
                    where=f"MuSiQue {user_id} paragraph {paragraph['idx']}",
                ),
            )
            for paragraph in row["paragraphs"]
        )
        if not messages:
            raise ValueError(f"MuSiQue {user_id}：候选段落为空")
        answer = row["answer"]
        aliases = row["answer_aliases"]
        if not isinstance(answer, str) or (row["answerable"] and not answer.strip()):
            raise ValueError(f"MuSiQue {user_id}：可答题参考答案为空或类型错误")
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError(f"MuSiQue {user_id}：answer_aliases 必须为字符串列表")
        question = Question(
            qid=user_id,
            question=normalize_content(row["question"], where=f"MuSiQue {user_id} question"),
            gold={"answer": answer, "answer_aliases": aliases, "answerable": row["answerable"]},
            category=f"{len(row['question_decomposition'])}-hop/{variant}",
            evidence=tuple(str(p["idx"]) for p in row["paragraphs"] if p["is_supporting"]),
            is_abstention=not row["answerable"],
        )
        samples.append(
            Sample(
                user_id=user_id,
                dataset="musique",
                sessions=(Session(session_id=f"{user_id}-corpus", messages=messages),),
                questions=(question,),
            )
        )
    return samples


# ── 链式依赖的 haystack 设定（2026-10-11）────────────────────────────────
#
# ## 为什么需要它
#
# MuSiQue 原生**每题只有 20 个候选段落**（实测 4834 题里 4803 题都是 20）。
# 而这条链是 `prefetch_limit=200（两路）→ RRF 根 limit=400 → 截到 top_k=100`：
# **20 < 100 < 400** ⇒ 那 20 段**原封不动全进精排、全进最终 prompt**
# （117,760 token 装 20 段绰绰有余）。
# ⇒ 答案模型永远看到全部证据（含所有跳的支持段落），**检索无事可做，链式依赖自动消失**。
# 拿这样的语料去测多跳检索，测出来的只会是"多余的一轮把答案搅坏"——与 2026-10-10 那次同型。
#
# ## 做法：把 N 道题的段落并成**一个用户**的语料
#
# 按**渲染后的正文**去重（多道题共用同一段 ⇒ 只留一份）⇒ 50 道题的 1000 个槽位得到 **912 段**，
# 远大于 `top_k`，检索真的有得选。
# 目标题自己的支持段落就在池子里，被其余 49 道题的段落埋着。
#
# ⛔ **不能按 `title` 去重**（2026-10-12 修）：`title` 是维基**文章名**，而一个候选段落是那篇
# 文章的一个 **chunk** ⇒ 同标题下可以有好几段**不同的正文**，其中可能有某道题的**支持段落**。
# 按 title 去重把它们当重复丢掉 ⇒ **那几道题在池子里根本无解**，而池子看起来完全正常。
# 实测：50 道题里**丢掉 16 段支持段落、涉及 14 道题**，16/16 都能找到"同标题的另一些段"。
# ⇒ 该去重的是**内容相同**的段落。
#
# ⚠ **只并入候选段落，不并入答案与分解**：`answer` / `question_decomposition` /
# `is_supporting` 一律不进 Add 载荷（与 `load_musique` 同一条纪律：§2.1 不许把答案伪装成记忆）。

HAYSTACK_USER = f"{USER_PREFIX}haystack"

#: "严格链"的第二跳起的引用形状（hop 的问句里写 `#1` / `#2`）。
_HOP_REF = re.compile(r"#\d")


def _is_strict_chain(hops: list[dict]) -> bool:
    """hop1 是根，其余**每一跳都引用前面的某一跳**。

    ⚠ **"并行分支"不算**：MuSiQue 里有一部分题是"两个独立子问题在最后合并"
    ——那类问题**第二轮检索是不必要的**（两边可以同时查），正是我们上一轮踩的坑。
    """
    if not hops:
        return False
    if "#" in str(hops[0].get("question") or ""):
        return False
    return all(_HOP_REF.search(str(hop.get("question") or "")) for hop in hops[1:])


def _is_entity_answer(answer: object) -> bool:
    """答案像不像**实体**（人 / 地 / 机构 / 作品）。

    ⚠ 判据取"**一个数字都不含**"：年份 / 日期 / 计数 / 金额在下一跳里**没法当检索词**——
    下一跳的问句形如 `"#1 >> spouse"`，而 `#1` 会被填成 `1999` 或 `March 29, 2018`。
    那不是链式依赖，是一次数值传递。

    ⚠ **实测当前清单那 25 道没有一条的中间跳含数字** ⇒ 收紧这条**不改变已生成的清单**
    （贪心只按顺序删候选，前面的判定不受后面影响），所以它是一次**免费**的收紧。
    """
    text = str(answer or "")
    if not text or any(char.isdigit() for char in text):
        return False
    return sum(char.isalpha() or char.isspace() for char in text) >= 3


def _appears_as_a_word(haystack: str, needle: str) -> bool:
    """`needle` 是否作为**一个独立的词**出现在 `haystack` 里（忽略大小写）。

    ⚠ **不能写成 `needle in haystack`**：中间跳的答案常常只有两三个字母
    （`NES` / `US` / `two`），子串匹配会把 `**geNESis**` 也算成"题面泄了 NES"。
    实测（2026-10-12，子 agent 独立复核）：纯子串版本**误杀 15 条**
    （整个 `…_15840_36002` 家族，Genesis vs NES）；改成词边界后候选 798 → **813**。
    ⚠ 边界用 `(?<![A-Za-z0-9])` 而不是 `\b`：答案里可能带标点（`$72,641`、`May 4`），
    而 `\b` 在标点那侧不成立。
    """
    pattern = rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])"
    return re.search(pattern, haystack, re.IGNORECASE) is not None


def _does_not_leak_the_intermediate_entity(row: dict) -> bool:
    """⭐ **用户看到的那句问题里，不许出现任何中间跳的答案。**

    ⚠ 这是"是不是真链"唯一一条**独立于 `#k` 记法**的判据。前面几条都用
    `question_decomposition` 的 `#k` 占位符来判——而那个占位符**本来就是记法**，
    等于用标注证明标注。只有这一条问的是：**把标注全拿掉，读题的人能不能直接查?**

    实测（2026-10-11）：原判据选出的 25 道里有 **2 道泄题**——
    `4hop1__21282_…` 的题面直接写了 `Angela Merkel`；另一道当时记的是 `4hop1__40316_…`
    写了 `NES`，**那条记录本身是错的**（见下）。加上这条之后候选从 814 降到 **798**，
    而选出的 25 道里换掉了那 2 道。

    ### ⚠ 2026-10-12 更正：判据必须是**词边界**

    上一版写的是 `answer in question`（子串）。于是中间跳答案是 `NES` 时，
    题面里的 **`geNESis`** 也算命中 ⇒ **误杀 15 条**（整个 `…_15840_36002` 家族）；
    当时记成"`40316` 写了 `NES`"，正是这个 bug 的产物。
    改成词边界（`_appears_as_a_word`）后候选 **798 → 813**，`40316` 回到候选池。

    ⚠ 这一次修正**会改动已交付清单**：`40316` 按 id 序在 `…_497223_15840_36002`
    这条骨架上排到前面，于是 `46196`（原清单成员）被"共享 ≥2 跳"挤出 ⇒
    **一道题换一道题**。⇒ 清单要重新生成，而**跑批与分解探针的产物都随之作废**
    （见 [`eval/reports/coq-musique-20261011.md`](../reports/coq-musique-20261011.md)）。
    """
    question = str(row["question"])
    return not any(
        _appears_as_a_word(question, str(hop["answer"]))
        for hop in (row.get("question_decomposition") or [])[:-1]
    )


#: ⛔ **金标坏掉的题**——**声明的排除名单**（id → 理由）。2026-10-12，子 agent 逐题读证据查出。
#:
#: ⚠ **它们过得了全部形式判据**：每一跳的答案串确实在支持段落里逐字存在、
#: 题面不泄中间实体、严格链、可答。坏的是**链的推理**——只有把证据读一遍才看得见
#: ⇒ **写不成规则，只能声明**。与 [`manifest.py`](manifest.py) 的 `local_patch` 同一条纪律：
#: **本地修订必须声明**，不许悄悄改；也**不许**只删掉清单里的行而不在这里记理由。
#:
#: 三类（A/B/C 的定名见 [`../../eval/reports/coq-musique-20261011.md`](../../eval/reports/coq-musique-20261011.md) §7）：  # noqa: E501
#:
#: * **A 跨地区消歧**：中间跳的答案是**短地名**，而同一题的语料里有两处同名不同地
#:   （`Atlanta` 在佐治亚，也在密歇根；`Winchester` 在伊利诺伊，也在弗吉尼亚）
#:   ⇒ 逐字包含检查**原理上看不见**，而链真的断了。
#: * **B 答非所问**：问"哪个城市"金标给**州名**、问"哪个州"给**城市名**
#:   ⇒ 把 `#1` 填进下一跳后不成话。
#: * **C 金标事实错**：能被它自己的支持段落直接反驳。
BROKEN_GOLD: dict[str, str] = {
    "4hop1__129682_59747_211319_557671": "A 跨地区消歧：H2 是佐治亚的 Atlanta，H3 却用密歇根那条的县",  # noqa: E501
    "4hop1__364476_439408_131976_90181": "A 跨地区消歧：H1 的支持段落写的是伊利诺伊的 Winchester，H2 用弗吉尼亚",  # noqa: E501
    "4hop1__43565_624859_355213_203322": "A 跨地区消歧：H2 出生地是俄亥俄的 Cleveland，H3 解到北卡的县",  # noqa: E501
    "3hop1__128554_39743_24526": "A 跨地区消歧：H1 问城市给的是弗吉尼亚【市】（明尼苏达），链却按弗吉尼亚【州】走",  # noqa: E501
    "4hop1__236903_153080_159767_81096": "B 答非所问：问 city，金标给州名 Arizona",
    "4hop1__726391_153080_33952_33939": "B 答非所问：问 city，金标给州名 Arizona",
    "3hop1__131783_131926_90707": "B 答非所问：问州，金标给城市 Minneapolis",
    "3hop1__104996_160713_77246": "B 答非所问：问 majority religion（名词）给形容词 Hindu；末跳金标是短语",  # noqa: E501
    "3hop1__106042_64399_52278": "B 答非所问：H1 金标填进 H2 后不成话；末跳金标是两个先后主体",
    "4hop1__159467_725495_49925_267352": "C 金标事实错：支持段落明写 Luther 发起的是 Reformation，金标给 German Renaissance",  # noqa: E501
}


def _hop_ids(question_id: str) -> frozenset[str]:
    """id 里那几段 hop 标识（`4hop1__a_b_c_d` → `{a,b,c,d}`）。"""
    return frozenset(str(question_id).split("__")[1].split("_"))


#: ⛔ 金标坏掉的题**连同它的"骨架"一起排除**：
#:
#: 缺陷住在**共享的那几跳**里，所以"同尾不同头"的**兄弟题继承同一条坏边**。
#: 实测（2026-10-12）：只按 id 排除时，贪心**立刻把兄弟题补了进来**——
#: `4hop1__129682_59747_211319_557671`（`Atlanta` 佐治亚 → 密歇根的县）被判死后，
#: `4hop1__130276_59747_211319_557671` 顶上来，**同一条坏边、同一个错**；
#: `4hop1__43565_624859_355213_203322` 与 `4hop1__463635_624859_355213_203322` 同理。
#: ⇒ 排除的判据是 "**与某道坏题共享 ≥2 跳**"，不是"id 相同"。
#: ⚠ 这个判据**偏保守**（可能多杀几道好题），但"多杀"比"放坏题进来"安全：
#: 坏题进来**不会报错**，只会把分数压低、把归因带偏。
_BROKEN_SKELETONS: tuple[frozenset[str], ...] = tuple(
    _hop_ids(question_id) for question_id in BROKEN_GOLD
)


def _has_a_broken_skeleton(question_id: str) -> bool:
    """这道题**自己**在金标坏题名单里，或者它与某道坏题**共享 ≥2 跳**。"""
    if str(question_id) in BROKEN_GOLD:
        return True
    hop_ids = _hop_ids(question_id)
    return any(len(hop_ids & broken) >= 2 for broken in _BROKEN_SKELETONS)


def select_chain_question_ids(rows: list[dict], *, limit: int = 50) -> list[str]:
    """按**确定性规则**挑出链式依赖题（`eval/coq-musique.jsonl` 头部写的生成规则）：

    1. `answerable == True`；
    2. **严格链**（见 `_is_strict_chain`）；
    3. **≥3 跳**，且 hop1..hop(n-1) 的答案都是**实体**；
    4. ⭐ **题面不泄中间实体**（见 `_does_not_leak_the_intermediate_entity`）；
    5. ⛔ **金标坏掉的题及其骨架**一律不要（见 `_has_a_broken_skeleton`）——
       形式判据看不见它们，只能声明；
    6. 排序 `(-跳数, id)` 后**贪心**：新题与已选题的 hop 集合**共享 <2 个元素**才收
       —— 共享 ≥2 跳 = 同一条骨架换了个入口，收进来就是重复出题。

    ⚠ 规则写**在代码里**而不是只写在清单的注释里：清单是可被替换的产物，
    而规则必须能重放（同一个 `limit` 永远给出同一批 id）。
    """
    candidates: list[tuple[int, str, frozenset[str]]] = []
    for row in rows:
        if not row.get("answerable"):
            continue
        if _has_a_broken_skeleton(str(row["id"])):
            continue
        hops = row.get("question_decomposition") or []
        if len(hops) < 3 or not _is_strict_chain(hops):
            continue
        if not all(_is_entity_answer(hop.get("answer")) for hop in hops[:-1]):
            continue
        if not _does_not_leak_the_intermediate_entity(row):
            continue
        candidates.append(
            (len(hops), str(row["id"]), frozenset(str(row["id"]).split("__")[1].split("_")))
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))

    picked: list[str] = []
    used: list[frozenset[str]] = []
    for _hops, question_id, hop_ids in candidates:
        if any(len(hop_ids & seen) >= 2 for seen in used):
            continue
        picked.append(question_id)
        used.append(hop_ids)
        if len(picked) >= limit:
            break
    return picked


def load_musique_haystack(bench_dir: str | Path, question_ids: Sequence[str]) -> Sample:
    """把 `question_ids` 这几十道题的候选段落并成**一个用户**的语料，题**共用**这个池子。

    返回**一个 `Sample`**（不是一堆）——这正是这个设定的要害：
    `N 题 : 1 个 haystack`，与 LoCoMo 的 `N 题 : 1 段对话`同形，
    而**与 `load_musique` 的 `1 题 : 1 个 haystack` 相反**。

    ⚠ 段落按标题与正文一起去重、**保留首次出现的顺序**（确定性）：同一篇维基段落会被多道题共用，
    不去重会让池子里出现同一段内容的多个副本——**而那会让"检索到它"变得更容易**。

    ⚠ `Question.evidence` 存的是支持段落的 **`title`**，不是 `idx`：并入之后 `idx`
    只在那道题自己的 20 段里成立，换到池子上就指错东西了。
    """
    wanted = list(dict.fromkeys(question_ids))  # 保序去重
    if not wanted:
        raise ValueError("question_ids 不得为空")

    by_id: dict[str, dict] = {}
    for line in (Path(bench_dir) / DATA_DIR / JSONL).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answerable"):
            by_id.setdefault(str(row["id"]), row)

    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise ValueError(f"MuSiQue：这些 id 不在可答题里：{missing[:3]}（共 {len(missing)} 个）")

    # ⚠ **按渲染后的正文去重，不是按 `title`**（2026-10-12 修）。
    #    `title` 是**维基文章名**，而一个候选段落是那篇文章的一个 **chunk**
    #    ⇒ 同一个 title 下**可以有好几段不同的正文**，其中可能有某道题的**支持段落**。
    #    按 title 去重会把它们当重复丢掉 ⇒ **那几道题在池子里根本无解**，
    #    而池子看起来完全正常（段数、覆盖率、判分都不报错）。
    #    实测（2026-10-12，50 道题）：**16 段支持段落被丢掉、涉及 14 道题**，
    #    16/16 都能在池子里找到"同标题的另一些段" ⇒ 全是这一条造成的。
    #    ⇒ 真正该去重的是**内容相同**的段落（多道题共用同一段），按正文比即可。
    bodies: dict[str, str] = {}  # 渲染后的正文 -> 它自己（内容相同的只留一份）
    for question_id in wanted:
        row = by_id[question_id]
        hops = row.get("question_decomposition") or []
        if not _is_strict_chain(hops):
            raise ValueError(f"MuSiQue {question_id}：不是严格链——清单不该收它")
        for paragraph in row["paragraphs"]:
            body = normalize_content(
                f"Title: {paragraph['title']}\n"
                + normalize_content(
                    paragraph["paragraph_text"], where=f"MuSiQue {question_id} paragraph body"
                ),
                where=f"MuSiQue haystack paragraph {paragraph['title']!r}",
            )
            bodies.setdefault(body, body)
    messages = tuple(Message(role="user", content=body) for body in bodies)

    questions = []
    for question_id in wanted:
        row = by_id[question_id]
        answer, aliases = row["answer"], row["answer_aliases"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"MuSiQue {question_id}：参考答案为空或类型错误")
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"MuSiQue {question_id}：answer_aliases 必须为字符串列表")
        questions.append(
            Question(
                qid=question_id,
                question=normalize_content(
                    row["question"], where=f"MuSiQue {question_id} question"
                ),
                gold={"answer": answer, "answer_aliases": aliases, "answerable": True},
                category=f"{len(row['question_decomposition'])}-hop/chain",
                evidence=tuple(
                    sorted({str(p["title"]) for p in row["paragraphs"] if p["is_supporting"]})
                ),
            )
        )

    return Sample(
        user_id=HAYSTACK_USER,
        dataset="musique",
        sessions=(Session(session_id=f"{HAYSTACK_USER}-corpus", messages=messages),),
        questions=tuple(questions),
    )


def supporting_snippets(
    bench_dir: str | Path, question_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """`qid → 支持段落的**正文片段**`——**只给诊断用**（覆盖率）。

    ⚠ **不能用 `title`**：MuSiQue 的 `title` 是**维基文章名**，而一个候选段落是那篇文章的
    一个 chunk ⇒ **同一篇文章会出现在多段里、`title` 相同**。拿 title 去返回文本里找，
    只要召回了同文章的另一段就会被判成"金标在"——**那是高估，而且不报错**。
    （2026-10-11 实测：title 口径给出 `25/25 ≥1 · 17/25 全`，看着像"链上证据都很好召回"。）

    片段取正文前 60 个字符：足够具体（同文章的相邻 chunk 不会共享前 60 字），
    又不会因为渲染时的空白规整而失配（`normalize_content` 只动首尾与空白形状）。
    """
    wanted = set(question_ids)
    out: dict[str, tuple[str, ...]] = {}
    for line in (Path(bench_dir) / DATA_DIR / JSONL).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row.get("id"))
        if question_id not in wanted or not row.get("answerable"):
            continue
        out[question_id] = tuple(
            sorted(
                {
                    normalize_content(str(paragraph["paragraph_text"]), where=question_id)[:60]
                    for paragraph in row["paragraphs"]
                    if paragraph.get("is_supporting")
                }
            )
        )
    return out
