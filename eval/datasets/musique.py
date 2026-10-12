"""MuSiQue-Full dev：每道题的候选段落组成一个独立用户的记忆语料。

Full 的可答/不可答变体共用 id，user_id/qid 必须带变体标记。只把 title 与
paragraph_text 送进 Add；answer、is_supporting 与 question_decomposition 均不注入。
spread 按跳数和可答性分层。只测本地答案别名精确匹配/拒答，不复现上游支持段落 F1
或成组 answer-sufficiency 指标。原生文本包装与采集不同；候选段落顺序保留源数据顺序。
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
USER_PREFIX = "musique-"
SHAPE_NOTE = (
    f"MuSiQue-Full dev（{ANSWER_CONTRACT}）：每题一个独立 user，"
    "语料只含候选段落 title/paragraph_text。"
    "可答/不可答变体用不同 user_id/qid；spread 按跳数和可答性分层。"
    "本地评分为答案别名归一化后精确匹配与拒答标记精确匹配，不是上游答案/支持段落 F1"
    "或成组 sufficiency 指标；候选段落保留源数据顺序，原生包装与采集不同。"
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
# 按 `title` 去重（同一篇维基段落会被多道题共用）⇒ 25 道 4 跳题的 500 个槽位得到 **427 段**，
# 与 benchmark 里 LoCoMo 的 357 段**同一量级**，可比。
# 目标题自己的支持段落就在池子里，被其余 24 道题的段落埋着。
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


def _does_not_leak_the_intermediate_entity(row: dict) -> bool:
    """⭐ **用户看到的那句问题里，不许出现任何中间跳的答案。**

    ⚠ 这是"是不是真链"唯一一条**独立于 `#k` 记法**的判据。前面几条都用
    `question_decomposition` 的 `#k` 占位符来判——而那个占位符**本来就是记法**，
    等于用标注证明标注。只有这一条问的是：**把标注全拿掉，读题的人能不能直接查?**

    实测（2026-10-11）：原判据选出的 25 道里有 **2 道泄题**——
    `4hop1__21282_…` 的题面直接写了 `Angela Merkel`、`4hop1__40316_…` 直接写了 `NES`，
    而它们正是链中间那一跳的答案 ⇒ 那两道退化成"一次查找"。
    加上这条之后候选从 814 降到 **798**，而选出的 25 道里**恰好换掉那 2 道**。
    """
    question = str(row["question"]).lower()
    return not any(
        str(hop["answer"]).lower() in question
        for hop in (row.get("question_decomposition") or [])[:-1]
    )


def select_chain_question_ids(rows: list[dict], *, limit: int = 25) -> list[str]:
    """按**确定性规则**挑出链式依赖题（`eval/coq-musique.jsonl` 头部写的生成规则）：

    1. `answerable == True`；
    2. **严格链**（见 `_is_strict_chain`）；
    3. **≥3 跳**，且 hop1..hop(n-1) 的答案都是**实体**；
    4. ⭐ **题面不泄中间实体**（见 `_does_not_leak_the_intermediate_entity`）；
    5. 排序 `(-跳数, id)` 后**贪心**：新题与已选题的 hop 集合**共享 <2 个元素**才收
       —— 共享 ≥2 跳 = 同一条骨架换了个入口，收进来就是重复出题。

    ⚠ 规则写**在代码里**而不是只写在清单的注释里：清单是可被替换的产物，
    而规则必须能重放（同一个 `limit` 永远给出同一批 id）。
    """
    candidates: list[tuple[int, str, frozenset[str]]] = []
    for row in rows:
        if not row.get("answerable"):
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

    ⚠ 段落按 `title` 去重、**保留首次出现的顺序**（确定性）：同一篇维基段落会被多道题共用，
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

    bodies: dict[str, str] = {}  # title -> 消息正文（首次出现的赢）
    for question_id in wanted:
        row = by_id[question_id]
        hops = row.get("question_decomposition") or []
        if not _is_strict_chain(hops):
            raise ValueError(f"MuSiQue {question_id}：不是严格链——清单不该收它")
        for paragraph in row["paragraphs"]:
            title = str(paragraph["title"])
            if title in bodies:
                continue
            bodies[title] = normalize_content(
                f"Title: {title}\n"
                + normalize_content(
                    paragraph["paragraph_text"], where=f"MuSiQue {question_id} paragraph body"
                ),
                where=f"MuSiQue haystack paragraph {title!r}",
            )
    messages = tuple(Message(role="user", content=body) for body in bodies.values())

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
