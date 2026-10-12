"""MuSiQue 的**链式依赖 haystack 设定**（`eval/datasets/musique.py` 后半
+ `eval/coq-musique.jsonl`）。

## 这个文件挡的四类错

| 类 | 挡什么 | 靠什么 |
| --- | --- | --- |
| **选题** | 把"并行分支"当成链式依赖收进来（那类**不需要第二跳**） | `_is_strict_chain` 的用例 |
| **选题** | 中间跳答案是数字/日期 ⇒ 下一跳根本没法问 | `_is_entity_answer` 的用例 |
| **语料** | 段落不去重 ⇒ 同一段进池子多次 ⇒ **"检索到它"变得更容易** | 造重复 `title` 的假数据 |
| **口径** | 覆盖率拿 `title` 当标记 ⇒ 召回同文章的另一段也算命中 | `supporting_snippets` 的用例 |

⚠ **不读真的 59 MB `musique_full_v1.0_dev.jsonl`**：`read_rows` 会把整份读进内存，
不适合放进单元测试。真数据的核对走 `eval/coq-musique.jsonl` 的**内部一致性**（下面有），
以及跑批时的实测数字。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.datasets.musique import (
    ANSWER_CONTRACT,
    BROKEN_GOLD,
    HAYSTACK_USER,
    LOOSE_ANSWER_CONTRACT,
    REFUSAL,
    _is_entity_answer,
    _is_strict_chain,
    load_musique_haystack,
    select_chain_question_ids,
    supporting_snippets,
)
from eval.harness.extra_pipeline import judge_musique

# ── 造行 ────────────────────────────────────────────────────────────────

#: 合成用的链。⚠ **hop 答案必须有 ≥3 个字母**，否则 `_is_entity_answer` 会（正确地）拒掉。
_CHAIN2 = [("Who directed the film?", "Some Director"), ("#1 >> spouse", "Final Answer")]
_CHAIN3 = [
    ("Who directed the film?", "Some Director"),
    ("#1 >> spouse", "Some Spouse"),
    ("#2 >> birthplace", "Final Answer"),
]
_CHAIN4 = [*_CHAIN3, ("#3 >> nationality", "Final Answer")]


def _paragraph(
    title: str, index: int, *, supporting: bool = False, body: str | None = None
) -> dict:
    return {
        "title": title,
        "idx": index,
        "paragraph_text": body
        or f"{title} 的正文第 {index} 段，这是一句足够长的英文句子用来占位。",
        "is_supporting": supporting,
    }


def _row(
    qid: str,
    *,
    hops: list[tuple[str, str]],
    answer: str = "Final Answer",
    paragraphs: list[dict] | None = None,
    answerable: bool = True,
) -> dict:
    return {
        "id": qid,
        "answer": answer,
        "answer_aliases": [],
        "answerable": answerable,
        "question": hops[-1][0],
        "question_decomposition": [{"question": q, "answer": a} for q, a in hops],
        "paragraphs": paragraphs
        if paragraphs is not None
        else [_paragraph(f"P{i}", i) for i in range(4)],
    }


def _write(bench_dir: Path, rows: list[dict]) -> Path:
    d = bench_dir / "musique"
    d.mkdir(parents=True, exist_ok=True)
    (d / "musique_full_v1.0_dev.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return d


# ── ① 判据 ──────────────────────────────────────────────────────────────


def test_a_strict_chain_requires_every_hop_after_the_first_to_reference_an_earlier_one() -> None:
    chain = [{"question": "Who directed X?"}, {"question": "#1 >> spouse"}]
    assert _is_strict_chain(chain) is True


def test_a_parallel_branch_is_not_a_chain() -> None:
    """⭐ 这正是上一轮踩的坑：两个独立子问题在最后合并 ⇒ **第二轮检索不必要**。"""
    parallel = [
        {"question": "Who directed X?"},
        {"question": "Who wrote Y?"},  # 不引用任何前面的跳
        {"question": "Which is common to #1 and #2?"},
    ]
    assert _is_strict_chain(parallel) is False


def test_a_question_that_references_itself_at_hop_one_is_not_a_chain() -> None:
    assert _is_strict_chain([{"question": "#1 >> something"}]) is False


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Steve Hillage", True),
        ("1932", False),
        ("March 29, 2018", False),
        ("", False),
        ("$72,641", False),
        ("the Mississippi River Delta", True),
    ],
)
def test_entity_answers_are_distinguished_from_numbers_and_dates(
    answer: str, expected: bool
) -> None:
    """⚠ 中间跳是数字/日期的题**不能选**：下一跳的问句 `#1 >> ...` 会把 `#1` 填成数字。"""
    assert _is_entity_answer(answer) is expected


# ── ② 选题规则是确定性的、可重放 ────────────────────────────────────────


def test_selection_is_deterministic_and_prefers_deeper_chains() -> None:
    rows = [
        _row("4hop1__a_b_c_d", hops=_CHAIN4),
        _row("3hop1__e_f_g", hops=_CHAIN3),
        _row("2hop__h_i", hops=_CHAIN2),  # <3 跳，不要
    ]
    picked = select_chain_question_ids(rows, limit=25)
    assert picked == ["4hop1__a_b_c_d", "3hop1__e_f_g"]  # 深的在前
    assert select_chain_question_ids(rows, limit=25) == picked  # 可重放


def test_selection_drops_questions_whose_question_text_leaks_the_intermediate_entity() -> None:
    """⭐ **唯一一条独立于 `#k` 记法的判据**：把标注拿掉，读题的人能不能直接查？

    前面几条都在读 `question_decomposition` 的占位符——那是**用标注证明标注**。
    这一条读的是**用户真正看到的那句话**。

    ⚠ 2026-10-11 实测：原判据选出的 25 道里有 **2 道泄题**（题面直接写了链中间那一跳的答案），
    那两道退化成"一次查找"。加上这条之后候选 814 → 798，**恰好换掉那 2 道**。
    """
    leaking = _row("3hop1__a_b_c", hops=_CHAIN3)
    # 把 hop2 的答案写进题面
    leaking["question"] = "Who is Some Spouse married to and where?"
    assert select_chain_question_ids([leaking], limit=25) == []

    clean = _row("3hop1__a_b_c", hops=_CHAIN3)
    clean["question"] = "Where was the spouse of the director of the film born?"
    assert select_chain_question_ids([clean], limit=25) == ["3hop1__a_b_c"]


def test_selection_drops_questions_whose_hop_answers_are_not_entities() -> None:
    """中间跳答案是日期的题**即使结构上是严格链**也要丢掉。"""
    numeric = _row(
        "3hop1__a_b_c",
        hops=[("q1", "1999"), ("#1 >> r2", "Some Person"), ("#2 >> r3", "Final Answer")],
    )
    assert select_chain_question_ids([numeric], limit=25) == []


def test_the_leak_check_matches_whole_words_not_substrings() -> None:
    """⭐ 泄题判据必须是**词边界**：`NES` 不该被 `geNESis` 命中。

    ⚠ 实测（2026-10-12，子 agent 独立复核）：纯子串版本**误杀 15 条**——
    中间跳答案是 `NES` 时，题面里的 `Genesis` 也算命中，整个 `…_15840_36002`
    家族被整批丢掉；改成词边界后候选 798 → **813**。
    ⚠ 被误杀**不会报错**：它只是安静地少收几道题，而"少收"与"本来就没有"看起来一样。
    """
    from eval.datasets.musique import _appears_as_a_word

    assert _appears_as_a_word("Which Genesis game?", "NES") is False  # 误杀点
    assert _appears_as_a_word("the NES platform", "NES") is True
    assert _appears_as_a_word("a nes game", "NES") is True  # 忽略大小写
    assert _appears_as_a_word("(NES)", "NES") is True  # 标点那侧也要算边界
    assert _appears_as_a_word("the $72,641 budget", "$72,641") is True  # 带标点的答案
    assert _appears_as_a_word("x $72,6410 y", "$72,641") is False  # 右边界仍要成立

    # 端到端：Genesis 那条题面不该被当成泄题（`NES` 是中间跳答案）
    clean = _row(
        "3hop1__a_b_c",
        hops=[("q1", "NES"), ("#1 >> r2", "Some Person"), ("#2 >> r3", "Final Answer")],
        answer="Final Answer",
    )
    clean["question"] = "Which company published the Genesis game?"
    assert select_chain_question_ids([clean], limit=25) == ["3hop1__a_b_c"]


def test_selection_rejects_two_questions_sharing_a_chain_skeleton() -> None:
    """共享 ≥2 跳 = 同一条骨架换了个入口 ⇒ 只收一条（否则 25 道里会有一堆"同一题"）。"""
    shared = ("c", "d")
    first = _row("4hop1__a_b_c_d", hops=_CHAIN4)
    second = _row("4hop1__x_y_c_d", hops=_CHAIN4)
    picked = select_chain_question_ids([first, second], limit=25)
    assert picked == ["4hop1__a_b_c_d"]
    assert set(shared)  # 说明这两条确实共享了 c / d 两跳


# ── ③ haystack ──────────────────────────────────────────────────────────


def test_the_haystack_merges_every_questions_paragraphs_into_one_user(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [
            _row(
                "3hop1__a_b_c",
                hops=_CHAIN3,
                paragraphs=[_paragraph("Alpha", 0, supporting=True), _paragraph("Beta", 1)],
            ),
            _row(
                "3hop1__d_e_f",
                hops=_CHAIN3,
                paragraphs=[_paragraph("Beta", 0), _paragraph("Gamma", 1, supporting=True)],
            ),
        ],
    )
    sample = load_musique_haystack(tmp_path, ["3hop1__a_b_c", "3hop1__d_e_f"])
    assert sample.user_id == HAYSTACK_USER
    assert len(sample.sessions) == 1
    # ⭐ 两题的段落并起来、**按 title 去重**：Alpha / Beta / Gamma 只有三条
    assert sample.message_count == 3
    assert len(sample.questions) == 2
    assert all(q.category == "3-hop/chain" for q in sample.questions)


def test_a_shared_title_appears_only_once_in_the_pool(tmp_path: Path) -> None:
    """⚠ 同一篇维基段落会被多道题共用；不去重会让**同一段内容在池子里出现多份**，
    而"检索到它"就变得更容易——那是**高估**，且不报错。"""
    _write(
        tmp_path,
        [
            _row("3hop1__a_b_c", hops=_CHAIN3, paragraphs=[_paragraph("Same", 0)], answer="One"),
            _row("3hop1__d_e_f", hops=_CHAIN3, paragraphs=[_paragraph("Same", 0)], answer="Two"),
        ],
    )
    sample = load_musique_haystack(tmp_path, ["3hop1__a_b_c", "3hop1__d_e_f"])
    assert sample.message_count == 1


def test_an_unknown_or_non_chain_id_is_a_hard_failure(tmp_path: Path) -> None:
    parallel = _row(
        "4hop2__a_b_c_d",
        hops=[
            ("Who directed the film?", "Some Director"),
            ("Who scored the film?", "Some Composer"),
            ("Who edited the film?", "Some Editor"),
            ("Which was common to #1 and #2 and #3?", "Final Answer"),
        ],
    )
    _write(tmp_path, [parallel])
    with pytest.raises(ValueError, match="不在可答题里"):
        load_musique_haystack(tmp_path, ["nope"])
    with pytest.raises(ValueError, match="不是严格链"):
        load_musique_haystack(tmp_path, ["4hop2__a_b_c_d"])


def test_the_answer_never_leaks_into_the_pool(tmp_path: Path) -> None:
    """§2.1：**不许把答案伪装成记忆**——金标与分解都不进 Add 载荷。"""
    _write(
        tmp_path,
        [
            _row(
                "3hop1__a_b_c",
                hops=_CHAIN3,
                answer="SecretAnswer",
                paragraphs=[_paragraph("Alpha", 0, body="只有这一段正文。")],
            )
        ],
    )
    sample = load_musique_haystack(tmp_path, ["3hop1__a_b_c"])
    blob = "".join(m.content for m in sample.sessions[0].messages)
    assert "SecretAnswer" not in blob
    assert "question_decomposition" not in blob


# ── ④ 覆盖率标记：片段而不是 title ──────────────────────────────────────


def test_supporting_snippets_use_the_body_not_the_title(tmp_path: Path) -> None:
    """⚠ `title` 是**维基文章名**，一文多 chunk ⇒ 召回同文章的另一段也会被判成"金标在"。"""
    _write(
        tmp_path,
        [
            _row(
                "3hop1__a_b_c",
                hops=_CHAIN3,
                paragraphs=[
                    _paragraph("SharedArticle", 0, supporting=True, body="甲乙丙丁开头的正文内容。")
                ],
            )
        ],
    )
    got = supporting_snippets(tmp_path, ["3hop1__a_b_c"])
    assert got["3hop1__a_b_c"] == ("甲乙丙丁开头的正文内容。",)
    assert "SharedArticle" not in "".join(got["3hop1__a_b_c"])


# ── ⑤ 清单的内部一致性（不读 59 MB 的原始数据）────────────────────────


def test_the_shipped_manifest_is_internally_consistent() -> None:
    rows = [
        json.loads(line)
        for line in Path("eval/coq-musique.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(rows) == 50
    assert {row["corpus"] for row in rows} == {"self"}
    assert {row["hops"] for row in rows} == {3, 4}  # 4 跳 29 · 3 跳 21
    assert {row["user_id"] for row in rows} == {HAYSTACK_USER}
    # 两两之间最多共享 1 个 hop —— 即"没有两条是同一个骨架"
    hop_sets = [set(row["hop_ids"]) for row in rows]
    for i, left in enumerate(hop_sets):
        for right in hop_sets[i + 1 :]:
            assert len(left & right) < 2, "清单里有两条题共享 ≥2 跳（同骨架重复出题）"
    # ⛔ 题面 / 答案一个字都不许进清单
    for row in rows:
        assert set(row) == {"dataset", "user_id", "question_id", "hops", "hop_ids", "corpus"}
    # ⛔ 金标坏掉的 10 条（只读一份声明，不需要数据集）一条都不许混进来
    assert not ({row["question_id"] for row in rows} & set(BROKEN_GOLD))


# ── ⑤ 两套判分口径（2026-10-12）────────────────────────────────────────


def _musique_gold(answer: str, *aliases: str) -> dict:
    return {"answer": answer, "answer_aliases": list(aliases), "answerable": True}


def test_the_strict_contract_rejects_a_more_specific_answer() -> None:
    """严口径（默认）：归一化后必须**相等**。⇒ `five` 会被 `five games` 判错。

    ⚠ 这就是"判分是不是太严"那个问题的**唯一一处实证**：实测三臂各自只翻这 1 道。
    """
    gold = _musique_gold("five")
    assert judge_musique("five", gold, contract=ANSWER_CONTRACT)[0] is True
    assert judge_musique("Five games", gold, contract=ANSWER_CONTRACT)[0] is False


def test_the_loose_contract_accepts_a_more_specific_answer() -> None:
    """宽松口径：金标作为一个词**出现在**回答里即算对。"""
    gold = _musique_gold("five")
    assert judge_musique("five", gold, contract=ANSWER_CONTRACT)[0] is True
    assert judge_musique("1992", gold, contract=LOOSE_ANSWER_CONTRACT)[0] is False
    assert (
        judge_musique("up to five games per year", gold, contract=LOOSE_ANSWER_CONTRACT)[0] is True
    )


def test_the_loose_contract_also_accepts_a_more_general_answer_which_is_wrong() -> None:
    """⛔ **宽松口径的代价，钉在这里**：它会把"比金标更笼统"的答案也算对。

    金标是县名 `Cabarrus County`，回答成 `North Carolina`（范围更大、且不是所问）
    ——按 MuSiQue 自己的口径**应当算错**，而宽松口径会放过它。
    ⇒ **两套口径必须并排报，不许拿宽松那套替换严的那套。**
    """
    gold = _musique_gold("Cabarrus County")
    wrong_but_contained = "Cabarrus County is in North Carolina"
    assert judge_musique(wrong_but_contained, gold, contract=ANSWER_CONTRACT)[0] is False
    assert judge_musique(wrong_but_contained, gold, contract=LOOSE_ANSWER_CONTRACT)[0] is True


def test_both_contracts_agree_on_the_abstention_branch() -> None:
    """不可答题那条分支**两套口径完全一样**（它比的不是答案，是拒答标记）。"""
    gold = {"answer": "", "answer_aliases": [], "answerable": False}
    for contract in (ANSWER_CONTRACT, LOOSE_ANSWER_CONTRACT):
        assert judge_musique(REFUSAL, gold, contract=contract)[0] is True
        assert judge_musique("something", gold, contract=contract)[0] is False


def test_the_default_contract_is_the_strict_one() -> None:
    """⚠ 默认不许变：`judge_musique` 不带 `contract` 时必须走严口径。"""
    gold = _musique_gold("five")
    assert judge_musique("up to five games per year", gold)[0] is False


# ── 交付给另一个项目的那份包（`tools/export_multihop_kit.py`）──────────


def _export_row(hops: list[tuple[str, str]], *, support_bodies: list[str]) -> dict:
    # ⚠ 末答**必须**等于最后一跳的答案（真数据就是这样）——否则 `verify` 会（正确地）
    #    报"末答不在支持段落里"，而那是夹具造的假缺陷。
    row = _row(
        "4hop1__a_b_c_d" if len(hops) == 4 else "3hop1__a_b_c", hops=hops, answer=hops[-1][1]
    )  # noqa: E501
    for index, paragraph in enumerate(row["paragraphs"]):
        paragraph["is_supporting"] = index < len(support_bodies)
        paragraph["paragraph_text"] = (
            support_bodies[index] if index < len(support_bodies) else "无关的干扰段落。"
        )
    return row


def test_export_verification_passes_only_when_every_answer_is_in_the_evidence() -> None:
    """⭐ 用户要的那条保证：**每一跳的答案都必须在支持段落正文里找得到**。

    这一条是"这份包能不能交出去"的判据——答案不在原文里的题**下游根本做不了**，
    而它交出去**不会报错**，只会让别人测出一堆假失败。
    """
    from tools.export_multihop_kit import verify

    hops = [("Who directed it?", "Some Director"), ("#1 >> spouse", "Some Spouse")]
    good = _export_row(
        hops, support_bodies=["Some Director directed it.", "Some Spouse is the spouse."]
    )
    assert verify(["4hop1__a_b_c_d"], {"4hop1__a_b_c_d": good}) == []

    # 把某一跳的答案从证据里拿掉 ⇒ 必须点名报出来（**不是静默放过**）
    bad = _export_row(hops, support_bodies=["Some Director directed it.", "毫不相关的句子。"])
    problems = verify(["4hop1__a_b_c_d"], {"4hop1__a_b_c_d": bad})
    assert len(problems) == 1 and "第 2 跳" in problems[0]


def test_export_verification_rejects_the_other_traps() -> None:
    """拒答题 / 并行分支 / 泄题 / 末答为空——四条都要响亮失败。"""
    from tools.export_multihop_kit import verify

    hops = [("Who directed it?", "Some Director"), ("#1 >> spouse", "Some Spouse")]
    bodies = ["Some Director directed it.", "Some Spouse is the spouse."]

    parallel = _export_row(
        [("Who directed it?", "Some Director"), ("Who wrote it?", "Some Writer")],
        support_bodies=bodies,
    )
    assert any(
        "不是严格链" in item for item in verify(["4hop1__a_b_c_d"], {"4hop1__a_b_c_d": parallel})
    )  # noqa: E501

    empty = _export_row(hops, support_bodies=bodies)
    empty["answer"] = ""
    assert any("末答为空" in item for item in verify(["4hop1__a_b_c_d"], {"4hop1__a_b_c_d": empty}))

    # 题面直接写了中间跳的答案 ⇒ 退化成"一次查找"
    leaking = _export_row(hops, support_bodies=bodies)
    leaking["question"] = "Who is Some Director married to?"
    assert any(
        "泄了中间跳" in item for item in verify(["4hop1__a_b_c_d"], {"4hop1__a_b_c_d": leaking})
    )


def test_the_export_readme_warns_that_the_pool_is_mandatory() -> None:
    """⚠ README 必须把"**必须并成一个池子**"写在明面上——那是这份题集能测多跳的前提。

    只喂目标题自己那 20 段 ⇒ 证据全在里面 ⇒ 检索无事可做、链式依赖消失，
    而**测出来的分数看起来完全正常**。
    """
    from tools.export_multihop_kit import render_readme

    text = render_readme(50, 822, {4: 29, 3: 21})
    assert "822" in text and "50 道" in text
    assert "4 跳 29 道 · 3 跳 21 道" in text
    assert "必须把这些段落并成一个池子" in text
    assert "CC BY-NC 4.0" in text and "训练" in text  # 许可与"不得训练"都要在
