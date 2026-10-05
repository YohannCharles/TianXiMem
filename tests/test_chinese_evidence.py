"""中文显式表达复用共同字段与取证；重点防止部分抽取冒充完整集合。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.grammar import chinese_quantity, extract_evidence
from tianximem.facts.query import compile_query
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.service.pipeline import SearchPipeline


def extract(text, *, answer=None):
    return extract_evidence(
        parent_memory_id="p", user_id="u", question=text, answer=answer, event_time=None
    )


def add(store, *texts, user="u", rid="r", indexed=True):
    return apply_batch(
        store,
        AddBatch(rid, user, "s", tuple(Message("user", t, None) for t in texts)),
        grounded_evidence=indexed,
    )


def search(store, counter, query, *, user="u", top_k=100):
    def retrieve(*, user_id, query, top_k):
        with store.read() as conn:
            return [
                Candidate(p.id, i) for i, p in enumerate(store.iter_pairs(conn, user_id=user_id))
            ]

    return (
        SearchPipeline(
            store=store,
            qdrant=SimpleNamespace(exists=lambda: True),
            retriever=SimpleNamespace(search=retrieve),
            checker=EvidenceChecker(),
            counter=counter,
            grounded_evidence=True,
            budget_tokens=117760,
            radius=0,
        )
        .run(user_id=user, query=query, top_k=top_k)
        .items
    )


@pytest.mark.parametrize(
    "role,org,canonical",
    [
        ("学生", "北辰学校", "student"),
        ("员工", "北辰科技", "employee"),
        ("患者", "青禾诊所", "patient"),
        ("志愿者", "海港救助站", "volunteer"),
        ("成员", "东林社团", "member"),
        ("教师", "青禾学校", "teacher"),
        ("合唱团指导员", "青禾学校", "合唱团指导员"),
    ],
)
def test_roles_share_list_count_plan_and_keep_literal_sources(store, counter, role, org, canonical):
    sources = [
        f"张岚：我是{org}的{role}。",
        f"王川: 我目前是{org}的{role}。",
        f"林悦：我仍然是{org}的{role}。",
    ]
    add(
        store,
        *sources,
        sources[0],
        f"陈明：我计划成为{org}的{role}。",
        f"周舟：我不是{org}的{role}。",
        f"赵然：示例说：“陈明是{org}的{role}。”",
    )
    add(store, sources[0].replace("张岚", "外人"), user="other", rid="other")
    queries = [
        f"{org}有哪些{role}？",
        f"{org}的{role}一共有多少人？",
        f"请列出{org}的{role}。",
        f"{org}有多少名{role}？",
    ]
    plans = [compile_query(q) for q in queries]
    assert all(p.operator == "filter" and p.patterns[0].relations == (canonical,) for p in plans)
    assert all(p.patterns == plans[0].patterns for p in plans)
    groups = [search(store, counter, q) for q in queries]
    assert all(g == groups[0] for g in groups)
    assert len(groups[0]) == 3
    assert all("Memory fact" in h.content and "Source quotation:" in h.content for h in groups[0])
    assert all("外人" not in h.content and "陈明" not in h.content for h in groups[0])
    assert len(search(store, counter, queries[0], top_k=1)) == 1
    assert not search(store, counter, queries[0], user="missing")
    with store.read() as conn:
        for row in conn.execute(
            "SELECT f.*,q.question FROM memory_facts f JOIN qa_pairs q ON f.parent_memory_id=q.id"
        ):
            assert row["source_quote"] in row["question"]


@pytest.mark.parametrize(
    "verb,role",
    [
        ("工作", "employee"),
        ("上班", "employee"),
        ("任职", "employee"),
        ("就读", "student"),
        ("念书", "student"),
        ("学习", "student"),
        ("做志愿者", "volunteer"),
        ("当志愿者", "volunteer"),
        ("任教", "teacher"),
    ],
)
def test_verb_paraphrases_bind_same_relation(verb, role):
    source = f"张岚：我目前在北辰机构{verb}。"
    facts = extract(source)
    assert len(facts) == 1 and facts[0].relation == role and facts[0].object == "北辰机构"
    questions = [
        f"请列出在北辰机构{verb}的人。",
        f"在北辰机构{verb}的人有多少？",
        f"谁在北辰机构{verb}？",
        f"有多少人在北辰机构{verb}？",
    ]
    plans = [compile_query(q) for q in questions]
    assert all(p and p.patterns == plans[0].patterns for p in plans)
    assert plans[0].patterns[0].relations == (role,)


def test_chinese_speakers_english_claims_and_named_answer_side():
    facts = extract(
        "张岚: I work for North Labs.\n王川：我是North Labs的员工。",
        answer="[assistant] 林悦：我是North Labs的员工。",
    )
    assert len(facts) == 3
    assert {f.subject for f in facts} == {"张岚", "王川", "林悦"}
    assert facts[-1].source_side == "answer"
    assert {f.relation.split()[0] for f in facts} == {"employee"}
    assert not extract(None, answer="[assistant] 我是North Labs的员工。")
    assert not extract("助手：我是North Labs的员工。")


def test_quantities_preserve_units_source_order_and_never_sum(store, counter):
    claim = "我的10升灰色箱子里有四个杯子、两只盘子和一个水壶。"
    add(
        store,
        "林悦：" + claim,
        "林悦：" + claim,
        "王川：我的10升灰色箱子里有九支笔。",
        "林悦：我计划往10升灰色箱子里再放四支笔，还没有买。",
    )
    queries = [
        "林悦的10升灰色箱子里有哪些物品？",
        "林悦的10升灰色箱子里总共有多少件物品？",
        "请列出林悦的10升灰色箱子里的物品种类。",
        "林悦的10升灰色箱子中有几件东西？",
    ]
    groups = [search(store, counter, q) for q in queries]
    assert all(g == groups[0] for g in groups) and len(groups[0]) == 3
    assert [h.content.split("包含")[1].split("，")[0] for h in groups[0]] == [
        "杯子",
        "盘子",
        "水壶",
    ]
    assert all(
        "数量为7" not in h.content and "Source quotation:" not in h.content for h in groups[0]
    )
    with store.read() as conn:
        rows = store.fetch_evidence(conn, "u", compile_query(queries[0]).patterns, limit=100)
        assert [f.get("quantity") for f in rows if f.subject == "林悦"] == [4, 2, 1, 4, 2, 1]
        assert all(f.source_quote == claim for f in rows if f.subject == "林悦")


@pytest.mark.parametrize(
    "text",
    [
        "张岚：我不是北辰学校的学生。",
        "张岚：我计划在北辰学校就读。",
        "张岚：如果我在北辰学校就读。",
        "张岚：示例说：“王川是北辰学校的学生。”",
        "张岚：她说：\n王川：我是北辰学校的学生。",
        "林悦：我的箱子里有两本笔记本和一些笔。",
        "林悦：我的箱子里有两三本笔记本。",
        "林悦：我的箱子里有一百二支笔。",
        "林悦：我的箱子里有一箱杯子。",
        "林悦：我的箱子里有10001支笔。",
        "林悦：我的箱子里有1.5支笔。",
    ],
)
def test_uncertain_planned_quoted_and_partial_claims_are_not_positive_facts(text):
    assert not extract(text)


@pytest.mark.parametrize(
    "extra",
    [
        "王川：北辰学校给我发了学生证。",  # 规则不理解的相关声明
        "张岚：我已经离开北辰学校，不再在那里就读。",  # 已选人物的新否定
        "王川：我计划成为北辰学校的学生。我目前已在北辰学校就读。",  # 混合句不能全忽略
    ],
)
def test_unexplained_or_conflicting_chinese_sources_fall_back_to_originals(store, counter, extra):
    add(store, "张岚：我是北辰学校的学生。", extra)
    hits = search(store, counter, "北辰学校有哪些学生？")
    assert hits and all("Memory fact" not in h.content for h in hits)
    assert any(extra in h.content for h in hits)


def test_partial_inventory_cannot_be_hidden_by_complete_physical_scan(store, counter):
    add(store, "林悦：我的蓝色背包里有两本笔记本。", "林悦：我的蓝色背包里有一些笔。")
    with store.read() as conn:
        assert store.has_fact_index_coverage(conn, "u", version=EVIDENCE_VERSION)
    hits = search(store, counter, "林悦的蓝色背包里有哪些物品？")
    assert hits and all("Memory fact" not in h.content for h in hits)
    assert any("一些笔" in h.content for h in hits)


def test_explicit_negative_with_modifier_excludes_outsider_but_blocks_conflicting_member(
    store, counter
):
    add(store, "张岚：我是北辰学校的学生。", "王川：我目前不是北辰学校的学生。")
    hits = search(store, counter, "北辰学校有哪些学生？")
    assert len(hits) == 1 and "Memory fact" in hits[0].content
    add(store, "张岚：我现在不是北辰学校的学生。", rid="changed")
    hits = search(store, counter, "北辰学校有哪些学生？")
    assert hits and all("Memory fact" not in h.content for h in hits)


def test_same_amount_with_different_units_cannot_be_silently_deduplicated(store, counter):
    add(store, "林悦：我的箱子里有两个水。", "林悦：我的箱子里有两瓶水。")
    hits = search(store, counter, "林悦的箱子里有哪些物品？")
    assert len(hits) == 2 and all("Memory fact" not in h.content for h in hits)


def test_overlapping_item_names_preserve_literal_item_order_and_source_position(store, counter):
    claim = "我的蓝色背包里有两本笔记本、三支笔和一把尺子。"
    add(store, "林悦：" + claim)
    hits = search(store, counter, "林悦的蓝色背包里有哪些物品？")
    assert [h.content.split("包含")[1].split("，")[0] for h in hits] == ["笔记本", "笔", "尺子"]
    with store.read() as conn:
        facts = store.fetch_evidence(
            conn, "u", compile_query("林悦的蓝色背包里有哪些物品？").patterns, limit=100
        )
    assert [f.get("item_quote") for f in facts] == ["两本笔记本", "三支笔", "一把尺子"]
    with store.transaction() as conn, pytest.raises(ValueError, match="物品定位引句"):
        store.insert_evidence(conn, [replace(facts[0], qualifiers=(("item_quote", "四支笔"),))])


def test_unnamed_pronoun_is_not_mistaken_for_a_person_name():
    assert not extract("他现在是北辰学校的学生。")
    assert not extract("我现在是北辰学校的学生。")


def test_curly_apostrophe_in_english_organization_is_not_a_quotation():
    facts = extract("Rosa Quinn: I work for O’Neil Foundation.")
    assert len(facts) == 1 and facts[0].object == "O’Neil Foundation"


def test_curly_single_quote_continuation_does_not_become_another_speaker():
    facts = extract("张岚：她说：‘\n王川：我是北辰学校的学生。’\n林悦：我是北辰学校的学生。")
    assert len(facts) == 1 and facts[0].subject == "林悦"


def test_user_alias_has_same_owner_binding_during_extraction_and_conflict_audit(store, counter):
    add(store, "用户：我的蓝色背包里有两本笔记本。", "用户：我现在不再拥有这个蓝色背包。")
    hits = search(store, counter, "我的蓝色背包里有哪些物品？")
    assert hits and all("Memory fact" not in h.content for h in hits)


@pytest.mark.parametrize(
    "query",
    ["去年北辰学校有哪些学生？", "哪些人不在北辰学校就读？", "除了张岚，北辰学校还有多少学生？"],
)
def test_unsupported_scope_is_not_silently_simplified(query):
    assert compile_query(query) is None


@pytest.mark.parametrize(
    "text,value",
    [
        ("零", 0),
        ("两", 2),
        ("十一", 11),
        ("一十二", 12),
        ("二十三", 23),
        ("两百一十三", 213),
        ("一千零一", 1001),
        ("一万", 10000),
        ("２３", 23),
        ("两三", None),
        ("十几", None),
        ("一百二", None),
        ("十十", None),
        ("10001", None),
    ],
)
def test_only_exact_chinese_integer_quantities_are_accepted(text, value):
    assert chinese_quantity(text) == value


def test_prior_empty_scan_is_backfilled_after_version_change_without_rewriting_raw(store, counter):
    add(store, "张岚：我是北辰学校的学生。", indexed=False)
    with store.transaction() as conn:
        original = store.iter_pairs(conn, user_id="u")[0]
        store.mark_fact_index_coverage(conn, original, version="memory-facts-v2")
    hits = search(store, counter, "北辰学校有哪些学生？")
    assert len(hits) == 1 and "Memory fact" in hits[0].content
    with store.read() as conn:
        assert store.iter_pairs(conn, user_id="u")[0] == original
        assert store.has_fact_index_coverage(conn, "u", version=EVIDENCE_VERSION)
