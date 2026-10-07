"""原专用路径的行为回归，按共同结构/算子组织；不调用外部模型。"""

from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tianximem.common.render import render_pair
from tianximem.common.tokens import O200kCounter
from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import compile_query
from tianximem.pairing import AddBatch, Message, PayloadMismatchError, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore


def extract(text, *, answer=None):
    return extract_evidence(
        parent_memory_id="p", user_id="u", question=text, answer=answer, event_time=None
    )


def add(store, *texts, rid="opaque", user="u", timestamp=None, role="user"):
    return apply_batch(
        store,
        AddBatch(rid, user, "s", tuple(Message(role, t, timestamp) for t in texts)),
        grounded_evidence=True,
    )


def run(store, query, *, user="u", limit=12, top_k=100, budget=117760, metrics=None, backfill=1024):
    def retrieve(*, user_id, query, top_k):
        with store.read() as conn:
            ids = [p.id for p in store.iter_pairs(conn) if p.user_id == user_id]
        return [Candidate(pid, i) for i, pid in enumerate(ids)]

    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=retrieve),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=budget,
        grounded_evidence=True,
        evidence_limit=limit,
        fact_backfill_limit=backfill,
        metrics=metrics,
        radius=0,
    )
    return pipeline.run(user_id=user, query=query, top_k=top_k).items


def mail(claim="An update.", company="Example Labs", actor="Alex Reader"):
    return (
        f"document: From: {actor} <alex@example.com>\nDate: 2024-03-30\nSubject: Update\n\n"
        f"{claim}\n\n{actor}\nSoftware Engineer\n{company}"
    )


def test_dated_replacements_choose_latest_even_when_add_arrival_is_reversed(store):
    add(store, "Corpus: Book — creator — Original", rid="base", timestamp=100)
    add(
        store,
        "Corpus: Book was created by Latest (this replaces the earlier value)",
        rid="arrived-first",
        timestamp=300,
    )
    add(
        store,
        "Corpus: Book was created by Earlier (this replaces the earlier value)",
        rid="arrived-last",
        timestamp=200,
    )
    hits = run(store, "Who is the creator of Book?")
    assert len(hits) == 1
    assert "Latest" in hits[0].content
    historical = run(store, "Who was the creator of Book before 1970?")
    assert any("Original" in hit.content for hit in historical)
    assert compile_query("Who was the previous creator of Book?") is None


def test_repeated_replacement_value_keeps_its_later_observation(store):
    for rid, value, timestamp in (
        ("a", "Writer A", 100),
        ("b", "Writer B", 200),
        ("c", "Writer A", 300),
    ):
        add(
            store,
            f"Corpus: Book was created by {value} (this replaces the earlier value)",
            rid=rid,
            timestamp=timestamp,
        )
    hits = run(store, "Who is the creator of Book?")
    assert len(hits) == 1
    assert "Writer A" in hits[0].content
    assert "Writer B" not in hits[0].content


@pytest.mark.parametrize("timestamps", [(None, None), (100, 100), (100, None)])
def test_unknown_or_tied_replacement_time_preserves_conflicting_sources(store, timestamps):
    for rid, value, timestamp in zip(("a", "b"), ("Writer A", "Writer B"), timestamps, strict=True):
        add(
            store,
            f"Corpus: Book was created by {value} (this replaces the earlier value)",
            rid=rid,
            timestamp=timestamp,
        )
    hits = run(store, "Who is the creator of Book?")
    assert len(hits) == 2
    assert any("Writer A" in h.content for h in hits)
    assert any("Writer B" in h.content for h in hits)


def test_later_ordinary_statement_is_not_hidden_by_an_older_replacement(store):
    add(
        store,
        "Corpus: Book was created by Writer A (this replaces the earlier value)",
        rid="a",
        timestamp=100,
    )
    add(store, "Corpus: Book — creator — Writer B", rid="b", timestamp=200)
    assert len(run(store, "Who is the creator of Book?")) == 2


def test_grounded_diagnostics_distinguish_selection_conflict_and_truncation(store):
    observations = []
    metrics = SimpleNamespace(record=observations.append)
    add(store, "Alex: I work for Example Labs.", rid="a")
    add(store, "Bea: I work for Example Labs.", rid="b")
    hits = run(store, "Who works for Example Labs?", top_k=1, metrics=metrics)
    assert len(hits) == 1
    observation = observations[-1]
    assert observation.grounded_status == "selected"
    assert observation.grounded_operator == "filter"
    assert observation.selected_facts == 2
    assert observation.selected_sources == 2
    assert observation.truncated_by_top_k
    assert observation.returned_segments == 1
    assert observation.rerank_disabled == 0

    add(store, "User: My tank has 1 fish.", rid="c")
    add(store, "User: My tank has 2 fish.", rid="d")
    run(store, "How many fish do I have in total in my tanks?", metrics=metrics)
    assert observations[-1].grounded_status == "quantity_conflict"
    assert observations[-1].rerank_disabled == 1


def test_incomplete_derived_index_records_a_distinct_whole_search_fallback(store):
    apply_batch(
        store, AddBatch("a", "u", "s", (Message("user", "Corpus: Book — creator — Writer"),))
    )
    apply_batch(
        store, AddBatch("b", "u", "s", (Message("user", "Corpus: Writer — place of birth — City"),))
    )
    observations = []
    hits = run(
        store,
        "Where was the creator of Book born?",
        backfill=1,
        metrics=SimpleNamespace(record=observations.append),
    )
    assert len(hits) == 2
    assert observations[-1].grounded_status == "index_incomplete"
    assert observations[-1].selected_facts == 0


@pytest.mark.parametrize(
    "text",
    [
        "Alex: I joined a church and a gym.",
        "Alex: I hope to make friends from the gym.",
        "Alex: I will volunteer at the shelter next month.",
        "Alex: I am not friends with one of my fellow volunteers.",
        'Alex: Bea said, "I have my church friends."',
        "Alex: Bea had dinner with some friends from the gym.",
        "Alex: Bea volunteers at a homeless shelter.",
        "Alex: Bea said:\nBea: I had dinner with my church friends.",
        'Alex: Bea said, "My home country, Sweden."',
        "Alex: If I've known these friends for 3 years, since I moved from my home country.",
        "Alex: I never moved from my home country.",
        "Alex: Bea has known these friends for 3 years, since Bea moved from her home country.",
        "Alex: My store will be open next month.",
        "Alex: If my store is open, I will sell shirts.",
        "Alex: Bea said my store is open.",
        'Alex: Bea said, "My store is open."',
        "Alex: My friend's store is open.",
        "Alex: I will get accepted for a design internship.",
        "Alex: I never just got accepted for a design internship.",
        "Alex: Bea just got accepted for a design internship.",
        "Alex: I plan to go to a fair.",
        "Alex: Bea went to a fair.",
        'Alex: Bea said, "I went to a fair."',
        "Alex: I never went to a fair.",
    ],
)
def test_quoted_planned_negated_and_third_person_sentences_are_not_own_positive_facts(text):
    assert not extract(text)


@pytest.mark.parametrize("intro", ["Bea wrote:\n", 'I read this quote: "\n', "Bea said: “\n"])
def test_named_quote_continuation_cannot_rebind_the_speaker(intro):
    assert not extract("Alex: " + intro + "Bea: My online clothing store is open!")


def test_one_source_can_bind_multiple_speakers_and_both_conversation_sides():
    facts = extract(
        "Alex: I'm still planning.\nBea: My online clothing store is open!",
        answer="[assistant] Cara: I just got accepted for a research internship!",
    )
    assert [(f.subject, f.relation) for f in facts] == [
        ("Bea", "is open"),
        ("Cara", "accepted for"),
    ]
    assert facts[1].source_side == "answer"
    assert extract(None, answer="[assistant] Bea: My store is open!")[0].subject == "Bea"
    assert not extract(None, answer="[assistant] Assistant: My store is open!")
    assert all(f.source_date == "" and "Opened on" not in f.content for f in facts)


def test_alias_matching_metadata_does_not_create_duplicate_logical_facts(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(store, "Alex: I volunteer at North Shelter.", "Bea: I volunteer at North Shelter.")
    hits = run(store, "How many volunteers are there at North Shelter?")
    assert len(hits) == 2
    assert all("Memory fact" in hit.content for hit in hits)


def test_direct_own_claim_is_preferred_over_shared_activity_for_the_same_fact(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(store, "Alex: Some friends from the library and I went camping.")
    add(store, "Alex: I ate dinner with my library friends.", rid="direct")
    hits = run(store, "Where has Alex made friends?")
    # 冠词别名之外这两句指向同一地点，来源仍逐条保存；使用明确归属的原话。
    assert any("Alex has library friends" in hit.content for hit in hits)
    assert len(hits) == 1
    assert not any("made friends at" in hit.content for hit in hits)


def test_signature_and_report_are_separate_claims_with_source_dates():
    facts = extract(mail("I've commenced my role at Example Labs, and I am settling in."))
    assert {f.relation for f in facts} == {"document title", "employee of", "role commencement"}
    assert all(
        f.source_date == "2024-03-30" and "hire date" not in f.content.lower() for f in facts
    )
    assert any(
        f.relation == "orientation schedule"
        for f in extract(
            mail(
                "Going through Example Labs's orientation schedule, I noticed a missing checklist."
            )
        )
    )
    assert not any(
        f.relation == "employee of" for f in extract(mail(company="No company in signature"))
    )
    quoted = (
        mail(company="No company in signature")
        + "\nOn Monday, Bea wrote:\n> Bea\n> Engineer\n> Example Labs"
    )
    assert not any(f.relation == "employee of" for f in extract(quoted))
    assert not any(
        f.relation == "role commencement"
        for f in extract(mail('Bea said: "I\'ve commenced my role at Example Labs."'))
    )


def test_reference_resolution_returns_separate_claims_and_rejects_missing_or_conflicting_support(
    tmp_path,
):
    store = SqliteStore.open(tmp_path / "db")
    peer = "Alex: I'm now friends with one of my fellow volunteers!"
    place = "Alex: I donated my old car to a homeless shelter I volunteer at yesterday."
    church = "Alex: I had dinner with my church friends."
    add(store, peer, church)
    assert all("Q:" in f.content for f in run(store, "Where has Alex made friends?"))
    add(store, place, rid="place")
    hits = run(store, "Where has Alex made friends?")
    assert len(hits) == 3
    assert any("volunteers at a homeless shelter" in f.content for f in hits)
    assert all("made friends at" not in f.content for f in hits)
    add(
        store,
        "Alex: I gave a talk at the homeless shelter I volunteer at.",
        rid="duplicate place",
    )
    assert len(run(store, "Where has Alex made friends?")) == 3
    moved = "Alex: I've known these friends for 3 years, since I moved from my home country."
    add(store, moved, "Alex: A gift from my home country, Sweden.", rid="origin")
    hits = run(store, "Where did Alex move from 3 years ago?")
    assert len(hits) == 2 and all("Memory fact" in h.content for h in hits)
    assert any("Sweden" in h.content for h in hits)
    assert all("moved from Sweden" not in h.content for h in hits)
    add(store, "Alex: A gift from my home country, Norway.", rid="conflict")
    assert all("Q:" in f.content for f in run(store, "Where did Alex move from 3 years ago?"))


def test_completed_actions_and_status_reports_preserve_distinct_sources_without_chosen_event_dates(
    tmp_path,
):
    store = SqliteStore.open(tmp_path / "db")
    for rid, stamp in (("old", 1000), ("new", 2000)):
        add(
            store,
            "Alex: Yesterday I went to a fair to show my shop.",
            "Bea: My store is open!",
            rid=rid,
            timestamp=stamp,
        )
    assert len(run(store, "Which events has Alex participated in to promote his business?")) == 2
    assert len(run(store, "When did Bea open her store?")) == 2


def test_capacity_is_container_identity_and_quantities_do_not_include_assistant_advice(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(
        store,
        "User: My new 20-gallon tank, which has 10 neon tetras, "
        "5 gouramis, and a small pleco catfish.",
        "User: My old 10-gallon tank, which has my betta fish, Bubbles.",
    )
    add(store, "Assistant: You should add 100 fish.", rid="advice", role="assistant")
    hits = run(store, "How many fish do I have in total in my tanks?")
    assert len(hits) == 4 and all("contains " in h.content for h in hits)
    assert sum(int(h.content.split("contains ")[1].split()[0]) for h in hits) == 17
    assert all("Quantity:" not in h.content and "Source quotation:" not in h.content for h in hits)
    assert len(run(store, "How many fish do I have in total in my tanks?", top_k=1)) == 1
    assert not run(store, "How many fish do I have in total in my tanks?", budget=1)
    assert not run(store, "How many fish do I have in total in my tanks?", user="other")


def test_quantity_changes_use_latest_source_time_even_when_the_latest_amount_repeats(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    for rid, stamp, quantity in (("late", 3000, 3), ("early", 1000, 3), ("middle", 2000, 5)):
        add(store, f"Alex: My case contains {quantity} pencils.", rid=rid, timestamp=stamp)
    hits = run(store, "How many items are in Alex's case in total?")
    assert len(hits) == 1 and "contains 3 pencils" in hits[0].content


def test_graph_selects_only_literal_sources_and_retains_contradictory_explicit_replacements(
    tmp_path,
):
    store = SqliteStore.open(tmp_path / "db")
    add(
        store,
        "Corpus: Book — creator — Old Writer",
        "Corpus: Book was created by New Writer (this replaces the earlier value)",
        "Corpus: New Writer was born in the city of New City (this replaces the earlier value)",
    )
    query = "In what city was the creator of Book born?"
    hits = run(store, query)
    assert len(hits) == 2 and all("Old Writer" not in h.content for h in hits)
    with store.read() as conn:
        originals = {render_pair(p) for p in store.iter_pairs(conn)}
    assert all(h.content in originals for h in hits)
    assert not any("Book was born" in h.content for h in hits)
    add(
        store,
        "Corpus: Book was created by Other Writer (this replaces the earlier value)",
        rid="conflict",
    )
    hits = run(store, query)
    assert len(hits) == 3 and any("Other Writer" in h.content for h in hits)
    assert all("Q:" in h.content for h in run(store, query, limit=1))


def test_unparsed_explicit_edit_remains_original_text_without_an_invented_edge(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(
        store,
        "Corpus: Alex — country of citizenship — State",
        "Corpus: State — head of government — Old Leader",
        "Corpus: The name of the current head of the State government is New Leader "
        "(this replaces the earlier value)",
    )
    hits = run(store, "Who is the head of government of the country of citizenship of Alex?")
    assert (
        len(hits) == 3
        and any("Old Leader" in h.content for h in hits)
        and any("New Leader" in h.content for h in hits)
    )


def test_punctuation_collisions_never_join_different_entities(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(store, "Corpus: C++ — creator — Alex", "Corpus: C# — creator — Bea")
    hits = run(store, "Who is the creator of C++?")
    assert len(hits) == 2  # 回退两条原文，而不是用有损规范化误连一个实体。


@pytest.mark.parametrize(
    "text",
    [
        "Source: Alice works for Org from Jan, 0 to Jan, 1995.",
        "Source: Alice works for Org from Jan, 1995 to Jan, 1990.",
        "Source: Alice works for Org from Xxx, 1990 to Jan, 1995.",
        'Source: "Alice" works for Org from Jan, 1990 to Jan, 1995.',
        "Source: Alice works for not Org from Jan, 1990 to Jan, 1995.",
        "Source: Alice works for Org from Jan, 1990 to Jan, 1995. Extra context.",
        "Source: Alice works for Org from Jan, 1990\nto Jan, 1995.",
    ],
)
def test_intervals_require_explicit_valid_single_line_boundaries(text):
    assert not any(f.get("start_month") is not None for f in extract(text))


def test_interval_at_before_after_and_ties_are_generic_operations(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    sources = (
        "Source: Alice plays for Old from Jan, 1990 to Jan, 1995.",
        "Source: Alice plays for Middle from Jan, 1995 to Jan, 2000.",
        "Source: Alice plays for New from Jan, 2000 to Jan, 2005.",
        "Source: Alice plays for Side from Jan, 2000 to Jan, 2003.",
    )
    add(store, *sources, sources[1])
    assert len(run(store, "Which team did Alice play for in Jun, 1997?")) == 1
    assert len(run(store, "Which team did Alice play for in Jan, 1995?")) == 2
    before = run(store, "Which team did Alice play for before Middle?")
    after = run(store, "Which team did Alice play for after Middle?")
    assert len(before) == 2 and all("New" not in h.content for h in before)
    assert len(after) == 3 and any("Side" in h.content for h in after)
    add(store, "Source: Alice plays for Middle from Jan, 2010 to Jan, 2015.", rid="ambiguous")
    assert len(run(store, "Which team did Alice play for before Middle?")) == 6


def test_inverse_interval_keeps_literal_entity_punctuation(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    add(
        store,
        "Source: Alice is the head coach of C++ Team from Jan, 1990 to Jan, 1995.",
        "Source: Bea is the head coach of C# Team from Jan, 1990 to Jan, 1995.",
    )
    hits = run(store, "Who was the head coach of the team C++ Team in Jan, 1993?")
    assert len(hits) == 1 and "Alice" in hits[0].content and "Bea" not in hits[0].content


def test_title_filter_is_a_metadata_condition_and_returns_original_source(tmp_path):
    store = SqliteStore.open(tmp_path / "db")
    source = (
        "document: Subject: Device validation\nDate: 2024-01-20\n"
        "CALENDAR INVITE\nScheduled for: 2024-02-01"
    )
    add(store, source, "document: Subject: Device validation notes\nDate: 2024-02-01")
    add(store, source, rid="other", user="other")
    query = "How many times did the Device validation meeting occur in February of 2024?"
    hits = run(store, query)
    assert len(hits) == 1 and hits[0].content == "Q: " + source
    alternate = "What is the total number of times the Device validation meeting occurred?"
    assert run(store, alternate)[0].content == hits[0].content
    assert len(run(store, alternate)) == 1
    assert len(run(store, "How many times did the Missing title meeting occur?")) == 2
    assert (
        compile_query(
            "Who was absent from at least one occurrence of the Device validation meeting?"
        )
        is None
    )


def test_fresh_schema_contains_only_common_derived_tables_and_freezes_first_representation(
    tmp_path, monkeypatch
):
    store = SqliteStore.open(tmp_path / "db")
    original = AddBatch("r", "u", "s", (Message("user", "Alex: I am a patient at Clinic."),))
    apply_batch(store, original, grounded_evidence=True)
    with store.read() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"qa_pairs", "applied_batches", "memory_facts", "evidence_coverage"}
        fact = store.fetch_evidence(conn, "u", (), limit=100)[0]
    with store.transaction() as conn:
        store.insert_evidence(conn, [replace(fact, content="Changed representation")])
    with store.read() as conn:
        assert store.fetch_evidence(conn, "u", (), limit=100)[0].content == fact.content
    with store.transaction() as conn, pytest.raises(ValueError):
        store.insert_evidence(conn, [replace(fact, user_id="other")])
    with store.transaction() as conn, pytest.raises(ValueError):
        store.insert_evidence(conn, [replace(fact, source_quote="Not in the source")])
    with pytest.raises(PayloadMismatchError):
        apply_batch(
            store, replace(original, messages=(Message("user", "Changed"),)), grounded_evidence=True
        )
    module = importlib.import_module("tianximem.pairing.apply")

    def fail(**kwargs):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(module, "extract_evidence", fail)
    with pytest.raises(RuntimeError):
        add(store, "Alex: I am a patient at Other Clinic.", rid="failed")
    with store.read() as conn:
        assert conn.execute("SELECT count(*) FROM applied_batches").fetchone()[0] == 1


def test_list_count_and_paraphrase_compile_to_the_same_generic_filter():
    questions = (
        "Who are the patients at Maple Clinic?",
        "How many patients are there at Maple Clinic?",
        "List all patients in Maple Clinic.",
    )
    assert len({compile_query(q) for q in questions}) == 1
    assert (
        compile_query(
            "Who is the current head of state in the country of citizenship of Alex?"
        ).operator
        == "walk"
    )
