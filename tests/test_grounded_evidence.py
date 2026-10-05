from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tianximem.common.tokens import O200kCounter
from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import FactPattern, compile_query
from tianximem.pairing import AddBatch, Message, PayloadMismatchError, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore


def batch(*texts, rid="opaque:request", user="u", timestamp=1756684800000):
    return AddBatch(rid, user, "s", tuple(Message("user", text, timestamp) for text in texts))


def pipeline(store, *, limit=12, backfill=1024):
    def search(*, user_id, query, top_k):
        with store.read() as conn:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM qa_pairs WHERE user_id=? ORDER BY request_id,local_index",
                    (user_id,),
                )
            ]
        return [Candidate(pid, i) for i, pid in enumerate(ids)]

    return SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=search),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
        evidence_limit=limit,
        fact_backfill_limit=backfill,
        radius=0,
        inject_abs_time=True,
    )


def facts(store, scope, user="u"):
    with store.read() as conn:
        return store.fetch_evidence(conn, user, (scope,), limit=100)


@pytest.mark.parametrize(
    "role,organization",
    [
        ("patient", "Maple Clinic"),
        ("coach", "North Athletics"),
        ("member", "Ridge Club"),
    ],
)
def test_roles_are_source_fields_not_company_specific_categories(tmp_path, role, organization):
    store = SqliteStore.open(tmp_path / "test.db")
    original = f"Rosa Quinn: I am a {role} at {organization}."
    apply_batch(store, batch(original), grounded_evidence=True)
    plural = "coaches" if role == "coach" else role + "s"
    list_query = f"Who are the {plural} at {organization}?"
    count_query = f"How many {plural} are there at {organization}?"
    service = pipeline(store)
    listed = service.run(user_id="u", query=list_query, top_k=100)
    counted = service.run(user_id="u", query=count_query, top_k=100)
    assert listed.items == counted.items
    assert len(listed.items) == 1
    assert "Statement: Rosa Quinn" in listed.items[0].content
    assert f"is a {role} " in listed.items[0].content
    assert "employee" not in listed.items[0].content
    assert "Source quotation: I am a " in listed.items[0].content


@pytest.mark.parametrize(
    "statement",
    [
        "I work for North Labs.",
        "I am a volunteer at North Labs.",
        "I study at North Labs.",
        "I belong to North Labs.",
    ],
)
def test_source_phrase_and_question_use_the_same_relation_binding(statement):
    extracted = extract_evidence(
        parent_memory_id="p",
        user_id="u",
        question="Rosa Quinn: " + statement,
        answer=None,
        event_time=None,
    )
    assert len(extracted) == 1
    expected = {
        "work": "employee",
        "volunteer": "volunteer",
        "study": "student",
        "belong": "member",
    }
    role = next(value for token, value in expected.items() if token in statement)
    assert compile_query(f"Who are the {role}s at North Labs?").patterns == (
        FactPattern((role,), object="north labs"),
    )
    assert extracted[0].subject == "Rosa Quinn"
    assert extracted[0].object == "North Labs"
    assert extracted[0].source_quote == statement
    assert "Source record date: \n" in extracted[0].content


def test_inventory_keeps_owner_container_quantity_and_original_item_order(tmp_path):
    store = SqliteStore.open(tmp_path / "test.db")
    source = "Rosa Quinn: My orange case contains two brushes, three pencils, and one eraser."
    apply_batch(
        store,
        batch(source, source, "Evan Cole: My orange case contains nine pencils."),
        grounded_evidence=True,
    )
    apply_batch(
        store,
        batch("Rosa Quinn: My orange case contains 99 pencils.", rid="other", user="other"),
        grounded_evidence=True,
    )
    service = pipeline(store)
    query = "How many items are in Rosa Quinn's orange case in total?"
    hits = service.run(user_id="u", query=query, top_k=100).items
    assert len(hits) == 3
    assert [
        hit.content.split(" contains ")[1].split(" ", 1)[1].split(".", 1)[0] for hit in hits
    ] == [
        "brushes",
        "pencils",
        "eraser",
    ]
    assert all("Statement: Rosa Quinn" in hit.content for hit in hits)
    assert all(
        "contains 99 " not in hit.content and "contains 9 " not in hit.content for hit in hits
    )
    assert all("contains 6 " not in hit.content for hit in hits)
    assert all(hit.created_at == "2025-09-01" for hit in hits)
    assert len(service.run(user_id="u", query=query, top_k=1).items) == 1
    assert not service.run(user_id="unknown", query=query, top_k=100).items


def test_inventory_plan_with_unbought_items_does_not_negate_existing_owned_items(tmp_path):
    store = SqliteStore.open(tmp_path / "test.db")
    apply_batch(
        store,
        batch(
            "Rosa Quinn: My orange case contains two brushes, three pencils, and one eraser.",
            "Rosa Quinn: I plan to add four more pencils to the orange case next week. "
            "I have not bought them.",
        ),
        grounded_evidence=True,
    )
    hits = (
        pipeline(store)
        .run(
            user_id="u", query="How many items are in Rosa Quinn's orange case in total?", top_k=100
        )
        .items
    )
    assert len(hits) == 3
    assert all("Memory fact" in hit.content and "four more" not in hit.content for hit in hits)


@pytest.mark.parametrize(
    "text",
    [
        "Rosa Quinn: I plan to volunteer at North Shelter next month.",
        "Rosa Quinn: I am not a volunteer at North Shelter.",
        "Rosa Quinn: A fictional example said 'Evan Cole volunteers at North Shelter.'",
        "Rosa Quinn: Evan Cole is a volunteer at North Shelter.",
        "Rosa Quinn: My orange case contains two brushes and some pencils.",
        "Rosa Quinn: My orange case contains two brushes and 10001 pencils.",
    ],
)
def test_uncertain_quoted_other_person_and_partial_inventory_are_not_positive_facts(text):
    assert not extract_evidence(
        parent_memory_id="p",
        user_id="u",
        question=text,
        answer=None,
        event_time=1,
    )
    assert not extract_evidence(
        parent_memory_id="p",
        user_id="u",
        question=None,
        answer="[assistant] Assistant: " + text,
        event_time=1,
    )


@pytest.mark.parametrize(
    "additional",
    [
        "Rosa Quinn: My orange case contains two brushes and some pencils.",
        "Rosa Quinn: My orange case contains five brushes.",
    ],
)
def test_unknown_quantities_and_conflicting_updates_fall_back_to_full_originals(
    tmp_path, additional
):
    store = SqliteStore.open(tmp_path / "test.db")
    apply_batch(
        store,
        batch("Rosa Quinn: My orange case contains two brushes.", additional),
        grounded_evidence=True,
    )
    hits = (
        pipeline(store)
        .run(
            user_id="u", query="How many items are in Rosa Quinn's orange case in total?", top_k=100
        )
        .items
    )
    assert len(hits) == 2
    assert all(" Q: " in hit.content for hit in hits)
    assert any(additional in hit.content for hit in hits)


def test_departure_and_unrecognized_member_preserve_originals(tmp_path):
    for index, additional in enumerate(
        [
            "Rosa Quinn: I am no longer a member of Ridge Club.",
            "Evan Cole: Ridge Club accepted my registration yesterday.",
        ]
    ):
        store = SqliteStore.open(tmp_path / f"test{index}.db")
        apply_batch(
            store,
            batch("Rosa Quinn: I am a member of Ridge Club.", additional),
            grounded_evidence=True,
        )
        hits = (
            pipeline(store)
            .run(user_id="u", query="How many members are there in Ridge Club?", top_k=100)
            .items
        )
        assert len(hits) == 2 and all(" Q: " in hit.content for hit in hits)


@pytest.mark.parametrize(
    "additional",
    [
        "Evan Cole: I plan to volunteer at North Shelter. "
        "I am already a volunteer at North Shelter.",
        "Evan Cole: A fictional example concerns Ridge Club. I volunteer at North Shelter.",
    ],
)
def test_plan_or_quote_does_not_hide_a_real_claim_in_the_same_record(tmp_path, additional):
    store = SqliteStore.open(tmp_path / "test.db")
    apply_batch(
        store,
        batch("Rosa Quinn: I volunteer at North Shelter.", additional),
        grounded_evidence=True,
    )
    hits = (
        pipeline(store).run(user_id="u", query="Who volunteers at North Shelter?", top_k=100).items
    )
    assert len(hits) == 2 and all(" Q: " in hit.content for hit in hits)


def test_replay_backfill_and_source_bounds_never_change_raw_memory(tmp_path):
    store = SqliteStore.open(tmp_path / "test.db")
    original = batch(
        "Rosa Quinn: I am a member of Ridge Club.", "Evan Cole: I am a member of Ridge Club."
    )
    apply_batch(store, original)
    scope = FactPattern(("member",), object="ridge club")
    assert not facts(store, scope)
    with store.read() as conn:
        raw_before = [tuple(row) for row in conn.execute("SELECT * FROM qa_pairs ORDER BY id")]
    service = pipeline(store, backfill=1)
    query = "Who are the members of Ridge Club?"
    assert all(
        " Q: " in hit.content for hit in service.run(user_id="u", query=query, top_k=100).items
    )
    assert all(
        "Memory fact" in hit.content
        for hit in service.run(user_id="u", query=query, top_k=100).items
    )
    assert not apply_batch(store, original, grounded_evidence=True).applied
    with store.read() as conn:
        assert raw_before == [
            tuple(row) for row in conn.execute("SELECT * FROM qa_pairs ORDER BY id")
        ]
    assert len(facts(store, scope)) == 2
    assert all(
        " Q: " in hit.content
        for hit in pipeline(store, limit=1)
        .run(
            user_id="u",
            query=query,
            top_k=100,
        )
        .items
    )
    with pytest.raises(PayloadMismatchError):
        apply_batch(store, batch("changed"), grounded_evidence=True)
    with pytest.raises(ValueError), store.transaction() as conn:
        store.insert_evidence(conn, [replace(facts(store, scope)[0], user_id="other")])
    with pytest.raises(ValueError), store.transaction() as conn:
        store.insert_evidence(conn, [replace(facts(store, scope)[0], source_quote="invented")])


@pytest.mark.parametrize(
    "query",
    [
        "How many members joined Ridge Club last year?",
        "Who are the former members of Ridge Club?",
        "Who should volunteer at North Shelter?",
        "Who are the members of Ridge Club except Rosa Quinn?",
    ],
)
def test_temporal_hypothetical_and_exclusion_queries_are_not_routed(query):
    assert compile_query(query) is None
