from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tianximem.common.tokens import O200kCounter
from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.query import FactPattern
from tianximem.pairing import AddBatch, Message, PayloadMismatchError, apply_batch
from tianximem.retrieve import EvidenceChecker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore


def _mail(name="Alex Reader", claim="An update."):
    return (
        f"document: Message-ID: message\nFrom: {name} <alex@example.com>\n"
        "Date: 2024-03-30\nSubject: Update\n\nHello. "
        f"{claim}\n\nRegards,\n{name}\nSpecialist\nExample Labs\n\nEmail: alex@example.com"
    )


def _add(store, rid, text, **flags):
    return apply_batch(store, AddBatch(rid, "u", "s", (Message("user", text, None),)), **flags)


def _pipeline(store, **flags):
    return SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: False),
        retriever=None,
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        **flags,
    )


def _raw(store):
    with store.read() as conn:
        return tuple(
            tuple(tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"))
            for table in ("qa_pairs", "applied_batches")
        )


def test_legacy_derived_tables_are_ignored_and_preserved_when_common_index_is_rebuilt(tmp_path):
    path = tmp_path / "legacy.db"
    store = SqliteStore.open(path)
    _add(store, "old", "Alex: I am a patient at Clinic.")
    with store.transaction() as conn:
        parent = store.iter_pairs(conn)[0]
        conn.execute("CREATE TABLE grounded_evidence(id TEXT PRIMARY KEY,content TEXT NOT NULL)")
        conn.execute("INSERT INTO grounded_evidence VALUES (?,?)", ("legacy", "Wrong old claim"))
        conn.execute(
            "CREATE TABLE fact_index_coverage(parent_memory_id TEXT,user_id TEXT,"
            "index_name TEXT,version TEXT)"
        )
        conn.execute(
            "INSERT INTO fact_index_coverage VALUES (?,?,?,?)",
            (parent.id, "u", "grounded_evidence", EVIDENCE_VERSION),
        )
    before = _raw(store)
    reopened = SqliteStore.open(path)
    hits = (
        _pipeline(reopened, grounded_evidence=True)
        .run(user_id="u", query="Who are the patients at Clinic?", top_k=100)
        .items
    )
    assert len(hits) == 1 and "Alex" in hits[0].content and "Wrong old claim" not in hits[0].content
    assert _raw(reopened) == before
    with reopened.read() as conn:
        assert [tuple(r) for r in conn.execute("SELECT * FROM grounded_evidence")] == [
            ("legacy", "Wrong old claim")
        ]
        assert [tuple(r) for r in conn.execute("SELECT * FROM fact_index_coverage")] == [
            (parent.id, "u", "grounded_evidence", EVIDENCE_VERSION)
        ]
        assert reopened.has_fact_index_coverage(conn, "u", version=EVIDENCE_VERSION)


CASES = [
    (
        "grounded_evidence",
        [
            "Alex Reader: I am a patient at North Community.",
            "Bea Writer: I am a patient at North Community.",
        ],
        "Who are the patients at North Community?",
    ),
    (
        "friend_contexts",
        ["Alex: I had dinner with my church friends."],
        "Where has Alex made friends?",
    ),
    ("employment_facts", [_mail(), _mail("Bea Writer")], "Who are the employees of Example Labs?"),
    (
        "linked_source_statements",
        [
            "Corpus: Example Book — creator — Old Writer",
            "Corpus: Example Book was created by New Writer (this replaces the earlier value)",
            "Corpus: New Writer was born in the city of New City (this replaces the earlier value)",
        ],
        "In what city was the creator of Example Book born?",
    ),
    (
        "scoped_temporal_statements",
        ["Source: Alice Reader works for Old School from Jan, 1990 to Jan, 1995."],
        "Which employer did Alice Reader work for in Jan, 1993?",
    ),
    (
        "inventory_facts",
        [
            "User: My 20-gallon tank, which has 10 neon tetras, 5 gouramis, "
            "and a small pleco catfish."
        ],
        "How many fish do I have in total in my tanks?",
    ),
    (
        "work_observations",
        [_mail(claim="I've commenced my role at Example Labs, and I am settling in.")],
        "Who joined Example Labs in March 2024?",
    ),
    (
        "status_observations",
        ["Alex: My online clothing store is open!"],
        "When did Alex open his online clothing store?",
    ),
    (
        "participation_observations",
        ["Alex: Yesterday, I went to a fair to show my shop."],
        "Which events has Alex participated in to promote his business?",
    ),
    (
        "personal_references",
        [
            "Alex: A gift from my grandma in my home country, New Zealand.",
            "Alex: I've known these friends for 3 years, since I moved from my home country.",
        ],
        "Where did Alex move from 3 years ago?",
    ),
]


@pytest.mark.parametrize("mode,texts,query", CASES)
def test_cold_search_backfills_same_facts_as_new_add_without_changing_sources(
    tmp_path, mode, texts, query
):
    legacy = SqliteStore.open(tmp_path / "legacy.db")
    fresh = SqliteStore.open(tmp_path / "fresh.db")
    for position, text in enumerate([*texts, "Unrelated note without any extractable facts."]):
        _add(legacy, f"opaque:{position}", text)
        _add(fresh, f"opaque:{position}", text, grounded_evidence=True)
    before = _raw(legacy)
    expected = _pipeline(fresh, grounded_evidence=True).run(user_id="u", query=query, top_k=100)
    assert expected.items
    pipeline = _pipeline(legacy, grounded_evidence=True)
    assert pipeline.run(user_id="u", query=query, top_k=100) == expected
    assert _raw(legacy) == before
    with legacy.read() as conn:
        assert legacy.has_fact_index_coverage(conn, "u", version=EVIDENCE_VERSION)
        assert (
            conn.execute("SELECT COUNT(*) FROM evidence_coverage").fetchone()[0] == len(texts) + 1
        )
    assert pipeline.run(user_id="u", query=query, top_k=100) == expected
    assert not pipeline.run(user_id="other", query=query, top_k=100).items


def test_old_sources_plus_new_add_are_not_reported_as_partial_roster(tmp_path):
    store = SqliteStore.open(tmp_path / "partial.db")
    _add(store, "old", _mail())
    _add(store, "new", _mail("Bea Writer"), grounded_evidence=True)
    before = _raw(store)
    result = _pipeline(store, grounded_evidence=True).run(
        user_id="u", query="Who are the employees of Example Labs?", top_k=100
    )
    assert len(result.items) == 2
    assert {
        name
        for name in ("Alex Reader", "Bea Writer")
        if any(name in item.content for item in result.items)
    } == {"Alex Reader", "Bea Writer"}
    assert _raw(store) == before


def test_bounded_backfill_never_returns_partial_fact_set_and_skips_empty_sources(tmp_path):
    store = SqliteStore.open(tmp_path / "bounded.db")
    for rid, text in [("a", _mail()), ("b", _mail("Bea Writer")), ("c", "Unrelated note.")]:
        _add(store, rid, text)
    pipeline = _pipeline(store, grounded_evidence=True, fact_backfill_limit=1)
    query = "Who are the employees of Example Labs?"
    for expected in (1, 2):
        assert not pipeline.run(user_id="u", query=query, top_k=100).items
        with store.read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM evidence_coverage").fetchone()[0] == expected
    assert len(pipeline.run(user_id="u", query=query, top_k=100).items) == 2
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence_coverage").fetchone()[0] == 3


def test_backfill_failure_rolls_back_facts_and_coverage_but_keeps_original_batches(
    tmp_path, monkeypatch
):
    store = SqliteStore.open(tmp_path / "failed.db")
    _add(store, "a", _mail())
    _add(store, "b", _mail("Bea Writer"))
    before = _raw(store)
    module = importlib.import_module("tianximem.pairing.apply")
    original = module.extract_evidence
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("extraction failed")
        return original(**kwargs)

    monkeypatch.setattr(module, "extract_evidence", fail_second)
    with pytest.raises(RuntimeError, match="extraction failed"):
        _pipeline(store, grounded_evidence=True).run(
            user_id="u", query="Who are the employees of Example Labs?", top_k=100
        )
    assert _raw(store) == before
    with store.read() as conn:
        for table in ("memory_facts", "evidence_coverage"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_stale_versions_cannot_supply_a_roster_or_skip_current_scan(tmp_path):
    store = SqliteStore.open(tmp_path / "versions.db")
    _add(store, "a", _mail(), grounded_evidence=True)
    with store.transaction() as conn:
        fact = store.fetch_evidence(conn, "u", (FactPattern(("employee",)),), limit=100)[0]
        pair = store.fetch_pairs_by_ids(conn, [fact.parent_memory_id])[0]
        conn.execute("DELETE FROM memory_facts")
        conn.execute("DELETE FROM evidence_coverage")
        store.insert_evidence(conn, [replace(fact, id="stale", version="retired-v0")])
        store.mark_fact_index_coverage(conn, pair, version="retired-v0")
        assert not store.fetch_evidence(conn, "u", (FactPattern(("employee",)),), limit=100)
    result = _pipeline(store, grounded_evidence=True).run(
        user_id="u", query="Who are the employees of Example Labs?", top_k=100
    )
    assert len(result.items) == 1 and result.items[0].id == fact.id


def test_read_snapshot_stays_consistent_when_an_add_commits(tmp_path):
    store = SqliteStore.open(tmp_path / "snapshot.db")
    _add(store, "a", _mail(), grounded_evidence=True)
    kwargs = {"version": EVIDENCE_VERSION}
    with store.read_snapshot() as conn:
        assert store.has_fact_index_coverage(conn, "u", **kwargs)
        _add(store, "b", _mail("Bea Writer"))
        assert store.has_fact_index_coverage(conn, "u", **kwargs)
        assert len(store.fetch_evidence(conn, "u", (FactPattern(("employee",)),), limit=100)) == 1
    with store.read_snapshot() as conn:
        assert not store.has_fact_index_coverage(conn, "u", **kwargs)


def test_coverage_checks_source_owner_and_respects_payload_mismatch(tmp_path):
    store = SqliteStore.open(tmp_path / "owner.db")
    _add(store, "a", _mail())
    with store.transaction() as conn:
        pair = store.fetch_by_request(conn, "u", "s", "a")[0]
        with pytest.raises(ValueError, match="其他用户"):
            store.mark_fact_index_coverage(conn, replace(pair, user_id="other"), version="v")
    with pytest.raises(PayloadMismatchError):
        _add(store, "a", _mail("Changed Person"), grounded_evidence=True)
    with store.read() as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence_coverage").fetchone()[0] == 0
