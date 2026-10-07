"""Incomplete literal relation walks must hand control back to hybrid retrieval."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tianximem.common.render import render_pair
from tianximem.common.tokens import O200kCounter
from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import compile_query
from tianximem.observability import SnapshotMetricsSink
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.retrieve.evidence import evaluate_evidence
from tianximem.service.pipeline import SearchPipeline

QUERY = "Where does the spouse of Alex work?"
SOURCES = {
    "citizenship": "Corpus: Alex — spouse — Blair",
    "leader": "Corpus: Blair — employer — Example Labs",
    "distractor": "Corpus: Other Person — employer — Other Labs",
    "cycle": "Corpus: Blair — spouse — Alex",
    "edit": "Corpus: Alex is married to New Blair (this replaces the earlier value)",
    "new_leader": "Corpus: New Blair — employer — New Labs",
}


@pytest.mark.parametrize(
    "names,hops",
    [
        (("citizenship",), 4),
        (("citizenship", "distractor"), 4),
        (("citizenship", "leader"), 1),
        (("citizenship", "cycle"), 4),
        (("citizenship", "leader", "edit"), 4),
    ],
)
def test_missing_relation_cannot_short_circuit_with_a_partial_path(names, hops):
    facts = [
        fact
        for name in names
        for fact in extract_evidence(
            parent_memory_id=name,
            user_id="u",
            question=SOURCES[name],
            answer=None,
            event_time=None,
        )
    ]
    result = evaluate_evidence(facts, compile_query(QUERY), source_limit=12, hop_limit=hops)
    assert result.selection is None
    assert result.reason == "unsupported_walk"


@pytest.mark.parametrize("complete", [False, True])
def test_search_pipeline_falls_back_only_when_the_literal_path_is_incomplete(
    store, tmp_path, complete
):
    names = ["citizenship", "distractor"]
    if complete:
        names.append("leader")
    apply_batch(
        store,
        AddBatch("opaque", "u", "s", tuple(Message("user", SOURCES[n]) for n in names)),
        grounded_evidence=True,
    )
    hybrid_calls = []

    def retrieve(*, user_id, query, top_k):
        hybrid_calls.append((user_id, query, top_k))
        with store.read() as conn:
            return [Candidate(p.id, i) for i, p in enumerate(store.iter_pairs(conn))]

    metrics = SnapshotMetricsSink(tmp_path / "metrics.json")
    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=retrieve),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
        metrics=metrics,
        radius=0,
    )
    response = pipeline.run(user_id="u", query=QUERY, top_k=100)
    with store.read() as conn:
        originals = {render_pair(p) for p in store.iter_pairs(conn)}
    assert response.items and all(item.content in originals for item in response.items)
    if complete:
        assert hybrid_calls == []
        assert len(response.items) == 2
        assert not any("Other Person" in item.content for item in response.items)
    else:
        assert hybrid_calls == [("u", QUERY, 100)]
        assert {item.content for item in response.items} == originals


def test_employment_predicate_binds_the_second_relation_and_preserves_its_source():
    facts = [
        f
        for i, source in enumerate(
            (
                "Corpus: Ann Druyan — spouse — Carl Sagan",
                "Corpus: Carl Sagan — employer — Cornell University",
            )
        )
        for f in extract_evidence(
            parent_memory_id=str(i),
            user_id="u",
            question=source,
            answer=None,
            event_time=None,
        )
    ]
    plan = compile_query("Where does the spouse of Ann Druyan work?")
    assert set(plan.patterns[0].relations) == {"spouse", "employee"}
    result = evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
    assert result.selection is not None
    assert {f.parent_memory_id for f in result.selection.facts} == {"0", "1"}
    assert evaluate_evidence(facts[:1], plan, source_limit=12, hop_limit=4).selection is None


@pytest.mark.parametrize(
    "query,relations",
    [
        ("Where does Alex work?", {"employee"}),
        ("Who works at Example Labs?", {"employee"}),
        ("What notable work is the author of Book known for?", {"author", "notable work"}),
        ("Who is the creator of the artwork of Alex?", {"creator"}),
    ],
)
def test_employment_predicate_does_not_treat_an_artists_work_as_an_employer(query, relations):
    plan = compile_query(query)
    assert plan is not None
    assert set(plan.patterns[0].relations) == relations


@pytest.mark.parametrize(
    "query",
    [
        "How often does Alex work out with his family?",
        "When does winter begin where Alex and Fran meet people who travel for work?",
        "What industry does Alex work in?",
    ],
)
def test_unrecognized_work_predicates_do_not_short_circuit_with_employment_sources(query):
    assert compile_query(query) is None


@pytest.mark.parametrize(
    "question,sources,fallback",
    [
        ("Where do I work?", ["User: I work for Acme."], False),
        (
            "Where do I work?",
            ["User: I work for Acme.", "User: I no longer work for Acme."],
            True,
        ),
        (
            "Where do I work?",
            ["User: I work for Acme.", "User: I left Acme and took a role at NewCo."],
            True,
        ),
        (
            "Where does Alex work?",
            ["Alex: I work for Acme.", "Alex: I no longer work for Acme."],
            True,
        ),
        (
            "Where do I work?",
            ["User: I work for Acme.", "Bob: I no longer work for OtherCo."],
            False,
        ),
        (
            "Where do I work?",
            ["User: I work for Acme.", "Assistant: I no longer work for OtherCo."],
            False,
        ),
        (
            "Where do I work?",
            ["User: I work for Acme.", "My previous employer was Acme. I have left."],
            True,
        ),
        (
            "Where does Alex work?",
            ["Alex: I work for Acme.\nAlex: I may change jobs next month."],
            False,
        ),
    ],
)
def test_walk_preserves_unparsed_related_declarations_and_speaker_ownership(
    store, question, sources, fallback
):
    apply_batch(
        store,
        AddBatch("opaque", "u", "s", tuple(Message("user", text) for text in sources)),
        grounded_evidence=True,
    )
    calls = []

    def retrieve(*, user_id, query, top_k):
        calls.append((user_id, query, top_k))
        with store.read() as conn:
            return [Candidate(p.id, i) for i, p in enumerate(store.iter_pairs(conn))]

    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=retrieve),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
        radius=0,
    )
    response = pipeline.run(user_id="u", query=question, top_k=100)
    with store.read() as conn:
        originals = {render_pair(p) for p in store.iter_pairs(conn)}
    returned = {item.content for item in response.items}
    assert returned and returned <= originals
    assert bool(calls) == fallback
    if fallback:
        assert calls == [("u", question, 100)]
        assert returned == originals
    else:
        assert len(response.items) == 1


def test_walk_audit_overflow_returns_all_original_evidence_in_hybrid(store):
    sources = ["Alex: I work for Acme."] + [
        f"Alex: I may change jobs after milestone {i}." for i in range(97)
    ]
    apply_batch(
        store,
        AddBatch("opaque", "u", "s", tuple(Message("user", text) for text in sources)),
        grounded_evidence=True,
    )
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        with store.read() as conn:
            return [Candidate(p.id, i) for i, p in enumerate(store.iter_pairs(conn))]

    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=retrieve),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
        radius=0,
    )
    assert pipeline._grounded_decision("u", compile_query("Where does Alex work?")).reason == (
        "source_scan_limit"
    )
    response = pipeline.run(user_id="u", query="Where does Alex work?", top_k=100)
    assert len(calls) == 1
    assert len(response.items) == len(sources)


def test_walk_audit_and_fact_selection_share_one_snapshot_and_execute_once(store, monkeypatch):
    import tianximem.service.pipeline as module

    apply_batch(
        store,
        AddBatch("opaque", "u", "s", (Message("user", "User: I work for Acme."),)),
        grounded_evidence=True,
    )
    connections, decisions = [], []
    for name in ("fetch_evidence", "fetch_evidence_sources"):
        original = getattr(type(store), name)

        def record(self, conn, *args, _original=original, **kwargs):
            assert conn.in_transaction
            connections.append(conn)
            return _original(self, conn, *args, **kwargs)

        monkeypatch.setattr(type(store), name, record)
    original_evaluate = module.evaluate_evidence

    def evaluate(*args, **kwargs):
        decisions.append(1)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_evidence", evaluate)
    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=lambda **kwargs: pytest.fail("unexpected fallback")),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
    )
    assert pipeline.run(user_id="u", query="Where do I work?", top_k=100).items
    assert len(connections) > 1 and all(conn is connections[0] for conn in connections)
    assert decisions == [1]


def test_legacy_walk_keeps_literal_evidence_for_controlled_knowledge_completion():
    facts = extract_evidence(
        parent_memory_id="citizenship",
        user_id="u",
        question="Corpus: Alex — country of citizenship — State",
        answer=None,
        event_time=None,
    )
    plan = compile_query("Who is the head of government of the country of citizenship of Alex?")
    assert not plan.require_complete_walk
    selected = evaluate_evidence(facts, plan, source_limit=12, hop_limit=4).selection
    assert selected is not None and selected.facts == facts


def test_passive_employment_edit_replaces_the_old_literal_edge_and_keeps_source():
    sources = (
        "Corpus: Ann Druyan — spouse — Carl Sagan",
        "Corpus: Carl Sagan — employer — Cornell University",
        "Corpus: Carl Sagan is employed by BBC (this replaces the earlier value)",
    )
    facts = [
        f
        for i, text in enumerate(sources)
        for f in extract_evidence(
            parent_memory_id=str(i), user_id="u", question=text, answer=None, event_time=None
        )
    ]
    plan = compile_query("Where does the spouse of Ann Druyan work?")
    selected = evaluate_evidence(facts, plan, source_limit=12, hop_limit=4).selection
    assert selected is not None
    assert {f.parent_memory_id for f in selected.facts} == {"0", "2"}
    assert next(f for f in selected.facts if f.parent_memory_id == "2").source_quote == sources[2]


@pytest.mark.parametrize(
    "body",
    [
        "Alex is not employed by Acme",
        "If Alex is employed by Acme",
        "Alex is employed by Acme if approved",
        "Alex will be employed by Acme",
        'Example reads "Alex is employed by Acme"',
        "Alex is employed by unknown",
    ],
)
def test_unsafe_passive_employment_edits_cannot_become_positive_edges(body):
    from tianximem.facts.evidence import relation_key

    text = f"Corpus: {body} (this replaces the earlier value)"
    facts = extract_evidence(
        parent_memory_id="edit", user_id="u", question=text, answer=None, event_time=None
    )
    assert not any(relation_key(f.relation) == "employee" for f in facts)
    assert any(f.get("unparsed") and f.source_quote == text for f in facts)


@pytest.mark.parametrize(
    "ordinary_time,parents",
    [
        (None, {"edit", "ordinary"}),
        (1735689600000, {"edit"}),
        (1767225600000, {"edit", "ordinary"}),
        (1798761600000, {"edit", "ordinary"}),
    ],
)
def test_dated_replacement_cannot_suppress_a_claim_with_unknown_source_time(ordinary_time, parents):
    facts = [
        f
        for parent, text, moment in (
            (
                "edit",
                "Corpus: Alex is employed by Acme (this replaces the earlier value)",
                1767225600000,
            ),
            ("ordinary", "Alex: I currently work for NewCo.", ordinary_time),
        )
        for f in extract_evidence(
            parent_memory_id=parent, user_id="u", question=text, answer=None, event_time=moment
        )
    ]
    result = evaluate_evidence(
        facts, compile_query("Where does Alex work?"), source_limit=12, hop_limit=4
    )
    assert result.selection is not None
    assert {f.parent_memory_id for f in result.selection.facts} == parents
