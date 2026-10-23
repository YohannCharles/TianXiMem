"""Grounded paths return original records without widening merged context."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tianximem.common.render import render_pair
from tianximem.facts.query import compile_query
from tianximem.observability import SnapshotMetricsSink
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.service.search_pipeline import SearchPipeline
from tianximem.service.search_stages import GroundedEvidenceStage
from tianximem.store.sqlite_store import make_source_message_id

QUERY = "Where does the spouse of Alex work?"
ROOT = "Corpus: Alex — spouse — Blair"
OLD = "Corpus: Blair — employer — Old Labs"
NEW = "Corpus: Blair is employed by New Labs (this replaces the earlier value)"
DISTRACTOR = "Corpus: Other Person — employer — Other Labs"
DAY = 1756684800000


def add(store, *texts, rid="first", user="u", dates=None):
    return apply_batch(
        store,
        AddBatch(
            rid,
            user,
            "s",
            tuple(
                Message("user", text, (dates[i] if dates else DAY)) for i, text in enumerate(texts)
            ),
        ),
        grounded_evidence=True,
    )


def stage(store, limit=12):
    return GroundedEvidenceStage(
        store=store, evidence_limit=limit, evidence_hop_limit=4, fact_backfill_limit=1024
    )


def service(store, counter, tmp_path, *, budget=117760):
    calls = []

    def search(*, user_id, query, top_k):
        calls.append(query)
        with store.read() as conn:
            pairs = [p for p in store.iter_pairs(conn) if p.user_id == user_id]
        return [Candidate(p.id, i) for i, p in enumerate(pairs)]

    metrics = SnapshotMetricsSink(tmp_path / "metrics.json")
    pipeline = SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=search),
        checker=EvidenceChecker(),
        counter=counter,
        budget_tokens=budget,
        grounded_evidence=True,
        radius=2,
        inject_abs_time=True,
        metrics=metrics,
    )
    return pipeline, calls


def test_walk_keeps_hop_order_original_dates_and_excludes_old_branch(store, counter, tmp_path):
    # Source order differs from path order; all originals occupy one merged block.
    add(store, OLD, DISTRACTOR, ROOT, dates=[DAY, DAY, DAY + 86400000])
    add(store, NEW, rid="update", dates=[DAY + 2 * 86400000])
    add(store, "Corpus: Blair — employer — Private Labs", user="other", rid="private")
    pipeline, calls = service(store, counter, tmp_path)
    response = pipeline.run(user_id="u", query=QUERY, top_k=100)
    assert calls == []
    assert [i.content for i in response.items] == [
        "[2025-09-02] Q: " + ROOT,
        "[2025-09-03] Q: " + NEW,
    ]
    assert [i.created_at for i in response.items] == ["2025-09-02", "2025-09-03"]
    decision = stage(store).decide("u", compile_query(QUERY))
    assert decision.selection.projection == "source_record"
    assert decision.selection.source_indices == (2, 0)
    assert [i.id for i in response.items] == [
        make_source_message_id(f.parent_memory_id, index)
        for f, index in zip(
            decision.selection.facts, decision.selection.source_indices, strict=True
        )
    ]
    with store.read() as conn:
        originals = [p for p in store.iter_pairs(conn) if p.user_id == "u"]
    assert len(originals) == 2
    assert OLD in originals[0].question or OLD in originals[1].question


def test_record_limit_counts_messages_even_when_the_parent_is_one_block(store):
    add(store, ROOT, OLD)
    decision = stage(store, limit=1).decide("u", compile_query(QUERY))
    assert decision.selection is None
    assert decision.reason == "source_limit"


def test_unknown_related_statement_in_selected_parent_still_requires_fallback(
    store, counter, tmp_path
):
    add(store, ROOT, OLD, "Blair: I no longer work for Old Labs.")
    pipeline, calls = service(store, counter, tmp_path)
    response = pipeline.run(user_id="u", query=QUERY, top_k=100)
    assert calls == [QUERY]
    assert len(response.items) == 1
    assert "Blair: I no longer work for Old Labs." in response.items[0].content


@pytest.mark.parametrize("legacy", [False, True])
def test_ambiguous_or_missing_boundaries_keep_original_parent_projection(store, legacy):
    add(store, ROOT, ROOT, OLD)
    if legacy:
        with store.transaction() as conn:
            conn.execute("DELETE FROM qa_pair_messages")
    decision = stage(store).decide("u", compile_query(QUERY))
    assert decision.selection.projection == "source"
    assert decision.selection.source_indices == ()


def test_repeated_selected_fact_deduplicates_by_record_identity(store, counter, tmp_path):
    add(store, ROOT, OLD)
    selection = stage(store).decide("u", compile_query(QUERY)).selection
    pipeline, _ = service(store, counter, tmp_path)
    repeated = replace(
        selection,
        facts=(selection.facts[0], selection.facts[0], selection.facts[1]),
        source_indices=(0, 0, 1),
    )
    response = pipeline._package_evidence(repeated, top_k=100)
    assert len(response.items) == 2
    assert len({item.id for item in response.items}) == 2


def test_top_k_overflow_falls_back_instead_of_returning_a_partial_path(store, counter, tmp_path):
    add(store, ROOT, OLD)
    pipeline, calls = service(store, counter, tmp_path)
    response = pipeline.run(user_id="u", query=QUERY, top_k=1)
    assert calls == [QUERY]
    assert len(response.items) == 1
    assert ROOT in response.items[0].content and OLD in response.items[0].content


def test_budget_overflow_does_not_short_circuit_with_a_partial_path(store, counter, tmp_path):
    add(store, ROOT, OLD)
    pipeline, calls = service(store, counter, tmp_path, budget=50)
    pipeline.run(user_id="u", query=QUERY, top_k=100)
    assert calls == [QUERY]


def test_source_resolution_rechecks_missing_records(store, counter, tmp_path):
    add(store, ROOT, OLD)
    selection = stage(store).decide("u", compile_query(QUERY)).selection
    with store.transaction() as conn:
        conn.execute("DELETE FROM qa_pair_messages")
    pipeline, _ = service(store, counter, tmp_path)
    packaged = pipeline._package_evidence(selection, top_k=100)
    assert packaged.items == () and packaged.dropped_missing


def test_narrative_retrieval_retains_merged_blocks(store, counter, tmp_path):
    add(
        store,
        "Alex: The moonlit scene closes the first act.",
        "Blair: The next act begins in the same room.",
    )
    pipeline, calls = service(store, counter, tmp_path)
    response = pipeline.run(user_id="u", query="Describe the scene transitions.", top_k=100)
    assert calls == ["Describe the scene transitions."]
    with store.read() as conn:
        originals = [render_pair(p, inject_abs_time=True) for p in store.iter_pairs(conn)]
    assert [item.content for item in response.items] == originals


def test_multiline_json_updates_keep_the_literal_marker_for_each_hop(store, counter, tmp_path):
    def record(subject, relation, obj, update=False):
        prefix = (
            "Corpus: UPDATE: replace the prior value of this subject and relation "
            "with the following fact.\n"
            if update
            else "Corpus: "
        )
        return prefix + json.dumps({"subject": subject, "relation": relation, "object": obj})

    add(store, record("Alex", "spouse", "Old Blair"), record("Old Blair", "employer", "Old Labs"))
    selected = [
        record("Alex", "spouse", "Blair", True),
        record("Blair", "employer", "New Labs", True),
    ]
    add(store, *reversed(selected), rid="update", dates=[DAY + 86400000] * 2)
    pipeline, calls = service(store, counter, tmp_path)
    response = pipeline.run(user_id="u", query=QUERY, top_k=100)
    assert calls == []
    assert [item.content for item in response.items] == [
        "[2025-09-02] Q: " + source for source in selected
    ]
