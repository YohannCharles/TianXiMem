"""Completed-sale provenance, event preservation, fallback and annotation purity."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from tianximem.common.amounts import annotate_unit_total
from tianximem.common.config import ConfigError, load_config
from tianximem.common.tokens import O200kCounter
from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import compile_query
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore

QUERY = "What is the total amount of money I earned from selling my products at the markets?"
SOURCES = (
    "user: I sold 15 jars of jam at the Handmade Market on May 29th, earning $225.",
    "user: I just sold 20 herb plants at the Solstice Market for $7.5 each.",
    "user: I even sold 12 bunches of herbs at the farmers' market on May 15th, "
    "earning a total of $120.",
)


def service(store, *, annotate=False, backfill=1024, budget=117760):
    def fallback(*, user_id, **_):
        with store.read() as conn:
            return [
                Candidate(p.id, i) for i, p in enumerate(store.iter_pairs(conn, user_id=user_id))
            ]

    return SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=fallback),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=budget,
        grounded_evidence=True,
        annotate_unit_totals=annotate,
        fact_backfill_limit=backfill,
    )


def add(store, text, *, user="u", rid="one", indexed=True):
    apply_batch(
        store,
        AddBatch(rid, user, "s", (Message("user", text, 1685577600000),)),
        grounded_evidence=indexed,
    )


def test_sales_are_literal_records_and_unit_annotation_is_search_only(tmp_path):
    store = SqliteStore.open(tmp_path / "sales.db")
    for i, text in enumerate(SOURCES):
        add(store, text, rid=str(i))
    add(store, "user: I sold 3 cups at the Pottery Market for $99 each.", user="other", rid="other")
    plain = service(store).run(user_id="u", query=QUERY, top_k=100)
    annotated = service(store, annotate=True).run(user_id="u", query=QUERY, top_k=100)
    assert len(plain.items) == len(annotated.items) == 3
    assert [x.id for x in plain.items] == [x.id for x in annotated.items]
    for a, b in zip(plain.items, annotated.items, strict=True):
        assert (a.created_at, a.score) == (b.created_at, b.score)
        before, after = json.loads(a.content), json.loads(b.content)
        assert before["user_quotation"] in "\n".join(SOURCES)
        after["user_quotation"] = re.sub(
            r" \(calculated sale total: [^)]+\)", "", after["user_quotation"]
        )
        assert after == before
        assert "$495" not in b.content and "$99" not in b.content
    assert sum("20 × $7.5 = $150" in x.content for x in annotated.items) == 1
    with store.read() as conn:
        assert all(
            "calculated sale total" not in f.content
            for f in store.fetch_evidence(conn, "u", (), limit=100)
        )
        assert all(
            "calculated sale total" not in (p.question or "") for p in store.iter_pairs(conn)
        )


def test_separate_sales_of_same_product_are_not_latest_state_quantities(tmp_path):
    store = SqliteStore.open(tmp_path / "events.db")
    add(store, "user: I sold 3 mugs at the Pottery Market on May 1st, earning $30.", rid="first")
    add(store, "user: I sold 5 mugs at the Pottery Market on May 2nd, earning $50.", rid="second")
    hits = service(store).run(user_id="u", query=QUERY, top_k=100).items
    assert len(hits) == 2
    assert {json.loads(h.content)["quantity"] for h in hits} == {"3", "5"}


@pytest.mark.parametrize(
    "text",
    [
        "user: I sold several plants at the Solstice Market for $7.5 each.",
        "user: I sold 20 packs of 2 plants at the Solstice Market for $7.5 each.",
        "user: I sold 20 plants at the Solstice Market for $7.5 each, but later refunded half.",
        "user: I sold 20 plants at the Solstice Market for $7.5 each; "
        "correction: it was 10 plants.",
    ],
)
def test_unexplained_sale_returns_raw_sources_instead_of_partial_records(tmp_path, text):
    store = SqliteStore.open(tmp_path / "fallback.db")
    add(store, SOURCES[0], rid="good")
    add(store, text, rid="unknown")
    hits = service(store, annotate=True).run(user_id="u", query=QUERY, top_k=100).items
    assert len(hits) == 2
    assert any(text in h.content for h in hits)
    assert all("calculated sale total" not in h.content for h in hits)


@pytest.mark.parametrize(
    "text",
    [
        "I will sell 20 plants at the Solstice Market for $7.5 each.",
        "If I sold 20 plants at the Solstice Market for $7.5 each, would that help?",
        'Example: "I sold 20 plants at the Solstice Market for $7.5 each."',
        "I never sold 20 plants at the Solstice Market for $7.5 each.",
    ],
)
def test_plans_quotes_and_negatives_are_not_sales(text):
    facts = extract_evidence(
        parent_memory_id="p",
        user_id="u",
        question="user: " + text,
        answer="assistant: I sold 20 plants at the Solstice Market for $7.5 each.",
        event_time=None,
    )
    assert all(f.relation != "sale" for f in facts)


def test_incomplete_backfill_and_small_top_k_preserve_contract(tmp_path):
    store = SqliteStore.open(tmp_path / "old.db")
    for i, text in enumerate(SOURCES):
        add(store, text, rid=str(i), indexed=False)
    first = service(store, annotate=True, backfill=1).run(user_id="u", query=QUERY, top_k=100)
    assert len(first.items) == 3 and all(h.content.startswith("Q:") for h in first.items)
    final = service(store, annotate=True).run(user_id="u", query=QUERY, top_k=1)
    assert len(final.items) == 1
    # A partial sale collection cannot certify the full evidence plan.
    assert final.items[0].content.startswith("Q:")


def test_temporal_and_profit_scopes_do_not_use_bare_sales_plan():
    assert compile_query(QUERY).guard_assertion_pattern
    for question in ("How much profit did I earn at the markets?", QUERY[:-1] + " last month?"):
        assert compile_query(question) is None


@pytest.mark.parametrize(
    "update",
    [
        "user: I refunded $50 from that plant sale.",
        "user: Correction: that plant sale earned $100.",
        "user: I cancelled the plant sale.",
    ],
)
def test_separate_transaction_updates_without_market_name_force_raw_fallback(tmp_path, update):
    store = SqliteStore.open(tmp_path / "updates.db")
    add(store, SOURCES[1], rid="sale")
    add(store, update, rid="update")
    hits = service(store, annotate=True).run(user_id="u", query=QUERY, top_k=100).items
    assert len(hits) == 2
    assert any(update in hit.content for hit in hits)
    assert all("calculated sale total" not in hit.content for hit in hits)


def test_annotation_is_counted_in_final_token_budget(tmp_path):
    store = SqliteStore.open(tmp_path / "budget.db")
    add(store, SOURCES[1])
    content = service(store).run(user_id="u", query=QUERY, top_k=1).items[0].content
    budget = O200kCounter().count(content)
    assert len(service(store, budget=budget).run(user_id="u", query=QUERY, top_k=1).items) == 1
    full = service(store, annotate=True).run(user_id="u", query=QUERY, top_k=1)
    assert O200kCounter().count(full.items[0].content) > budget
    limited = service(store, annotate=True, budget=budget).run(user_id="u", query=QUERY, top_k=1)
    # Annotation exceeding the budget triggers whole-path fallback; smaller
    # original evidence may still fit without pretending the fact plan finished.
    assert limited.items and limited.items[0].content.startswith("Q:")
    assert O200kCounter().count(limited.items[0].content) <= budget


def test_decimal_annotation_does_not_silently_round_explicit_unit_price():
    price = "0.12345678901234567890123456789"
    record = {
        "record_kind": "reported sale",
        "quantity": "20",
        "unit_price": price,
        "currency_symbol": "$",
        "user_quotation": f"I sold 20 plants at the Market for ${price} each.",
    }
    annotated = json.loads(annotate_unit_total(json.dumps(record)))
    assert "20 × $" + price + " = $2.4691357802469135780246913578" in annotated["user_quotation"]


def test_annotation_config_is_explicit_and_rejects_non_boolean(tmp_path):
    (tmp_path / "default.yaml").write_text("packaging:\n  annotate_unit_totals: true\n")
    env = {"AML_EMB_BASE_URL": "http://unused/v1", "AML_EMB_API_KEY": "unused"}
    assert load_config(env, config_dir=tmp_path).packaging.annotate_unit_totals
    (tmp_path / "default.yaml").write_text("packaging:\n  annotate_unit_totals: 1\n")
    with pytest.raises(ConfigError, match="annotate_unit_totals"):
        load_config(env, config_dir=tmp_path)
