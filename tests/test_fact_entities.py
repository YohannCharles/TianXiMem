from __future__ import annotations

import pytest
from tools.fact_entity_smoke import (
    evaluate,
    evaluate_gold_comparison,
    evaluate_model_fact_precision,
    load_cases,
)

from tianximem.facts.entities import SpacyEntityExtractor
from tianximem.facts.grammar import extract_evidence

spacy = pytest.importorskip("spacy")


def test_entity_sidecar_keeps_model_spans_and_rule_numbers() -> None:
    nlp = spacy.blank("en")
    extractor = SpacyEntityExtractor(nlp, model_name="blank-en")

    result = extractor.extract("Alice works for Acme Labs and sold 12 cups for $3.50 each.")

    assert result.backend == "rules"
    assert {entity.text for entity in result.entities} >= {
        "Alice",
        "Acme Labs",
        "12",
        "$3.50",
    }
    for entity in result.entities:
        source = "Alice works for Acme Labs and sold 12 cups for $3.50 each."
        assert source[entity.start : entity.end] == entity.text
        assert entity.normalized


def test_model_entities_are_preserved_for_fact_validation() -> None:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "PERSON", "pattern": "Alice"}])
    extractor = SpacyEntityExtractor(nlp, model_name="test-ruler")

    result = extractor.extract("Alice works for Acme Labs.")

    alice = next(entity for entity in result.entities if entity.text == "Alice")
    assert alice.label == "PERSON"
    assert alice.source == "model"
    assert result.backend == "spacy+rules"


def test_model_sidecar_ignores_unattributed_answer_text() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    facts = extract_evidence(
        parent_memory_id="m1",
        user_id="u1",
        question="What does Alice do?",
        answer="Alice works for Acme Labs.",
        event_time=None,
        entity_extractor=extractor,
    )

    assert not any(fact.get("model_entity") for fact in facts)


def test_model_sidecar_accepts_explicit_human_answer_speaker() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    facts = extract_evidence(
        parent_memory_id="m1",
        user_id="u1",
        question="What does Alice do?",
        answer="Alice: Alice works for Acme Labs.",
        event_time=None,
        entity_extractor=extractor,
    )

    assert any(
        fact.subject == "Alice"
        and fact.relation == "employee of"
        and fact.object == "Acme Labs"
        and fact.source_side == "answer"
        and fact.source_quote == "Alice works for Acme Labs."
        for fact in facts
    )


def test_no_entity_model_still_has_a_deterministic_fallback() -> None:
    nlp = spacy.blank("en")
    extractor = SpacyEntityExtractor(nlp)

    result = extractor.extract("The team met in Paris on 2025-01-03.")

    assert any(entity.text == "Paris" for entity in result.entities)
    assert any(entity.text == "2025-01-03" for entity in result.entities)
    assert result.relations == ()


def test_subject_line_keeps_the_full_title_span() -> None:
    nlp = spacy.blank("en")
    extractor = SpacyEntityExtractor(nlp)

    result = extractor.extract("Subject: Working Session: consumer feedback integration")

    title = next(entity for entity in result.entities if entity.label == "RULE_TITLE")
    assert title.text == "Working Session: consumer feedback integration"


@pytest.mark.parametrize(
    ("text", "relation", "subject", "object"),
    [
        ("Alice works for Acme Labs.", "employee", "Alice", "Acme Labs"),
        ("Alice studied at Oxford University.", "student", "Alice", "Oxford University"),
    ],
)
def test_surface_relations_recover_common_prepositional_facts(
    text: str, relation: str, subject: str, object: str
) -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    candidates = extractor.extract(text).relations

    assert any(
        candidate.relation == relation
        and candidate.subject.text == subject
        and candidate.object is not None
        and candidate.object.text == object
        for candidate in candidates
    )


def test_composite_model_entities_keep_full_object_span() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    result = extractor.extract(
        "David Thewlis is a citizen of United Kingdom of Great Britain and Ireland."
    )

    assert any(
        relation.relation == "employee" or relation.relation == "student"
        for relation in result.relations
    ) is False
    assert any(
        entity.text == "United Kingdom of Great Britain and Ireland"
        for entity in result.entities
    )


def test_structured_profile_lines_recover_employee_relation() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    candidates = extractor.extract(
        "Jurre Price\nMarketing Specialist\nZenith Labs"
    ).relations

    assert any(
        candidate.relation == "employee"
        and candidate.subject.text == "Jurre Price"
        and candidate.object is not None
        and candidate.object.text == "Zenith Labs"
        and candidate.source == "structured-rule"
        for candidate in candidates
    )


def test_structured_profile_lines_ignore_cross_line_model_spans() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    result = extractor.extract(
        "Jared Chartier\nDepartment Manager, Sales & Marketing\nSales & Marketing | Zenith Labs"
    )

    assert any(entity.text == "Jared Chartier" for entity in result.entities)
    assert any(entity.text == "Zenith Labs" for entity in result.entities)
    assert any(
        relation.relation == "employee"
        and relation.subject.text == "Jared Chartier"
        and relation.object is not None
        and relation.object.text == "Zenith Labs"
        for relation in result.relations
    )


def test_structured_profile_extracts_org_suffix_from_one_line_title() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    result = extractor.extract("Mohammed Austermuhle\nMarketing Specialist, Zenith Labs")

    assert any(
        relation.relation == "employee"
        and relation.subject.text == "Mohammed Austermuhle"
        and relation.object is not None
        and relation.object.text == "Zenith Labs"
        for relation in result.relations
    )


def test_gold_jsonl_can_score_exact_fact_triples(tmp_path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        '{"id":"employee","text":"Alice works for Acme Labs.",'
        '"entities":["Alice","Acme Labs"],'
        '"relations":[{"subject":"Alice","relation":"employee","object":"Acme Labs"}]}\n',
        encoding="utf-8",
    )
    report = evaluate(SpacyEntityExtractor(spacy.load("en_core_web_sm")), load_cases(gold))

    assert report["relation_precision"] == 1.0
    assert report["relation_recall"] == 1.0
    assert report["relation_f1"] == 1.0


def test_gold_comparison_reports_rules_model_and_combined() -> None:
    cases = (
        {
            "id": "employee",
            "text": "Alice works for Acme Labs.",
            "entities": (),
            "relations": (("Alice", "employee", "Acme Labs"),),
        },
    )

    report = evaluate_gold_comparison(
        SpacyEntityExtractor(spacy.load("en_core_web_sm")), cases
    )

    # 规则层现在也能抽出这条（2026-10-18 扩了事件/角色动词表）——这条断言原本
    # 钉的是"规则抽不出、只有 spaCy 抽得出"，那个前提已经不成立。
    assert report["rules"]["f1"] == 1.0
    assert report["model_sidecar"]["f1"] == 1.0
    assert report["combined"]["f1"] == 1.0


def test_model_fact_precision_uses_same_source_reference(tmp_path) -> None:
    database = tmp_path / "facts.db"
    connection = __import__("sqlite3").connect(database)
    connection.executescript(
        """
        CREATE TABLE qa_pairs (id TEXT PRIMARY KEY, question TEXT, answer TEXT);
        CREATE TABLE memory_facts (
            parent_memory_id TEXT, subject TEXT, relation TEXT, object TEXT, source_side TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO qa_pairs VALUES (?, ?, ?)",
        ("m1", "Who works for Acme Labs?", "Alice: Alice works for Acme Labs."),
    )
    connection.execute(
        "INSERT INTO memory_facts VALUES (?, ?, ?, ?, ?)",
        ("m1", "Alice", "employee of", "Acme Labs", "answer"),
    )
    connection.commit()
    connection.close()

    report = evaluate_model_fact_precision(
        SpacyEntityExtractor(spacy.load("en_core_web_sm")), database, 5
    )

    assert report["reference_only"] is True
    assert report["predicted"] == report["hits"] == 1
    assert report["precision"] == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "Does Alice work for Acme Labs?",
        "Alice does not work for Acme Labs.",
        "Alice plans to work for Acme Labs.",
        "Alice said: Alice works for Acme Labs.",
    ],
)
def test_surface_relations_do_not_emit_unsafe_assertions(text: str) -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    assert extractor.extract(text).relations == ()


def test_surface_relation_rejects_discourse_let_me_know_phrase() -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    assert extractor.extract(
        "Let me know what works for your schedule—I'm pretty flexible."
    ).relations == ()


@pytest.mark.parametrize(
    ("text", "subject"),
    [
        (
            "Corpus: Father Brown was created by Andy Breckman "
            "(this replaces the earlier value)",
            "Father Brown",
        ),
        (
            "Corpus: Louis X of France was born in the city of Denison "
            "(this replaces the earlier value)",
            "Louis X of France",
        ),
    ],
)
def test_corpus_model_relation_keeps_full_subject(text: str, subject: str) -> None:
    extractor = SpacyEntityExtractor(spacy.load("en_core_web_sm"))

    facts = extract_evidence(
        parent_memory_id="m1",
        user_id="u1",
        question=text,
        answer=None,
        event_time=None,
        entity_extractor=extractor,
    )

    assert any(fact.subject == subject for fact in facts)
