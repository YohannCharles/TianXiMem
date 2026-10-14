"""Evidence paths must preserve dependencies, direction, and source identity."""

from dataclasses import replace

import pytest

from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import FactPattern, QueryPlan, compile_query
from tianximem.retrieve.evidence import evaluate_evidence


def decide(question, statements, *, hops=4, shared_parent=False):
    facts = [
        fact
        for index, statement in enumerate(statements)
        for fact in extract_evidence(
            parent_memory_id="shared" if shared_parent else str(index),
            user_id="u",
            question="Corpus: " + statement,
            answer=None,
            event_time=None,
        )
    ]
    if shared_parent:
        # Distinct positions within one QA produce distinct fact IDs.
        facts = [replace(fact, id=f"{index}:{fact.id}") for index, fact in enumerate(facts)]
    plan = compile_query(question)
    assert plan is not None
    return evaluate_evidence(facts, plan, source_limit=12, hop_limit=hops)


def test_ordered_path_does_not_accept_a_reverse_edge():
    question = "What is the country of citizenship of the author of Atlas?"
    complete = decide(question, ["Atlas — author — Mira", "Mira — country of citizenship — Nation"])
    assert complete.selection and len(complete.selection.facts) == 2
    incomplete = decide(
        question, ["Atlas — author — Mira", "Nation — country of citizenship — Mira"]
    )
    assert incomplete.selection is None


def test_each_branch_requires_the_next_step():
    result = decide(
        "Where was the author of Atlas born?",
        [
            "Atlas — author — Mira",
            "Atlas — author — Rowan",
            "Mira — place of birth — City",
        ],
    )
    assert result.selection is None


def test_repeated_relation_is_executed_twice_even_when_it_returns_to_root():
    plan = compile_query("Who is the spouse of the spouse of Mira?")
    assert plan.walk_relations == ("spouse", "spouse")
    result = decide(
        "Who is the spouse of the spouse of Mira?",
        [
            "Mira — spouse — Rowan",
            "Rowan — spouse — Mira",
        ],
    )
    assert result.selection and len(result.selection.facts) == 2


def test_limit_rejects_the_whole_path_instead_of_returning_a_prefix():
    result = decide(
        "Who is the spouse of the spouse of Mira?",
        [
            "Mira — spouse — Rowan",
            "Rowan — spouse — Mira",
        ],
        hops=1,
    )
    assert result.selection is None


def test_two_edges_from_the_same_qa_remain_available_to_completeness_audit():
    result = decide(
        "Where was the author of Atlas born?",
        [
            "Atlas — author — Mira",
            "Mira — place of birth — City",
        ],
        shared_parent=True,
    )
    assert result.selection and len(result.selection.facts) == 2


def test_keyword_only_legacy_plan_cannot_claim_completeness():
    plan = QueryPlan("walk", (FactPattern(("spouse", "place of birth")),), query="Mira")
    result = evaluate_evidence([], plan, source_limit=12, hop_limit=4)
    assert result.selection is None


def test_personal_language_is_distinct_from_official_language():
    plan = compile_query(
        "Which language is the native tongue of the person who is married to Mira?"
    )
    assert plan.walk_relations == ("spouse", "languages spoken, written or signed")


@pytest.mark.parametrize(
    "question",
    [
        "Who is the current leader of the country of citizenship of Mira?",
        "Where was the author of Atlas born before 1990?",
    ],
)
def test_ambiguous_or_unparsed_constraints_do_not_short_circuit(question):
    assert compile_query(question) is None
