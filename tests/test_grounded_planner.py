from __future__ import annotations

import json

from tianximem.facts.query import QueryPlan
from tianximem.service.grounded_planner import GroundedPlanner


class FakeLlm:
    name = "fake-qwen"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 8, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self.reply


def test_planner_accepts_a_safe_unseen_filter_surface() -> None:
    llm = FakeLlm(
        json.dumps(
            {
                "operator": "filter",
                "relations": ["employee"],
                "subject": "",
                "object": "Zenith Labs",
            }
        )
    )
    plan = GroundedPlanner(llm).compile("Which staff members are associated with Zenith Labs?")
    assert isinstance(plan, QueryPlan)
    assert plan.operator == "filter"
    assert plan.patterns[0].relations == ("employee",)
    assert plan.patterns[0].object == "zenith labs"
    assert len(llm.calls) == 1


def test_planner_accepts_a_bounded_relationship_walk() -> None:
    llm = FakeLlm(
        '{"operator":"walk","root":"Book","relations":["author","place of birth"]}'
    )
    plan = GroundedPlanner(llm).compile("Where was the author of Book born?")
    assert plan is not None
    assert plan.operator == "walk"
    assert plan.walk_root == "book"
    assert plan.walk_relations == ("author", "place of birth")
    assert plan.require_complete_walk


def test_planner_rejects_hallucinated_literals_and_malformed_output() -> None:
    llm = FakeLlm(
        '{"operator":"filter","relations":["employee"],"subject":"","object":"Other Labs"}'
    )
    assert GroundedPlanner(llm).compile("Who works for Zenith Labs?") is None

    malformed = GroundedPlanner(FakeLlm("not json"))
    assert malformed.compile("Who works for Zenith Labs?") is None


def test_planner_does_not_call_llm_for_unsafe_scope() -> None:
    llm = FakeLlm("null")
    planner = GroundedPlanner(llm)
    assert planner.compile("Who did not work for Zenith Labs?") is None
    assert llm.calls == []


def test_planner_rejects_extra_fields_instead_of_dropping_constraints() -> None:
    llm = FakeLlm(
        '{"operator":"filter","relations":["employee"],"subject":"","object":"Zenith Labs",'
        '"date":"2024"}'
    )
    assert GroundedPlanner(llm).compile("Who worked for Zenith Labs in 2024?") is None


def test_planner_rejects_a_relation_outside_the_whitelist() -> None:
    llm = FakeLlm(
        '{"operator":"filter","relations":["invented relation"],"subject":"Alice","object":""}'
    )
    assert GroundedPlanner(llm).compile("What did Alice invent?") is None


def test_planner_carries_a_verbatim_period_as_local_bounds() -> None:
    question = "Did direct report meetings take place about testing and validation checkpoint in Q1 of 2024?"
    llm = FakeLlm(
        '{"operator":"filter","relations":["meeting topic"],"subject":"",'
        '"object":"testing and validation checkpoint","period":"Q1 of 2024"}'
    )
    decision = GroundedPlanner(llm).decide(question)
    assert decision.status == "planned" and decision.plan is not None
    assert decision.plan.date_bounds == ("2024-01-01", "2024-04-01")
    # 同一条事实在每次会议里各算一条证据（"哪几次"），不得按三元组去重。
    assert decision.plan.per_occurrence


def test_planner_rejects_a_period_that_is_not_in_the_question_or_not_a_date() -> None:
    question = "Which meetings about roadmap review took place in Q1 of 2024?"
    invented = FakeLlm(
        '{"operator":"filter","relations":["meeting topic"],"subject":"",'
        '"object":"roadmap review","period":"in Q2 of 2024"}'
    )
    assert GroundedPlanner(invented).decide(question).status == "invalid_period"
    unparseable = FakeLlm(
        '{"operator":"filter","relations":["meeting topic"],"subject":"",'
        '"object":"roadmap review","period":"recently"}'
    )
    assert GroundedPlanner(unparseable).decide(question).status == "invalid_period"


def test_planner_refuses_a_plan_that_drops_a_time_restriction() -> None:
    question = "Which meetings about roadmap review took place during Q1 of 2024?"
    llm = FakeLlm(
        '{"operator":"filter","relations":["meeting topic"],"subject":"",'
        '"object":"roadmap review"}'
    )
    decision = GroundedPlanner(llm).decide(question)
    assert decision.plan is None and decision.status == "missing_period"

