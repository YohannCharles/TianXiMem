"""Prototype speaker/owned-container rules without modifying product files."""

from __future__ import annotations

import json
import re
from pathlib import Path

from tianximem.facts import conversation, grammar
from tianximem.facts.evidence import key, relation_key
from tianximem.facts.query import FactPattern, QueryPlan, compile_query
from tianximem.retrieve.evidence import select_evidence

OUT = Path(__file__).resolve().parent


def prototype_extractor():
    convo = {"__name__": "prototype_conversation"}
    source = Path(conversation.__file__).read_text()
    source = source.replace("(?P<actor>{NAME})", "(?P<actor>(?:(?i:user|assistant|system)|{NAME}))")
    source = source.replace("?{NAME}[:：]", "?(?:(?i:user|assistant|system)|{NAME})[:：]")
    exec(compile(source, "prototype_conversation", "exec"), convo)
    source = Path(grammar.__file__).read_text()
    source = source.replace(
        'actor in {"Assistant", "助手", "系统"}',
        'key(actor) in {"assistant", "system", "助手", "系统"}',
    )
    source = source.replace(
        'actor in {"User", "用户"} and side == "question"',
        'key(actor) in {"user", "用户"} and side == "question"',
    )
    source = source.replace(
        'if speaker and actor != "I" and safe_attributes', "if speaker and safe_attributes"
    )
    namespace = {"__name__": "prototype_grammar"}
    exec(compile(source, "prototype_grammar", "exec"), namespace)
    namespace.update(SPEAKER=convo["SPEAKER"], named_utterances=convo["named_utterances"])
    return namespace["extract_evidence"]


def prototype_query(query):
    text = query.strip().rstrip("?.!")
    match = re.fullmatch(r"How many (.+?) (?:are(?: there)?|do I have) in my (.+)", text, re.I)
    if match:
        noun = relation_key(match[1])
        return QueryPlan(
            "filter",
            (
                FactPattern(
                    ("contain",),
                    subject="i",
                    object_terms=rf"\b{re.escape(noun)}s?\b",
                    container_terms=rf"\b{re.escape(key(match[2]))}\b",
                ),
            ),
            guard_literal=key(match[2]),
        )
    return compile_query(query)


def main():
    extract = prototype_extractor()
    tests = [
        ("user_own_inventory", "user: My tank has 2 fish.", {("I", "fish")}),
        ("uppercase_user_inventory", "USER: My tank has 2 fish.", {("I", "fish")}),
        ("assistant_is_not_owner", "assistant: My tank has 100 fish.", set()),
        ("system_is_not_owner", "system: My tank has 100 fish.", set()),
        ("explicit_user_role", "user: I am an engineer at Example Labs.", {("I", "Example Labs")}),
        ("named_human_role", "Alex: I am an engineer at Example Labs.", {("Alex", "Example Labs")}),
        (
            "quote_cannot_rebind_user",
            'Alex: I read this quote: "\nuser: My tank has 100 fish.',
            set(),
        ),
        ("planned_inventory", "user: I will put 100 fish in my tank.", set()),
    ]
    results = []
    for name, source, expected in tests:
        facts = extract(
            parent_memory_id=name, user_id="u", question=source, answer=None, event_time=None
        )
        actual = {(f.subject, f.object) for f in facts}
        passed = actual == expected
        results.append(
            {
                "case": name,
                "source": source,
                "expected": sorted(expected),
                "actual": sorted(actual),
                "passed": passed,
            }
        )
        print(name, actual, "PASS" if passed else "FAIL")
    facts = extract(
        parent_memory_id="p",
        user_id="u",
        question="user: My tank has 2 fish.",
        answer=None,
        event_time=None,
    )
    question = "How many fish are in my tank?"
    original = select_evidence(facts, compile_query(question), source_limit=12, hop_limit=4)
    selected = select_evidence(facts, prototype_query(question), source_limit=12, hop_limit=4)
    passed = original is None and selected is not None and selected.facts[0].get("quantity") == 2
    results.append(
        {
            "case": "owned_container_question",
            "source": "user: My tank has 2 fish.",
            "question": question,
            "baseline_selection": original is not None,
            "prototype_selection": selected is not None,
            "passed": passed,
        }
    )
    print("owned_container_question", "PASS" if passed else "FAIL")
    (OUT / "prototype-ownership.json").write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2)
    )
    return int(not all(result["passed"] for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
