"""A fast personalization prototype: exact self-employment questions need audit."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from tianximem.facts.grammar import extract_evidence
from tianximem.facts.query import FactPattern, QueryPlan, compile_query
from tianximem.retrieve.evidence import evaluate_evidence

HERE = Path(__file__).resolve().parent


def proposed_query(text):
    if re.fullmatch(r"(?:who|what) is my employer[?.!]?", text.strip(), re.I):
        return QueryPlan("filter", (FactPattern(("employee",), subject="i"),), "source")
    return compile_query(text)


def main():
    cases = (
        ("explicit_self", "Who is my employer?", ["User: I work for Acme."], ["0"]),
        ("third_person_is_not_me", "Who is my employer?", ["Corpus: Alex — employer — Acme"], None),
        ("assistant_is_not_me", "Who is my employer?", ["Assistant: I work for Acme."], None),
        (
            "a_colleague_is_not_me",
            "Who is my colleague's employer?",
            ["User: I work for Acme."],
            None,
        ),
        (
            "unknown_job_change_must_keep_the_original_chain",
            "Who is my employer?",
            ["User: I work for Acme.", "User: I left Acme and took a role at NewCo."],
            None,
        ),
        (
            "negative_old_job_must_not_become_current",
            "Who is my employer?",
            ["User: I work for Acme.", "User: I no longer work for Acme."],
            None,
        ),
    )
    results = []
    for name, text, sources, expected in cases:
        facts = [
            f
            for i, source in enumerate(sources)
            for f in extract_evidence(
                parent_memory_id=str(i),
                user_id="manual",
                question=source,
                answer=None,
                event_time=int(datetime(2026, 10, 5 + i, tzinfo=UTC).timestamp() * 1000),
            )
        ]
        plan = proposed_query(text)
        result = (
            None if plan is None else evaluate_evidence(facts, plan, source_limit=12, hop_limit=4)
        )
        actual = (
            None
            if result is None or result.selection is None
            else sorted({f.parent_memory_id for f in result.selection.facts})
        )
        passed = expected == actual
        print(name, "PASS" if passed else "FAIL", flush=True)
        results.append(
            dict(
                case=name,
                question=text,
                sources=sources,
                expected=expected,
                actual=actual,
                passed=passed,
            )
        )
    (HERE / "prototype-self-employer.json").write_text(
        json.dumps(dict(manual=results, product_modified=False), ensure_ascii=False, indent=2)
        + "\n"
    )
    if not all(r["passed"] for r in results):
        raise RuntimeError("Unsafe partial personalization evidence; reject before product changes")


if __name__ == "__main__":
    main()
