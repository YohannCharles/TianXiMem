"""Check explicit employment predicates before changing query compilation."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from tianximem.facts import query
from tianximem.facts.grammar import extract_evidence
from tianximem.retrieve.evidence import select_evidence


def prototype():
    source = Path(query.__file__).read_text()
    old = r'("employee", r"\bemployer\b|\bworkplace\b|\bwork(?:s)? for\b"),'
    new = (
        r'("employee", r"\bemployer\b|\bworkplace\b|\bwork(?:s)? (?:for|at)\b|'
        r'\bwhere\s+(?:do|does)\s+.+?\s+work$"),'
    )
    assert source.count(old) == 1
    module = types.ModuleType("work_query_prototype")
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new), "<work_query_prototype>", "exec"), module.__dict__)
    return module


def main():
    proposed = prototype()
    sources = (
        "Corpus: Alex — employer — Example Labs",
        "Corpus: Ann Druyan — spouse — Carl Sagan",
        "Corpus: Carl Sagan — employer — Cornell University",
    )
    facts = [
        f
        for i, source in enumerate(sources)
        for f in extract_evidence(
            parent_memory_id=str(i),
            user_id="manual",
            question=source,
            answer=None,
            event_time=None,
        )
    ]
    cases = (
        ("Where does Alex work?", {"employee"}, {"0"}),
        ("Who works at Example Labs?", {"employee"}, {"0"}),
        ("Where does the spouse of Ann Druyan work?", {"employee", "spouse"}, {"1", "2"}),
        (
            "What notable work is the author of Book known for?",
            {"author", "notable work"},
            None,
        ),
        ("Who is the creator of the artwork of Alex?", {"creator"}, None),
        ("Where does Alex not work?", None, None),
        ("Where did Alex work before 2020?", None, None),
        ("Who was Alex's previous employer?", None, None),
        ("How often does Alex work out with his family?", None, None),
        (
            "When does winter begin where Alex and Fran meet people who travel for work?",
            None,
            None,
        ),
        ("What industry does Alex work in?", None, None),
    )
    result = []
    for question, expected_relations, expected_sources in cases:
        plan = proposed.compile_query(question)
        relations = None if plan is None else set(plan.patterns[0].relations)
        selected = (
            None if plan is None else select_evidence(facts, plan, source_limit=12, hop_limit=4)
        )
        source_ids = None if selected is None else {f.parent_memory_id for f in selected.facts}
        passed = relations == expected_relations and source_ids == expected_sources
        result.append(
            {
                "question": question,
                "expected_relations": None
                if expected_relations is None
                else sorted(expected_relations),
                "actual_relations": None if relations is None else sorted(relations),
                "expected_sources": None if expected_sources is None else sorted(expected_sources),
                "actual_sources": None if source_ids is None else sorted(source_ids),
                "passed": passed,
            }
        )
        print(question, "PASS" if passed else "FAIL", flush=True)
    missing = select_evidence(
        facts[:2],
        proposed.compile_query("Where does the spouse of Ann Druyan work?"),
        source_limit=12,
        hop_limit=4,
    )
    result.append({"case": "missing_employment_source_falls_back", "passed": missing is None})
    print("missing_employment_source_falls_back", "PASS" if missing is None else "FAIL")
    Path(__file__).with_name("prototype-work-query.json").write_text(
        json.dumps({"cases": result, "literal_sources": sources}, ensure_ascii=False, indent=2)
        + "\n"
    )
    return int(not all(r["passed"] for r in result))


if __name__ == "__main__":
    raise SystemExit(main())
